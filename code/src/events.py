"""Build each user's 90-day forward cash-flow forecast from
financial_events.csv, recurrence detection, and message-derived facts.

Design (see AGENTS.md / problem_statement.md for the source rules):

- `current_available_balance` in financial_profiles.csv is the balance AT
  the request's request_date (each user has exactly one request in this
  dataset, so there is no ambiguity about "as of when"). Historical events
  (event_date < request_date) are therefore never replayed into the
  forecast balance -- they are only used to detect recurring patterns.
- Forward-looking cash impact only comes from: (a) events already dated
  >= request_date in financial_events.csv with an eligible status, and
  (b) recurring series detected from settled history and projected forward,
  amended by any message-derived facts (see messages.py).
- Ignored everywhere: cancelled/failed/unrealized status, non_cash
  direction, and pending credits (reserve pending debits instead).
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from statistics import mean, median
from typing import Callable, Dict, List, Optional

import pandas as pd

from .currency import CurrencyConverter
from .data_loader import Dataset
from .messages import (
    KIND_CONFIRMED_ONE_OFF_INCOME,
    KIND_SALARY_ARREARS,
    KIND_SALARY_BASE_ONLY,
    KIND_SALARY_CONFIRMED,
    KIND_SALARY_NEXT_DATE,
    KIND_SALARY_PERMANENT_RAISE,
    KIND_SALARY_STREAM_ENDED,
    KIND_SALARY_TEMPORARY_AMOUNT,
    MessageFact,
    extract_all_facts,
    latest_salary_facts,
)
from .models import ForecastItem, SpendingChangeOption

FORECAST_HORIZON_DAYS = 90
RECURRENCE_MIN_OCCURRENCES = 3
RECURRENCE_MAX_INTERVAL_DAYS = 45
RECURRENCE_LOOKBACK_DAYS = 400  # ~13 months of history considered for pattern detection
RECENT_WINDOW_FOR_AMOUNT = 6  # use at most this many most-recent occurrences for the amount estimate

_IGNORED_STATUSES = {"cancelled", "failed", "unrealized"}


def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


@dataclass
class RecurringSeries:
    category: str
    interval_days: int
    amount: float  # signed, home currency, conservative estimate
    last_date: date
    anchor_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    active: bool = True
    monthly_day: Optional[int] = None  # set when the series lands on a fixed day-of-month


def _add_months(d: date, months: int, target_day: Optional[int] = None) -> date:
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = target_day or d.day
    day = min(day, _days_in_month(year, month))
    return date(year, month, day)


_AMOUNT_ESTIMATOR = os.environ.get("BUYORWAIT_AMOUNT_ESTIMATOR", "mean")
_EXPENSE_CONSERVATISM = float(os.environ.get("BUYORWAIT_EXPENSE_CONSERVATISM", "1.0"))


def _estimate_amount(observations: List[float]) -> float:
    """Point estimate for a recurring series' per-occurrence amount from its
    recent observations. Strategy is selectable via BUYORWAIT_AMOUNT_ESTIMATOR
    for controlled experimentation (see evaluation/ estimator sweep); default
    is a plain recent-window mean."""
    obs = sorted(observations)
    n = len(obs)
    strategy = _AMOUNT_ESTIMATOR
    if strategy == "median":
        return median(obs)
    if strategy == "trimmed_mean" and n >= 4:
        trimmed = obs[1:-1]
        return mean(trimmed) if trimmed else mean(obs)
    if strategy == "last":
        return observations[-1]
    if strategy == "weighted":
        weights = list(range(1, len(observations) + 1))
        return sum(o * w for o, w in zip(observations, weights)) / sum(weights)
    return mean(obs)


def _split_bimodal(grp: pd.DataFrame, all_amounts: dict):
    """If the group's amounts cleanly separate into two clusters (e.g. two
    concurrent household income earners under one "salary" category), return
    (high_cluster_df, low_cluster_df) each sorted by date; else None.

    Detection: sort amounts, find the single largest relative gap between
    consecutive values, and require it to be a decisive separator (the low
    side of the gap is at least ~1.5x smaller than the high side) with both
    resulting clusters getting a fair, non-trivial share of the points --
    otherwise this is just ordinary variance within one stream, not two
    streams, and must not be split.
    """
    idx_amt = sorted(((i, all_amounts[i]) for i in grp.index), key=lambda t: t[1])
    n = len(idx_amt)
    if n < 4:
        return None
    best_gap_ratio = 0.0
    best_split = None
    for k in range(1, n):
        lo_val = idx_amt[k - 1][1]
        hi_val = idx_amt[k][1]
        if lo_val <= 0:
            continue
        ratio = (hi_val - lo_val) / lo_val
        # both sides must carry a reasonable share of the points -- a split
        # that peels off a single outlier isn't "two streams", it's noise
        # (already handled separately by the recent-median outlier filter).
        if min(k, n - k) < max(2, n // 3):
            continue
        if ratio > best_gap_ratio:
            best_gap_ratio = ratio
            best_split = k
    if best_split is None or best_gap_ratio < 0.15:
        return None
    low_idx = [i for i, _ in idx_amt[:best_split]]
    high_idx = [i for i, _ in idx_amt[best_split:]]
    return grp.loc[high_idx].sort_values("_date"), grp.loc[low_idx].sort_values("_date")


def _step_forward(series: "RecurringSeries", d: date) -> date:
    """Next occurrence date strictly after d."""
    if series.monthly_day is not None:
        return _add_months(d, 1, series.monthly_day)
    return d + timedelta(days=series.interval_days)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days


class EventEngine:
    def __init__(self, ds: Dataset, converter: CurrencyConverter, resolve_amount: Callable[[pd.Series], Optional[float]]):
        self._ds = ds
        self._fx = converter
        self._resolve_amount = resolve_amount

    def _home_amount(self, row, home_currency: str) -> Optional[float]:
        amt = row["amount"]
        if pd.isna(amt):
            amt = self._resolve_amount(row)
            if amt is None:
                return None
        when = row["settlement_date"] if not pd.isna(row.get("settlement_date")) else row["event_date"]
        try:
            converted = self._fx.convert(float(amt), row["currency"], home_currency, when)
        except ValueError:
            return None
        return converted

    def build_forecast(self, user_id: str, request_date_raw) -> "UserForecast":
        profile = self._ds.profile(user_id)
        home_currency = profile["home_currency"]
        request_date = _to_date(request_date_raw)
        horizon_end = request_date + timedelta(days=FORECAST_HORIZON_DAYS)

        events = self._ds.user_events(user_id).copy()
        events["_date"] = events["event_date"].apply(_to_date)
        events["_settle_date"] = events.apply(
            lambda r: _to_date(r["settlement_date"]) if not pd.isna(r["settlement_date"]) else _to_date(r["event_date"]),
            axis=1,
        )

        history = events[
            (events["_date"] < request_date)
            & (events["status"] == "settled")
            & (events["direction"] != "non_cash")
        ].sort_values("_date")

        forward_explicit_rows = []
        for _, row in events.iterrows():
            eff_date = row["_settle_date"]
            status = row["status"]
            if status in _IGNORED_STATUSES or row["direction"] == "non_cash":
                continue
            if status == "pending":
                if row["direction"] != "debit":
                    continue
                eff_date = max(eff_date, request_date)
            elif status == "scheduled":
                pass
            elif status == "settled":
                if eff_date < request_date:
                    continue  # already reflected in current_available_balance
            else:
                continue
            if eff_date < request_date or eff_date > horizon_end:
                continue
            home_amt = self._home_amount(row, home_currency)
            if home_amt is None:
                continue  # never silently treat an unresolved amount as zero
            signed = home_amt if row["direction"] == "credit" else -home_amt
            forward_explicit_rows.append(
                ForecastItem(
                    date=eff_date,
                    amount=signed,
                    category=row["category"],
                    series_key=f"event:{row['category']}",
                    source="explicit",
                    event_id=row["event_id"],
                    can_stop=False,
                    can_reduce=False,
                    minimum_allowed_amount=None,
                    anchor_event_id=row["event_id"],
                )
            )

        message_facts = extract_all_facts(self._ds.user_messages(user_id))

        # Scheduled events (any date) are confirmed facts, even when they're
        # the only two data points establishing a new recurring salary (e.g.
        # a "prorated first salary" settled event plus a "next confirmed
        # salary" scheduled event for a brand-new job).
        scheduled_any_date = events[
            (events["status"] == "scheduled") & (events["direction"] != "non_cash")
        ]
        pattern_source = pd.concat([history, scheduled_any_date]).sort_values("_date")

        series_by_category = self._detect_recurring_series(pattern_source, home_currency, request_date)
        self._apply_income_message_facts(series_by_category, message_facts, home_currency, request_date)

        projected = self._project_series(series_by_category, request_date, horizon_end, forward_explicit_rows)

        one_off_income = self._one_off_income_from_messages(message_facts, home_currency, request_date, horizon_end)

        items = forward_explicit_rows + projected + one_off_income
        items.sort(key=lambda it: it.date)

        change_options = self._build_spending_change_options(series_by_category, profile, request_date, horizon_end)

        return UserForecast(
            user_id=user_id,
            request_date=request_date,
            horizon_end=horizon_end,
            home_currency=home_currency,
            starting_balance=float(profile["current_available_balance"]),
            minimum_balance=float(profile["minimum_balance_to_keep"]),
            items=items,
            change_options=change_options,
        )

    # -- recurrence detection -------------------------------------------------
    def _detect_recurring_series(self, history: pd.DataFrame, home_currency: str, request_date: date) -> Dict[str, RecurringSeries]:
        cutoff = request_date - timedelta(days=RECURRENCE_LOOKBACK_DAYS)
        recent = history[history["_date"] >= cutoff]

        result: Dict[str, RecurringSeries] = {}
        eligible_types = {"expense", "subscription", "income", "debt_payment"}
        for category, grp in recent.groupby("category"):
            grp = grp[grp["event_type"].isin(eligible_types)].sort_values("_date")
            if grp.empty:
                continue
            # A description like "Final employer payroll" / "Final salary
            # payment" on the most recent occurrence means the stream has
            # ended -- don't project it forward even though the amount looks
            # like a normal, in-pattern payment.
            last_description = str(grp.iloc[-1]["description"] or "").lower()
            if "final" in last_description:
                continue

            all_amounts = {}
            for idx, row in grp.iterrows():
                home_amt = self._home_amount(row, home_currency)
                if home_amt is not None:
                    all_amounts[idx] = home_amt
            grp = grp.loc[[i for i in grp.index if i in all_amounts]]
            if grp.empty:
                continue
            is_credit = grp.iloc[-1]["direction"] == "credit"
            min_occurrences = 2 if is_credit else RECURRENCE_MIN_OCCURRENCES

            # Some categories (documented in this dataset as e.g. "household
            # income" with a "Primary household salary" + a "Second household
            # income" earner) genuinely combine two concurrent, independently
            # recurring streams under one category. A single blended fit
            # mis-estimates both the cadence and the amount. Detect this via
            # the simplest defensible signal -- a single large, consistent
            # relative gap splitting the recent amounts into two clusters --
            # and fit each cluster as its own series when both sides
            # independently clear the occurrence/cadence bar.
            primary = self._fit_series_from_group(grp, all_amounts, category, min_occurrences, is_credit)
            secondary = None
            if is_credit and len(grp) >= 2 * min_occurrences:
                split = _split_bimodal(grp, all_amounts)
                if split is not None:
                    grp_hi, grp_lo = split
                    fit_hi = self._fit_series_from_group(grp_hi, all_amounts, category, min_occurrences, is_credit)
                    fit_lo = self._fit_series_from_group(grp_lo, all_amounts, category, min_occurrences, is_credit)
                    if fit_hi is not None and fit_lo is not None:
                        primary, secondary = fit_hi, fit_lo

            if primary is not None:
                result[category] = primary
            if secondary is not None:
                result[f"{category}~2"] = secondary
        return result

    def _fit_series_from_group(
        self, grp: pd.DataFrame, all_amounts: dict, category: str, min_occurrences: int, is_credit: bool
    ) -> Optional[RecurringSeries]:
        if len(grp) < min_occurrences:
            return None

        # A category can mix a genuinely recurring stream (e.g. monthly
        # "Payroll credit") with irregular one-off entries sharing the same
        # category (a quarterly bonus, an arrears adjustment), or even a
        # regime change (a raise, a new employer at a different rate). With
        # enough data points, drop amount outliers -- more than 30% away
        # from the *recent* median (so a sustained new level, not just the
        # historically dominant one, wins) -- before fitting cadence/amount,
        # so one odd entry can't distort it.
        recent_tail_idx = list(grp.index)[-RECENT_WINDOW_FOR_AMOUNT:]
        med_recent = median(all_amounts[i] for i in recent_tail_idx)
        if len(grp) >= 4 and med_recent > 0:
            core_idx = [i for i in grp.index if abs(all_amounts[i] - med_recent) <= 0.3 * med_recent]
            if len(core_idx) >= min_occurrences:
                grp = grp.loc[core_idx]

        dates = list(grp["_date"])

        # A single stray point that survives the amount-outlier filter (its
        # amount happens to be close enough to the recurring level) can still
        # corrupt the cadence: e.g. one commission payment landing near a
        # base-salary amount, dated mid-month, drags the average interval
        # away from the true ~30-day monthly rhythm the other points clearly
        # share. When a large majority of points agree on the same
        # day-of-month, drop the minority before computing intervals --
        # mirroring the amount-outlier filter above, but for the date axis.
        if len(dates) >= 4:
            days_of_month = [d.day for d in dates]
            mode_day = max(set(days_of_month), key=days_of_month.count)
            keep = [i for i, d in enumerate(days_of_month) if abs(d - mode_day) <= 1]
            if len(keep) >= max(min_occurrences, int(0.7 * len(dates))) and len(keep) < len(dates):
                grp = grp.iloc[keep]
                dates = list(grp["_date"])

        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        intervals = [i for i in intervals if i > 0]
        if not intervals:
            return None
        avg_interval = mean(intervals)
        if avg_interval > RECURRENCE_MAX_INTERVAL_DAYS:
            return None

        # Many monthly bills/salary in this dataset land on a fixed calendar
        # day every month; prefer stepping by calendar months over a raw
        # average day-count, which drifts by a day or two whenever a month's
        # length shifts the mean.
        monthly_day = None
        if 25 <= avg_interval <= 35 and len(dates) >= 2:
            days_of_month = [d.day for d in dates]
            mode_day = max(set(days_of_month), key=days_of_month.count)
            consistent = sum(1 for d in days_of_month if abs(d - mode_day) <= 1)
            if consistent >= max(2, int(0.7 * len(days_of_month))):
                monthly_day = mode_day

        recent_idx = list(grp.index)[-RECENT_WINDOW_FOR_AMOUNT:]
        home_amounts = [all_amounts[i] for i in recent_idx]
        estimate = _estimate_amount(home_amounts)
        if not is_credit:
            estimate *= _EXPENSE_CONSERVATISM
        signed_amount = estimate if is_credit else -estimate

        last_row = grp.iloc[-1]
        return RecurringSeries(
            category=category,
            interval_days=max(1, round(avg_interval)),
            amount=signed_amount,
            last_date=dates[-1],
            anchor_event_id=last_row["event_id"],
            flexibility=last_row["flexibility"] if not pd.isna(last_row["flexibility"]) else "fixed",
            minimum_allowed_amount=(
                float(last_row["minimum_allowed_amount"]) if not pd.isna(last_row["minimum_allowed_amount"]) else None
            ),
            monthly_day=monthly_day,
        )

    # -- message-derived amendments to the "salary" series --------------------
    def _apply_income_message_facts(
        self,
        series_by_category: Dict[str, RecurringSeries],
        facts: List[MessageFact],
        home_currency: str,
        request_date: date,
    ) -> None:
        salary_facts = latest_salary_facts(facts)
        if not salary_facts:
            return

        series = series_by_category.get("salary")
        pending_arrears: List[ForecastItem] = []

        for fact in salary_facts:
            if fact.kind == KIND_SALARY_STREAM_ENDED:
                if series is not None:
                    series.active = False
                continue

            amount_home = None
            if fact.amount is not None and fact.currency:
                try:
                    amount_home = self._fx.convert(fact.amount, fact.currency, home_currency, request_date)
                except ValueError:
                    amount_home = None

            if fact.kind in (KIND_SALARY_CONFIRMED,):
                if amount_home is None:
                    continue
                anchor = fact.effective_date or request_date
                interval = series.interval_days if series else 30
                series_by_category["salary"] = RecurringSeries(
                    category="salary",
                    interval_days=interval,
                    amount=amount_home,
                    last_date=anchor - timedelta(days=interval),
                    anchor_event_id=(series.anchor_event_id if series else "message"),
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                    active=True,
                )
                series = series_by_category["salary"]
            elif fact.kind == KIND_SALARY_BASE_ONLY and series is not None:
                if amount_home is not None:
                    series.amount = amount_home
                # "One household employment record has ended. The remaining
                # confirmed monthly salary is X" replaces the combined total
                # with a single figure -- if a second concurrent income
                # stream had been detected (see _split_bimodal), it must not
                # keep being projected on top of this restated total.
                secondary = series_by_category.get("salary~2")
                if secondary is not None:
                    secondary.active = False
            elif fact.kind == KIND_SALARY_PERMANENT_RAISE and series is not None:
                # Ongoing change effective from a stated date -- unlike the
                # temporary-dip patterns below, this permanently overwrites
                # the baseline amount for every future occurrence.
                if amount_home is not None:
                    series.amount = amount_home
                if fact.effective_date:
                    series.last_date = fact.effective_date - timedelta(days=series.interval_days)
                    series.monthly_day = fact.effective_date.day
            elif fact.kind == KIND_SALARY_TEMPORARY_AMOUNT and series is not None:
                # Affects ONLY the single next occurrence (unpaid leave, one
                # affected pay cycle), then the series must revert to its
                # normal baseline amount -- so we inject a one-off item for
                # just that cycle and advance last_date past it, without
                # touching series.amount.
                if amount_home is not None:
                    next_date = _step_forward(series, series.last_date)
                    if request_date <= next_date:
                        pending_arrears.append(
                            ForecastItem(
                                date=next_date,
                                amount=amount_home,
                                category="salary",
                                series_key="event:salary",
                                source="explicit",
                                event_id=None,
                                anchor_event_id=series.anchor_event_id,
                            )
                        )
                    series.last_date = next_date
            elif fact.kind == KIND_SALARY_NEXT_DATE and series is not None and fact.effective_date:
                series.last_date = fact.effective_date - timedelta(days=series.interval_days)
                series.monthly_day = fact.effective_date.day
            elif fact.kind == KIND_SALARY_ARREARS and series is not None:
                if amount_home is not None:
                    series.amount = amount_home
                if fact.extra_amount is not None and fact.currency:
                    try:
                        extra_home = self._fx.convert(fact.extra_amount, fact.currency, home_currency, request_date)
                        next_date = _step_forward(series, series.last_date)
                        if request_date <= next_date:
                            pending_arrears.append(
                                ForecastItem(
                                    date=next_date,
                                    amount=extra_home,
                                    category="salary",
                                    series_key="event:salary",
                                    source="explicit",
                                    event_id=None,
                                    anchor_event_id=series.anchor_event_id,
                                )
                            )
                    except ValueError:
                        pass

        self._pending_arrears = getattr(self, "_pending_arrears", [])
        self._pending_arrears = pending_arrears

    def _one_off_income_from_messages(
        self, facts: List[MessageFact], home_currency: str, request_date: date, horizon_end: date
    ) -> List[ForecastItem]:
        items = list(getattr(self, "_pending_arrears", []))
        seen_dates = set()
        for fact in facts:
            if fact.kind != KIND_CONFIRMED_ONE_OFF_INCOME:
                continue
            if fact.amount is None or fact.currency is None or fact.effective_date is None:
                continue
            if not (request_date <= fact.effective_date <= horizon_end):
                continue
            try:
                home_amt = self._fx.convert(fact.amount, fact.currency, home_currency, fact.effective_date)
            except ValueError:
                continue
            key = (fact.effective_date, round(home_amt, 2))
            if key in seen_dates:
                continue
            seen_dates.add(key)
            items.append(
                ForecastItem(
                    date=fact.effective_date,
                    amount=home_amt,
                    category="salary",
                    series_key="message:income",
                    source="explicit",
                    event_id=None,
                    anchor_event_id=fact.related_event_id,
                )
            )
        return items

    # -- forward projection ----------------------------------------------------
    def _project_series(
        self,
        series_by_category: Dict[str, RecurringSeries],
        request_date: date,
        horizon_end: date,
        forward_explicit: List[ForecastItem],
    ) -> List[ForecastItem]:
        explicit_by_category: Dict[str, List[tuple]] = defaultdict(list)
        for item in forward_explicit:
            explicit_by_category[item.category].append((item.date, item.amount))

        projected: List[ForecastItem] = []
        for series_dict_key, series in series_by_category.items():
            if not series.active:
                continue
            category = series.category
            can_stop, can_reduce = self._series_actions(series)
            next_date = _step_forward(series, series.last_date)
            explicit_points = explicit_by_category.get(category, [])
            n = 0
            while next_date <= horizon_end and n < 400:
                if next_date >= request_date:
                    # Only treat an explicit forward-dated event as "this is
                    # the same occurrence" (and skip generating a duplicate
                    # projected one) when it is close in BOTH date and
                    # amount -- a same-category explicit event with a wildly
                    # different amount (e.g. a one-off pending purchase vs. a
                    # recurring subscription) is a genuinely separate cash
                    # event and must not be silently dropped.
                    window = max(3, series.interval_days // 2)
                    near_explicit = any(
                        abs((next_date - d).days) <= window
                        and abs(abs(amt) - abs(series.amount)) <= 0.4 * max(abs(series.amount), 1.0)
                        for d, amt in explicit_points
                    )
                    if not near_explicit:
                        projected.append(
                            ForecastItem(
                                date=next_date,
                                amount=series.amount,
                                category=category,
                                series_key=f"series:{series_dict_key}",
                                source="projected",
                                event_id=None,
                                can_stop=can_stop,
                                can_reduce=can_reduce,
                                minimum_allowed_amount=series.minimum_allowed_amount,
                                anchor_event_id=series.anchor_event_id,
                            )
                        )
                next_date = _step_forward(series, next_date)
                n += 1
        return projected

    @staticmethod
    def _series_actions(series: RecurringSeries):
        flex = series.flexibility
        can_stop = flex in ("stoppable", "reducible_or_stoppable")
        can_reduce = flex in ("reducible", "reducible_or_stoppable") and series.minimum_allowed_amount is not None
        return can_stop, can_reduce

    def _build_spending_change_options(
        self,
        series_by_category: Dict[str, RecurringSeries],
        profile: pd.Series,
        request_date: date,
        horizon_end: date,
    ) -> List[SpendingChangeOption]:
        protect = _split(profile.get("expense_categories_to_protect"))
        willing_reduce = _split(profile.get("expense_categories_user_is_willing_to_reduce"))
        willing_stop = _split(profile.get("expense_categories_user_is_willing_to_stop"))

        options: List[SpendingChangeOption] = []
        for series_dict_key, series in series_by_category.items():
            category = series.category
            if category in protect or not series.active:
                continue
            occurrences = _count_occurrences(series, request_date, horizon_end)
            if occurrences <= 0:
                continue
            can_stop, can_reduce = self._series_actions(series)
            if can_stop and category in willing_stop:
                savings = abs(series.amount) * occurrences
                options.append(
                    SpendingChangeOption(
                        series_key=f"series:{series_dict_key}",
                        category=category,
                        anchor_event_id=series.anchor_event_id,
                        action="stop",
                        reduce_to_amount=None,
                        savings_total=savings,
                    )
                )
            if can_reduce and category in willing_reduce:
                per_occurrence_savings = abs(series.amount) - series.minimum_allowed_amount
                if per_occurrence_savings > 0:
                    options.append(
                        SpendingChangeOption(
                            series_key=f"series:{series_dict_key}",
                            category=category,
                            anchor_event_id=series.anchor_event_id,
                            action="reduce_to",
                            reduce_to_amount=series.minimum_allowed_amount,
                            savings_total=per_occurrence_savings * occurrences,
                        )
                    )
        return options


def _count_occurrences(series: RecurringSeries, request_date: date, horizon_end: date) -> int:
    n = 0
    next_date = _step_forward(series, series.last_date)
    guard = 0
    while next_date <= horizon_end and guard < 400:
        if next_date >= request_date:
            n += 1
        next_date = _step_forward(series, next_date)
        guard += 1
    return n


def _split(value) -> set:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return set(str(value).split("|"))


@dataclass
class UserForecast:
    user_id: str
    request_date: date
    horizon_end: date
    home_currency: str
    starting_balance: float
    minimum_balance: float
    items: List[ForecastItem]
    change_options: List[SpendingChangeOption]
