"""Deterministic 90-day balance simulator.

Balances are netted per calendar day (every date in this dataset is
day-granularity, so same-day events are simply summed together). This gives
a clean closed-form for the two headline numbers:

  base_balance(t)       = starting_balance + sum of all forecast deltas
                           from request_date through day t (no candidate
                           payment applied)
  amount_safe_to_pay     = clamp(min_t base_balance(t) - minimum_balance, 0, requested_amount)
                           (paying X today shifts base_balance(t) down by X
                           for every t >= request_date, uniformly, since X is
                           a one-time debit at t = request_date)
  earliest_date_for_full_payment
                          = first day D such that
                            min_{t=D..horizon_end} base_balance(t) >= minimum_balance + requested_amount
                            (a suffix-minimum, monotonic in D, found by scanning forward)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Set

from .events import UserForecast
from .models import ForecastItem, SpendingChangeOption


@dataclass
class DailyTrajectory:
    dates: List[date]
    end_of_day_balance: List[float]  # base balance, no candidate payment applied
    suffix_min: List[float]

    def suffix_min_from(self, d: date) -> Optional[float]:
        if d < self.dates[0]:
            d = self.dates[0]
        if d > self.dates[-1]:
            return None
        idx = (d - self.dates[0]).days
        return self.suffix_min[idx]

    def min_over_window(self) -> float:
        return self.suffix_min[0]


def _apply_changes(items: List[ForecastItem], changes: List[SpendingChangeOption]) -> List[ForecastItem]:
    if not changes:
        return items
    stop_keys = {c.series_key for c in changes if c.action == "stop"}
    reduce_map = {c.series_key: c.reduce_to_amount for c in changes if c.action == "reduce_to"}
    out = []
    for it in items:
        if it.series_key in stop_keys and it.source == "projected":
            continue
        if it.series_key in reduce_map and it.source == "projected":
            new_abs = reduce_map[it.series_key]
            out.append(
                ForecastItem(
                    date=it.date, amount=-abs(new_abs), category=it.category, series_key=it.series_key,
                    source=it.source, event_id=it.event_id, can_stop=it.can_stop, can_reduce=it.can_reduce,
                    minimum_allowed_amount=it.minimum_allowed_amount, anchor_event_id=it.anchor_event_id,
                )
            )
            continue
        out.append(it)
    return out


def build_trajectory(
    forecast: UserForecast,
    changes: Optional[List[SpendingChangeOption]] = None,
    extra_payments: Optional[List[tuple]] = None,
) -> DailyTrajectory:
    """extra_payments: list of (date, signed_amount) additional cash events
    (e.g. a candidate installment schedule) to fold into the base trajectory."""
    items = _apply_changes(forecast.items, changes or [])

    n_days = (forecast.horizon_end - forecast.request_date).days + 1
    deltas = [0.0] * n_days
    for it in items:
        if forecast.request_date <= it.date <= forecast.horizon_end:
            idx = (it.date - forecast.request_date).days
            deltas[idx] += it.amount
    for d, amt in (extra_payments or []):
        if forecast.request_date <= d <= forecast.horizon_end:
            idx = (d - forecast.request_date).days
            deltas[idx] += amt

    balances = [0.0] * n_days
    running = forecast.starting_balance
    for i in range(n_days):
        running += deltas[i]
        balances[i] = running

    suffix_min = [0.0] * n_days
    suffix_min[-1] = balances[-1]
    for i in range(n_days - 2, -1, -1):
        suffix_min[i] = min(balances[i], suffix_min[i + 1])

    dates = [forecast.request_date + timedelta(days=i) for i in range(n_days)]
    return DailyTrajectory(dates=dates, end_of_day_balance=balances, suffix_min=suffix_min)


def amount_safe_to_pay(forecast: UserForecast, requested_amount: float, trajectory: Optional[DailyTrajectory] = None) -> float:
    traj = trajectory or build_trajectory(forecast)
    safe = traj.min_over_window() - forecast.minimum_balance
    return max(0.0, min(requested_amount, round(safe, 2)))


def earliest_full_payment_date(
    forecast: UserForecast, requested_amount: float, trajectory: Optional[DailyTrajectory] = None
) -> Optional[date]:
    traj = trajectory or build_trajectory(forecast)
    threshold = forecast.minimum_balance + requested_amount
    for i, d in enumerate(traj.dates):
        if traj.suffix_min[i] >= threshold - 1e-6:
            return d
    return None


def is_amount_safe_on_date(
    forecast: UserForecast,
    amount: float,
    pay_date: date,
    trajectory: Optional[DailyTrajectory] = None,
) -> bool:
    traj = trajectory or build_trajectory(forecast)
    suf = traj.suffix_min_from(pay_date)
    if suf is None:
        return False
    return suf - amount >= forecast.minimum_balance - 1e-6


def min_balance_with_payments(
    forecast: UserForecast,
    payments: List[tuple],
    changes: Optional[List[SpendingChangeOption]] = None,
) -> float:
    """Minimum projected balance across the whole 90-day window if the given
    (date, positive_amount) payments are subtracted on top of the base
    (optionally change-adjusted) forecast."""
    extra = [(d, -amt) for d, amt in payments]
    traj = build_trajectory(forecast, changes=changes, extra_payments=extra)
    return traj.min_over_window()
