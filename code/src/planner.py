"""Candidate payment-plan generation, tiered status selection, and ranking.

Status precedence (highest first): affordable_now > affordable_with_plan >
affordable_later > not_affordable. We gather every safe, preference-eligible
candidate plan, keep only the ones in the best available tier, then break
ties using the six ranking rules from the problem statement (complete by
deadline, no spending changes, minimize total paid, start earlier, fewer
payments, lowest payment_option_id).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

from .events import UserForecast
from .models import SpendingChangeOption
from .payment_options import InstallmentOption, load_installment_options
from .simulator import (
    amount_safe_to_pay as calc_amount_safe_to_pay,
    build_trajectory,
    earliest_full_payment_date,
    is_amount_safe_on_date,
    min_balance_with_payments,
)
from .spending_changes import find_minimal_changes

TIER_NOW = 0
TIER_WITH_PLAN = 1
TIER_LATER = 2


@dataclass
class Candidate:
    method: str  # full_payment | partial_payment | installments | wait
    tier: int
    payments: List[tuple]  # (date, amount)
    changes: List[SpendingChangeOption] = field(default_factory=list)
    payment_option_id: Optional[str] = None

    @property
    def completes_by(self) -> date:
        return max(d for d, _ in self.payments)

    @property
    def total_paid(self) -> float:
        return round(sum(a for _, a in self.payments), 2)

    def meets_deadline(self, desired_completion_date: date) -> bool:
        return self.completes_by <= desired_completion_date

    def sort_key(self, desired_completion_date: date):
        return (
            0 if self.meets_deadline(desired_completion_date) else 1,
            len(self.changes),
            self.total_paid,
            min(d for d, _ in self.payments),
            len(self.payments),
            self.payment_option_id or "",
        )


@dataclass
class PlanResult:
    status: str
    method: str
    amount_safe_to_pay: float
    earliest_date_for_full_payment: Optional[date]
    payments: List[tuple]
    changes: List[SpendingChangeOption]
    payment_option_id: Optional[str] = None


def _accepted_methods(profile: pd.Series) -> set:
    val = profile.get("payment_methods_user_will_consider")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    return set(str(val).split("|"))


def _installment_duration_months(opt: InstallmentOption) -> float:
    span_days = (opt.last_payment_date - opt.first_payment_date).days
    return span_days / 30.0


def build_plan(
    forecast: UserForecast,
    request: pd.Series,
    profile: pd.Series,
    options_df: pd.DataFrame,
) -> PlanResult:
    requested_amount = round(float(request["requested_amount"]), 2)
    request_date = forecast.request_date
    desired_completion_date = _to_date(request["desired_completion_date"])
    allows_partial = bool(request["allows_partial_payment"])
    accepted = _accepted_methods(profile)

    base_trajectory = build_trajectory(forecast)
    base_safe_amount = calc_amount_safe_to_pay(forecast, requested_amount, trajectory=base_trajectory)
    base_earliest_date = earliest_full_payment_date(forecast, requested_amount, trajectory=base_trajectory)

    candidates: List[Candidate] = []

    # 1. Full payment today, no changes.
    full_now_safe = is_amount_safe_on_date(forecast, requested_amount, request_date, trajectory=base_trajectory)
    if "full_payment" in accepted and full_now_safe:
        candidates.append(Candidate("full_payment", TIER_NOW, [(request_date, requested_amount)]))

    # 2. Partial payment.
    if (
        allows_partial
        and "partial_payment" in accepted
        and 0 < base_safe_amount < requested_amount
        and base_earliest_date is not None
        and base_earliest_date <= desired_completion_date
    ):
        remaining = round(requested_amount - base_safe_amount, 2)
        candidates.append(
            Candidate(
                "partial_payment",
                TIER_WITH_PLAN,
                [(request_date, base_safe_amount), (base_earliest_date, remaining)],
            )
        )

    # 3. Installments (must exactly match a supplied option).
    max_months = profile.get("max_installment_months")
    has_max_months = max_months is not None and not (isinstance(max_months, float) and pd.isna(max_months))
    if "installments" in accepted and has_max_months:
        for opt in load_installment_options(options_df):
            if _installment_duration_months(opt) > float(max_months) + 1e-6:
                continue
            min_bal = min_balance_with_payments(forecast, opt.schedule)
            if min_bal >= forecast.minimum_balance - 1e-6:
                candidates.append(
                    Candidate(
                        "installments", TIER_WITH_PLAN, opt.schedule, payment_option_id=opt.payment_option_id
                    )
                )

    # 4. Full payment today with flexible spending changes (only tried if
    #    plain full payment today isn't already safe).
    if "full_payment" in accepted and not full_now_safe:
        changes = find_minimal_changes(forecast, requested_amount, request_date)
        if changes:
            candidates.append(
                Candidate("full_payment", TIER_WITH_PLAN, [(request_date, requested_amount)], changes=changes)
            )

    # 5. Wait: full payment later, no changes.
    if "full_payment" in accepted and base_earliest_date is not None:
        candidates.append(Candidate("wait", TIER_LATER, [(base_earliest_date, requested_amount)]))

    if not candidates:
        return PlanResult(
            status="not_affordable",
            method="not_recommended",
            amount_safe_to_pay=base_safe_amount,
            earliest_date_for_full_payment=base_earliest_date,
            payments=[],
            changes=[],
        )

    best_tier = min(c.tier for c in candidates)
    tier_candidates = [c for c in candidates if c.tier == best_tier]
    tier_candidates.sort(key=lambda c: c.sort_key(desired_completion_date))
    winner = tier_candidates[0]

    if best_tier == TIER_NOW:
        status = "affordable_now"
    elif best_tier == TIER_WITH_PLAN:
        status = "affordable_with_plan"
    else:
        status = "affordable_later"

    return PlanResult(
        status=status,
        method=winner.method,
        amount_safe_to_pay=base_safe_amount,
        earliest_date_for_full_payment=base_earliest_date,
        payments=winner.payments,
        changes=winner.changes,
        payment_option_id=winner.payment_option_id,
    )


def _to_date(s) -> date:
    from datetime import datetime

    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
