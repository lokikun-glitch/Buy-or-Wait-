"""Final deterministic safety net: re-validate a PlanResult against the
dataset contract's invariants before it is written to output.csv. Anything
that fails is downgraded to a safe not_recommended answer rather than
shipping a rule-violating row.
"""
from __future__ import annotations

from typing import List, Tuple

from .events import UserForecast
from .planner import PlanResult
from .simulator import build_trajectory


def verify(forecast: UserForecast, requested_amount: float, plan: PlanResult) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    if not (0 - 1e-6 <= plan.amount_safe_to_pay <= requested_amount + 1e-6):
        problems.append("amount_safe_to_pay out of bounds")

    if plan.status == "not_affordable":
        if plan.method != "not_recommended" or plan.payments:
            problems.append("not_affordable must have no payments / not_recommended")
        return (not problems, problems)

    if not plan.payments:
        problems.append("missing payments for a recommended plan")
        return (not problems, problems)

    total_paid = round(sum(a for _, a in plan.payments), 2)
    if plan.method in ("full_payment", "partial_payment", "installments") and plan.status != "affordable_later":
        if abs(total_paid - requested_amount) > 0.02 * max(1.0, requested_amount) and plan.method != "installments":
            problems.append(f"payments do not sum to requested amount ({total_paid} vs {requested_amount})")

    if plan.method == "partial_payment" and len(plan.payments) != 2:
        problems.append("partial_payment must have exactly two payments")

    dates_sorted = sorted(d for d, _ in plan.payments)
    if dates_sorted != [d for d, _ in plan.payments]:
        problems.append("payments not in chronological order")

    extra = [(d, -amt) for d, amt in plan.payments]
    full_traj = build_trajectory(forecast, changes=plan.changes, extra_payments=extra)
    if full_traj.min_over_window() < forecast.minimum_balance - 0.02 * max(1.0, forecast.minimum_balance) - 1.0:
        problems.append("plan breaches minimum balance somewhere in the 90-day forecast")

    return (not problems, problems)
