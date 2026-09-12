"""Final deterministic safety net: re-validate a PlanResult against the
dataset contract's invariants before it is written to output.csv. Anything
that fails is downgraded to a safe not_recommended answer rather than
shipping a rule-violating row.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

from .events import UserForecast
from .payment_options import load_installment_options
from .planner import PlanResult
from .simulator import build_trajectory


def verify(
    forecast: UserForecast,
    requested_amount: float,
    plan: PlanResult,
    desired_completion_date: Optional[date] = None,
    options_df: Optional[pd.DataFrame] = None,
) -> Tuple[bool, List[str]]:
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

    if len(plan.changes) > 3:
        problems.append(f"more than 3 spending changes ({len(plan.changes)})")
    changed_events = [c.anchor_event_id for c in plan.changes]
    if len(changed_events) != len(set(changed_events)):
        problems.append("stop and reduce_to both target the same event (must be mutually exclusive)")

    # "The plan must complete the request by desired_completion_date" is part
    # of what makes a plan safe (90-Day Safety Check), for every method
    # except `wait` -- which is definitionally the "not yet, but later" case.
    if desired_completion_date is not None and plan.method != "wait" and dates_sorted:
        if dates_sorted[-1] > desired_completion_date:
            problems.append(
                f"last payment {dates_sorted[-1]} is after desired_completion_date {desired_completion_date}"
            )

    # Installment plans must exactly match a *supplied* payment option --
    # don't just trust that the planner only ever builds them that way;
    # independently re-check against request_payment_options.csv.
    if plan.method == "installments":
        if options_df is None:
            problems.append("installments plan cannot be verified: no payment options supplied")
        else:
            matched = any(
                len(plan.payments) == len(opt.schedule)
                and all(
                    d == sched_d and abs(amt - sched_amt) <= 0.02 * max(1.0, sched_amt)
                    for (d, amt), (sched_d, sched_amt) in zip(plan.payments, opt.schedule)
                )
                for opt in load_installment_options(options_df)
            )
            if not matched:
                problems.append("installments plan does not exactly match any supplied payment option")

    extra = [(d, -amt) for d, amt in plan.payments]
    full_traj = build_trajectory(forecast, changes=plan.changes, extra_payments=extra)
    if full_traj.min_over_window() < forecast.minimum_balance - 0.02 * max(1.0, forecast.minimum_balance) - 1.0:
        problems.append("plan breaches minimum balance somewhere in the 90-day forecast")

    return (not problems, problems)
