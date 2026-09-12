"""Deterministic, template-based explanation generation.

No LLM is used (per project decision). The explanation is assembled purely
from the already-finalized, verified decision facts -- it never influences
the decision itself, only describes it in plain language, matching the
style of dataset/sample_requests.csv.
"""
from __future__ import annotations

from datetime import date

from .models import SpendingChangeOption
from .planner import PlanResult


def _fmt_amount(amount: float) -> str:
    amount = round(amount, 2)
    if abs(amount - round(amount)) < 1e-9:
        return f"{amount:,.0f}"
    return f"{amount:,.2f}"


def _fmt_date(d: date) -> str:
    return f"{d.day} {d.strftime('%B %Y')}"


def _change_phrase(change: SpendingChangeOption) -> str:
    label = change.category.replace("_", " ")
    if change.action == "stop":
        return f"Stop the {label} expense"
    return f"Reduce {label} spending to {_fmt_amount(change.reduce_to_amount)}"


def build_explanation(
    currency: str,
    requested_amount: float,
    minimum_balance: float,
    plan: PlanResult,
) -> str:
    cur = currency

    if plan.status == "not_affordable":
        return (
            f"Do not proceed with this {cur} {_fmt_amount(requested_amount)} request. "
            f"No available option keeps the {cur} {_fmt_amount(minimum_balance)} minimum balance protected "
            f"within the 90-day forecast, even though {cur} {_fmt_amount(plan.amount_safe_to_pay)} is safe to pay today."
        )

    change_clause = ""
    if plan.changes:
        phrases = [_change_phrase(c) for c in plan.changes]
        change_clause = ", then ".join(phrases) + ", then "
        change_clause = change_clause[0].upper() + change_clause[1:]

    if plan.method == "full_payment":
        pay_date, amt = plan.payments[0]
        if change_clause:
            body = f"{change_clause}pay {cur} {_fmt_amount(amt)} on {_fmt_date(pay_date)}."
        else:
            body = f"Pay {cur} {_fmt_amount(amt)} on {_fmt_date(pay_date)}."
        return body + f" This keeps at least {cur} {_fmt_amount(minimum_balance)} available over the next 90 days."

    if plan.method == "wait":
        pay_date, amt = plan.payments[0]
        return (
            f"Wait until {_fmt_date(pay_date)}, then pay {cur} {_fmt_amount(amt)} in full. "
            f"Paying sooner would take the balance below the {cur} {_fmt_amount(minimum_balance)} minimum."
        )

    if plan.method == "partial_payment":
        (d1, a1), (d2, a2) = plan.payments
        return (
            f"Pay {cur} {_fmt_amount(a1)} on {_fmt_date(d1)} and the remaining {cur} {_fmt_amount(a2)} "
            f"on {_fmt_date(d2)}. This completes the full request and keeps the {cur} {_fmt_amount(minimum_balance)} "
            f"minimum protected."
        )

    if plan.method == "installments":
        n = len(plan.payments)
        first_date, first_amt = plan.payments[0]
        return (
            f"Use {n} installments of {cur} {_fmt_amount(first_amt)}, starting {_fmt_date(first_date)}. "
            f"This leaves at least {cur} {_fmt_amount(minimum_balance)} available."
        )

    return "No safe payment option is available for this request."
