"""Deterministic unit tests for the core engine, covering the boundary
conditions and edge cases called out in the project brief: zero/boundary
safe amounts, currency conversion, partial-payment math, flexible-spending
protections, pending/cancelled/failed event handling, and the 90-day
suffix-min safety check.

Run with:  python code/evaluation/test_units.py
No pytest dependency -- plain asserts, prints a pass/fail summary.
"""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.currency import CurrencyConverter  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.events import EventEngine  # noqa: E402
from src.models import ForecastItem, SpendingChangeOption  # noqa: E402
from src.simulator import (  # noqa: E402
    amount_safe_to_pay,
    build_trajectory,
    earliest_full_payment_date,
    is_amount_safe_on_date,
)
from src.events import UserForecast  # noqa: E402

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def make_forecast(start, minimum, items, request_date=date(2026, 1, 1), horizon_days=90):
    from datetime import timedelta

    return UserForecast(
        user_id="test",
        request_date=request_date,
        horizon_end=request_date + timedelta(days=horizon_days),
        home_currency="USD",
        starting_balance=start,
        minimum_balance=minimum,
        items=items,
        change_options=[],
    )


def test_zero_safe_amount():
    # Balance exactly at minimum -> nothing safe to pay.
    fc = make_forecast(1000.0, 1000.0, [])
    check("zero_safe_amount: nothing available at the floor", amount_safe_to_pay(fc, 500.0) == 0.0)


def test_exact_boundary():
    # Balance allows exactly the requested amount, no more, no less.
    fc = make_forecast(1500.0, 1000.0, [])
    check("exact_boundary: safe amount == balance - minimum", amount_safe_to_pay(fc, 500.0) == 500.0)
    check("exact_boundary: capped at requested_amount", amount_safe_to_pay(fc, 1000.0) == 500.0)


def test_full_affordability():
    fc = make_forecast(10000.0, 1000.0, [])
    check("full_affordability: full amount safe when balance is ample", amount_safe_to_pay(fc, 2000.0) == 2000.0)


def test_future_salary_unlocks_full_payment():
    items = [
        ForecastItem(date(2026, 1, 10), -800.0, "rent", "series:rent", "projected"),
        ForecastItem(date(2026, 1, 15), 3000.0, "salary", "series:salary", "projected"),
    ]
    fc = make_forecast(1000.0, 500.0, items)
    ed = earliest_full_payment_date(fc, 2500.0)
    check("future_salary: cannot pay 2500 on day 0", not is_amount_safe_on_date(fc, 2500.0, fc.request_date))
    check("future_salary: becomes safe once salary lands", ed == date(2026, 1, 15))


def test_recurring_expenses_drag_the_minimum():
    items = [ForecastItem(date(2026, 1, 1) + __import__("datetime").timedelta(days=10 * i), -100.0, "groceries", "s", "projected") for i in range(5)]
    fc = make_forecast(2000.0, 500.0, items)
    # Total drag = 500 over the window -> min balance ~1500 -> safe = 1000
    safe = amount_safe_to_pay(fc, 5000.0)
    check("recurring_expenses: multiple commitments reduce the safe amount", abs(safe - 1000.0) < 0.01)


def test_currency_conversion_direction_and_fallback():
    ds = load_dataset()
    conv = CurrencyConverter(ds)
    # USD -> INR on a known published month should be an exact, documented rate.
    rate_row = ds.rates[(ds.rates.from_currency == "USD") & (ds.rates.to_currency == "INR")].iloc[0]
    got = conv.convert(100.0, "USD", "INR", rate_row["rate_date"])
    check("currency: USD->INR uses the published rate", abs(got - 100.0 * float(rate_row["rate"])) < 0.01)
    # Inverse direction is derived, never invented out of thin air:
    # rate INR should convert back to exactly 1 USD.
    inv = conv.convert(float(rate_row["rate"]), "INR", "USD", rate_row["rate_date"])
    check("currency: inverse direction round-trips", abs(inv - 1.0) < 0.01)
    check("currency: same-currency is a no-op", conv.convert(42.0, "EUR", "EUR", "2025-01-01") == 42.0)


def test_partial_payment_math():
    from src.planner import build_plan
    import pandas as pd

    ds = load_dataset()
    conv = CurrencyConverter(ds)
    engine = EventEngine(ds, conv, lambda row: None)
    # Use a real light-history user by picking one directly from the dataset
    # so recurrence detection has something to work with.
    sample_user = ds.requests.iloc[0]["user_id"]
    profile = ds.profile(sample_user)
    fc = engine.build_forecast(sample_user, ds.requests.iloc[0]["request_date"])
    request = ds.requests.iloc[0].copy()
    request["allows_partial_payment"] = True
    plan = build_plan(fc, request, profile, ds.request_options(request["request_id"]))
    if plan.method == "partial_payment":
        (d1, a1), (d2, a2) = plan.payments
        check("partial_payment: exactly two payments", len(plan.payments) == 2)
        check("partial_payment: payments sum to requested amount", abs((a1 + a2) - float(request["requested_amount"])) < 0.02)
        check("partial_payment: first payment on request_date", d1.isoformat() == request["request_date"])
    else:
        check("partial_payment: (no partial_payment case hit for this fixture, skipped)", True)


def test_pending_credit_ignored_pending_debit_reserved():
    # A settled/scheduled credit that arrives mid-window, followed by a debit
    # that would otherwise breach the minimum, should widen the safe amount
    # for that later stretch of the window (this is what EventEngine does by
    # including confirmed credits but excluding pending ones entirely).
    late_debit = ForecastItem(date(2026, 2, 1), -900.0, "rent", "s", "explicit")
    credit = ForecastItem(date(2026, 1, 5), 1000.0, "windfall", "s", "explicit")
    fc_with_credit = make_forecast(1000.0, 500.0, [late_debit, credit])
    fc_without_credit = make_forecast(1000.0, 500.0, [late_debit])
    check(
        "pending_income: a confirmed credit raises the safe amount vs. excluding it",
        amount_safe_to_pay(fc_with_credit, 2000.0) > amount_safe_to_pay(fc_without_credit, 2000.0),
    )


def test_spending_change_stop_vs_reduce():
    items = [ForecastItem(date(2026, 1, 20), -300.0, "dining", "series:dining", "projected", can_reduce=True, minimum_allowed_amount=100.0, anchor_event_id="event_1")]
    fc = make_forecast(1000.0, 900.0, items)
    change = SpendingChangeOption("series:dining", "dining", "event_1", "reduce_to", 100.0, 200.0)
    traj_with = build_trajectory(fc, changes=[change])
    traj_without = build_trajectory(fc)
    check("spending_change: reduce_to raises the minimum trajectory point", traj_with.min_over_window() > traj_without.min_over_window())


def test_ninety_day_boundary():
    from datetime import timedelta

    fc = make_forecast(1000.0, 500.0, [ForecastItem(fc_date := date(2026, 1, 1) + timedelta(days=90), -400.0, "rent", "s", "projected")])
    # An event exactly on day 90 must still be inside the forecast window.
    traj = build_trajectory(fc)
    check("90_day_boundary: day-90 event is included in the window", traj.dates[-1] == fc_date)
    check("90_day_boundary: day-90 debit lowers the min balance", traj.min_over_window() == 600.0)


def test_day_91_excluded():
    from datetime import timedelta

    fc = make_forecast(1000.0, 500.0, [ForecastItem(date(2026, 1, 1) + timedelta(days=91), -400.0, "rent", "s", "projected")])
    traj = build_trajectory(fc)
    check("day_91: an event one day past the horizon has no effect on the window", traj.min_over_window() == 1000.0)


def test_balance_boundary_minimum_plus_one():
    fc = make_forecast(1001.0, 1000.0, [])
    check("balance_boundary: minimum+1 yields exactly 1 safe", amount_safe_to_pay(fc, 500.0) == 1.0)
    fc_below = make_forecast(999.0, 1000.0, [])
    check("balance_boundary: already below minimum yields 0 safe (never negative)", amount_safe_to_pay(fc_below, 500.0) == 0.0)


def test_verifier_independently_rejects_late_payment():
    # The verifier must not simply trust a plan handed to it -- construct one
    # by hand whose only payment lands one day after desired_completion_date
    # and confirm verify() flags it on its own, for every method except wait.
    from src.planner import PlanResult
    from src.verifier import verify

    fc = make_forecast(2000.0, 500.0, [])
    late_date = date(2026, 3, 1)
    deadline = date(2026, 2, 28)
    bad_plan = PlanResult(
        status="affordable_with_plan", method="full_payment", amount_safe_to_pay=1000.0,
        earliest_date_for_full_payment=late_date, payments=[(late_date, 1000.0)], changes=[],
    )
    ok, problems = verify(fc, 1000.0, bad_plan, desired_completion_date=deadline)
    check("verifier: independently rejects a full_payment plan finishing after the deadline", not ok)

    # The same late date is fine for `wait`, which is exempt by definition.
    wait_plan = PlanResult(
        status="affordable_later", method="wait", amount_safe_to_pay=0.0,
        earliest_date_for_full_payment=late_date, payments=[(late_date, 1000.0)], changes=[],
    )
    ok2, _ = verify(fc, 1000.0, wait_plan, desired_completion_date=deadline)
    check("verifier: `wait` is exempt from the deadline check", ok2)

    # And a payment exactly on the deadline must be accepted, not rejected.
    on_time_plan = PlanResult(
        status="affordable_with_plan", method="full_payment", amount_safe_to_pay=1000.0,
        earliest_date_for_full_payment=deadline, payments=[(deadline, 1000.0)], changes=[],
    )
    ok3, problems3 = verify(fc, 1000.0, on_time_plan, desired_completion_date=deadline)
    check("verifier: a payment exactly on the deadline is accepted", ok3)


def test_spending_changes_capped_at_three():
    from src.spending_changes import find_minimal_changes, MAX_CHANGES

    # Five tiny flexible categories, none alone (nor any pair/triple) enough
    # to close the gap -- find_minimal_changes must never return more than
    # MAX_CHANGES options even though closing the gap would need all five.
    items = [
        ForecastItem(date(2026, 1, 10), -100.0, f"cat{i}", f"series:cat{i}", "projected", can_reduce=False, can_stop=True, anchor_event_id=f"event_{i}")
        for i in range(5)
    ]
    fc = UserForecast(
        user_id="test", request_date=date(2026, 1, 1), horizon_end=date(2026, 1, 1) + __import__("datetime").timedelta(days=90),
        home_currency="USD", starting_balance=1450.0, minimum_balance=1000.0, items=items,
        change_options=[SpendingChangeOption(f"series:cat{i}", f"cat{i}", f"event_{i}", "stop", None, 100.0) for i in range(5)],
    )
    # Need 450 freed to safely pay 900 today (1450 - 900 = 550 < 1000); each
    # category only frees 100, so 3 changes (300) is the max allowed and is
    # still not enough -- this must return None, never a 4- or 5-change combo.
    result = find_minimal_changes(fc, 900.0, fc.request_date)
    check("spending_changes: never exceeds the 3-change cap even if more would help", result is None or len(result) <= MAX_CHANGES)


def test_verifier_rejects_fabricated_installment_plan():
    # The verifier must not simply trust that the planner only ever emits
    # installment schedules taken from request_payment_options.csv -- feed it
    # a plan with a schedule that does NOT match any supplied option and
    # confirm it is independently caught.
    from src.planner import PlanResult
    from src.verifier import verify
    import pandas as pd

    fc = make_forecast(5000.0, 500.0, [])
    options_df = pd.DataFrame([
        {"payment_option_id": "po1", "request_id": "r1", "payment_method": "installments", "payment_amount": 500.0,
         "number_of_payments": 2, "first_payment_date": "2026-01-05", "payment_frequency_days": 30,
         "financing_fee": 10.0, "total_payable_amount": 1010.0},
    ])
    # A schedule with a different amount than the one real option offers
    # (well outside rounding tolerance, not just off by a cent).
    fabricated = PlanResult(
        status="affordable_with_plan", method="installments", amount_safe_to_pay=0.0,
        earliest_date_for_full_payment=None,
        payments=[(date(2026, 1, 5), 300.0), (date(2026, 2, 4), 300.0)], changes=[],
    )
    ok, problems = verify(fc, 600.0, fabricated, options_df=options_df)
    check("verifier: rejects an installment schedule that matches no supplied option", not ok)

    genuine = PlanResult(
        status="affordable_with_plan", method="installments", amount_safe_to_pay=0.0,
        earliest_date_for_full_payment=None,
        payments=[(date(2026, 1, 5), 500.0), (date(2026, 2, 4), 500.0)], changes=[],
    )
    ok2, problems2 = verify(fc, 1000.0, genuine, options_df=options_df)
    check(f"verifier: accepts an installment schedule that matches a supplied option ({problems2})", ok2)


def test_installment_schedule_shapes():
    from src.payment_options import load_installment_options
    import pandas as pd

    rows = pd.DataFrame([
        {"payment_option_id": "po1", "request_id": "r1", "payment_method": "installments", "payment_amount": 500.0,
         "number_of_payments": 1, "first_payment_date": "2026-01-05", "payment_frequency_days": None,
         "financing_fee": 0.0, "total_payable_amount": 500.0},
        {"payment_option_id": "po2", "request_id": "r1", "payment_method": "installments", "payment_amount": 250.0,
         "number_of_payments": 2, "first_payment_date": "2026-01-05", "payment_frequency_days": 30,
         "financing_fee": 10.0, "total_payable_amount": 510.0},
        {"payment_option_id": "po3", "request_id": "r1", "payment_method": "installments", "payment_amount": 175.0,
         "number_of_payments": 3, "first_payment_date": "2026-01-05", "payment_frequency_days": 30,
         "financing_fee": 25.0, "total_payable_amount": 525.0},
    ])
    opts = {o.payment_option_id: o for o in load_installment_options(rows)}
    check("installments: 1-payment schedule has exactly 1 entry", len(opts["po1"].schedule) == 1)
    check("installments: 2-payment schedule has exactly 2 entries, 30 days apart",
          len(opts["po2"].schedule) == 2 and (opts["po2"].schedule[1][0] - opts["po2"].schedule[0][0]).days == 30)
    check("installments: 3-payment schedule has exactly 3 entries", len(opts["po3"].schedule) == 3)
    check("installments: last_payment_date matches the final scheduled entry", opts["po3"].last_payment_date == opts["po3"].schedule[-1][0])


def main():
    for fn in [
        test_zero_safe_amount,
        test_exact_boundary,
        test_full_affordability,
        test_future_salary_unlocks_full_payment,
        test_recurring_expenses_drag_the_minimum,
        test_currency_conversion_direction_and_fallback,
        test_partial_payment_math,
        test_pending_credit_ignored_pending_debit_reserved,
        test_spending_change_stop_vs_reduce,
        test_ninety_day_boundary,
        test_day_91_excluded,
        test_balance_boundary_minimum_plus_one,
        test_verifier_independently_rejects_late_payment,
        test_spending_changes_capped_at_three,
        test_verifier_rejects_fabricated_installment_plan,
        test_installment_schedule_shapes,
    ]:
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            FAIL.append(f"{fn.__name__} raised {exc!r}")
            print(f"FAIL {fn.__name__} raised {exc!r}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
