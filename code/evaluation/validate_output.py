"""Validate the final output.csv against the dataset contract in
AGENTS.md / problem_statement.md. Run after code/main.py.

Usage:
    python code/evaluation/validate_output.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]
STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def fail(msg, errors):
    errors.append(msg)


def main() -> int:
    requests = pd.read_csv(os.path.join(ROOT, "dataset", "requests.csv"))
    options = pd.read_csv(os.path.join(ROOT, "dataset", "request_payment_options.csv"))
    out_path = os.path.join(ROOT, "output.csv")
    out = pd.read_csv(out_path, dtype=str, keep_default_na=False)

    errors = []

    if list(out.columns) != REQUIRED_COLUMNS:
        fail(f"column mismatch: {list(out.columns)}", errors)

    if set(out["request_id"]) != set(requests["request_id"]):
        missing = set(requests["request_id"]) - set(out["request_id"])
        extra = set(out["request_id"]) - set(requests["request_id"])
        fail(f"request_id set mismatch: missing={missing} extra={extra}", errors)

    if len(out) != len(requests):
        fail(f"row count mismatch: {len(out)} vs {len(requests)}", errors)

    req_by_id = requests.set_index("request_id")
    opts_by_request = {rid: grp for rid, grp in options.groupby("request_id")}

    for _, row in out.iterrows():
        rid = row["request_id"]
        if rid not in req_by_id.index:
            continue
        req = req_by_id.loc[rid]
        requested_amount = float(req["requested_amount"])
        desired_completion = datetime.strptime(str(req["desired_completion_date"])[:10], "%Y-%m-%d").date()

        try:
            amt = float(row["amount_safe_to_pay"])
        except ValueError:
            fail(f"{rid}: amount_safe_to_pay not numeric: {row['amount_safe_to_pay']!r}", errors)
            continue
        if not (-1e-6 <= amt <= requested_amount + 1e-6):
            fail(f"{rid}: amount_safe_to_pay {amt} out of [0, {requested_amount}]", errors)

        if row["affordability_status"] not in STATUSES:
            fail(f"{rid}: invalid affordability_status {row['affordability_status']!r}", errors)
        if row["recommended_payment_method"] not in METHODS:
            fail(f"{rid}: invalid recommended_payment_method {row['recommended_payment_method']!r}", errors)

        plan = row["payment_plan"]
        if plan != "none":
            entries = plan.split("|")
            dates = []
            total = 0.0
            for e in entries:
                try:
                    d_str, a_str = e.split(":")
                    d = datetime.strptime(d_str, "%Y-%m-%d").date()
                    a = float(a_str)
                except ValueError:
                    fail(f"{rid}: malformed payment_plan entry {e!r}", errors)
                    continue
                dates.append(d)
                total += a
            if dates != sorted(dates):
                fail(f"{rid}: payment_plan not chronological: {plan}", errors)

            method = row["recommended_payment_method"]
            if method == "partial_payment":
                if len(entries) != 2:
                    fail(f"{rid}: partial_payment must have exactly 2 payments: {plan}", errors)
                elif abs(total - requested_amount) > 0.02:
                    fail(f"{rid}: partial_payment total {total} != requested {requested_amount}", errors)
            if method in ("full_payment", "partial_payment") and dates:
                if dates[-1] > desired_completion and row["affordability_status"] != "affordable_later":
                    fail(f"{rid}: last payment {dates[-1]} after deadline {desired_completion}", errors)

        if row["affordability_status"] == "affordable_now":
            if row["earliest_date_for_full_payment"] != req["request_date"]:
                fail(
                    f"{rid}: affordable_now but earliest_date_for_full_payment="
                    f"{row['earliest_date_for_full_payment']!r} != request_date {req['request_date']!r}",
                    errors,
                )

        if row["affordability_status"] == "not_affordable":
            if row["recommended_payment_method"] != "not_recommended" or row["payment_plan"] != "none":
                fail(f"{rid}: not_affordable must be not_recommended/none", errors)

        changes = row["spending_changes_needed"]
        if changes != "none":
            parts = changes.split("|")
            if len(parts) > 3:
                fail(f"{rid}: more than 3 spending changes: {changes}", errors)
            for p in parts:
                if not (p.startswith("stop:") or p.startswith("reduce_to:")):
                    fail(f"{rid}: malformed spending change {p!r}", errors)

        if row["recommended_payment_method"] == "installments" and plan != "none":
            rid_opts = opts_by_request.get(rid)
            matched = False
            if rid_opts is not None:
                entries = [e.split(":") for e in plan.split("|")]
                amounts = [round(float(a), 2) for _, a in entries]
                for _, opt in rid_opts[rid_opts["payment_method"] == "installments"].iterrows():
                    if len(amounts) == int(opt["number_of_payments"]) and abs(amounts[0] - float(opt["payment_amount"])) < 0.02:
                        matched = True
                        break
            if not matched:
                fail(f"{rid}: installments plan does not match a supplied payment_option: {plan}", errors)

    if errors:
        print(f"VALIDATION FAILED: {len(errors)} problem(s)")
        for e in errors[:50]:
            print(" -", e)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1

    print(f"VALIDATION PASSED: {len(out)} rows, all checks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
