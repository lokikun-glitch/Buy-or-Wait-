"""One-off experiment (Phase 3/11 of the amount_safe_to_pay audit): compare
candidate recurring-amount estimators against all 25 public samples.

Not part of the shipped pipeline -- run manually, not imported by main.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data_loader import load_dataset
from src.pipeline import Pipeline

OUTPUT_COLS = [
    "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment",
]


def main():
    ds = load_dataset()
    pipe = Pipeline(ds)
    s = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "sample_requests.csv"))

    matches = {c: 0 for c in OUTPUT_COLS}
    abs_errors = []
    for _, row in s.iterrows():
        cols = [
            "request_id", "user_id", "request_date", "request_type", "requested_amount",
            "desired_completion_date", "allows_partial_payment", "request_text",
        ]
        res = pipe.process_request(row[cols])
        for c in OUTPUT_COLS:
            if str(row[c]).strip() == str(res[c]).strip():
                matches[c] += 1
            elif c == "earliest_date_for_full_payment" and pd.isna(row[c]) and res[c] == "":
                matches[c] += 1
        try:
            abs_errors.append(abs(float(res["amount_safe_to_pay"]) - float(row["amount_safe_to_pay"])))
        except Exception:
            pass

    mae = sum(abs_errors) / len(abs_errors)
    max_err = max(abs_errors)
    strategy = os.environ.get("BUYORWAIT_AMOUNT_ESTIMATOR", "mean")
    print(f"strategy={strategy:14s} status={matches['affordability_status']:2d} method={matches['recommended_payment_method']:2d} "
          f"plan={matches['payment_plan']:2d} date={matches['earliest_date_for_full_payment']:2d} "
          f"MAE={mae:14,.0f} MAX={max_err:14,.0f}")


if __name__ == "__main__":
    main()
