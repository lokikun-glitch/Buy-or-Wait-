"""Regression check against dataset/sample_requests.csv.

Not used to tune per-row answers directly (that would be overfitting to
public examples) -- used to sanity-check that the deterministic engine's
methodology produces outputs in the right ballpark and with the right
shape, and to catch regressions while iterating.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data_loader import load_dataset
from src.pipeline import Pipeline

OUTPUT_COLS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


def main():
    ds = load_dataset()
    pipe = Pipeline(ds)
    samples = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "sample_requests.csv"))

    matches = {c: 0 for c in OUTPUT_COLS}
    amount_close = 0
    rows_out = []
    for _, row in samples.iterrows():
        request_cols = [
            "request_id", "user_id", "request_date", "request_type", "requested_amount",
            "desired_completion_date", "allows_partial_payment", "request_text",
        ]
        result = pipe.process_request(row[request_cols])
        rows_out.append(result)

        for c in OUTPUT_COLS:
            expected = row[c]
            got = result[c]
            if c == "amount_safe_to_pay":
                try:
                    if abs(float(expected) - float(got)) < 0.02 * max(1.0, abs(float(expected))) + 1.0:
                        amount_close += 1
                except Exception:
                    pass
            if str(expected).strip() == str(got).strip():
                matches[c] += 1
            elif c == "earliest_date_for_full_payment" and pd.isna(expected) and got == "":
                matches[c] += 1

    n = len(samples)
    print(f"n = {n}")
    for c in OUTPUT_COLS:
        print(f"  exact match {c}: {matches[c]}/{n}")
    print(f"  amount_safe_to_pay within tolerance: {amount_close}/{n}")

    out_df = pd.DataFrame(rows_out)
    out_path = os.path.join(os.path.dirname(__file__), "sample_predictions_debug.csv")
    out_df.to_csv(out_path, index=False)
    print(f"wrote debug predictions to {out_path}")


if __name__ == "__main__":
    main()
