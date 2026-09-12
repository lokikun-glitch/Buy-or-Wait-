#!/usr/bin/env python3
"""Entry point for the "Buy or Wait?" solution.

Usage:
    python code/main.py

Reads dataset/requests.csv, runs the deterministic financial-decision
pipeline (see code/src/), and writes the completed predictions to
output.csv in the repository root.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_dataset  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    start = time.time()
    ds = load_dataset()
    pipeline = Pipeline(ds)

    print(f"Loaded {len(ds.requests)} requests for {ds.profiles['user_id'].nunique()} users.")
    output_df = pipeline.run(ds.requests)

    out_path = os.path.join(ROOT_DIR, "output.csv")
    output_df.to_csv(out_path, index=False)

    elapsed = time.time() - start
    print(f"Wrote {len(output_df)} predictions to {out_path} in {elapsed:.1f}s.")
    print(output_df["affordability_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
