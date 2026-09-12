"""Per-request orchestration: build forecast -> plan -> verify -> explain ->
one output.csv row.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .currency import CurrencyConverter
from .data_loader import Dataset
from .events import EventEngine
from .explain import build_explanation
from .images import extract_amount as ocr_extract_amount
from .planner import PlanResult, build_plan
from .verifier import verify


def _to_date(s):
    return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()

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


def _fmt_num(x: float) -> str:
    x = round(float(x), 2)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}"


def _fmt_plan(payments) -> str:
    if not payments:
        return "none"
    return "|".join(f"{d.isoformat()}:{_fmt_num(a)}" for d, a in payments)


def _fmt_changes(changes) -> str:
    if not changes:
        return "none"
    parts = []
    for c in changes:
        if c.action == "stop":
            parts.append(f"stop:{c.anchor_event_id}")
        else:
            parts.append(f"reduce_to:{c.anchor_event_id}:{_fmt_num(c.reduce_to_amount)}")
    return "|".join(parts)


class ImageAmountResolver:
    """Resolves a blank financial_events.csv amount via the linked image,
    cached per user so repeated lookups (across recurrence detection and the
    forward-event pass) never re-run OCR."""

    def __init__(self, ds: Dataset):
        self._ds = ds
        self._cache = {}

    def __call__(self, row: pd.Series) -> Optional[float]:
        event_id = row["event_id"]
        if event_id in self._cache:
            return self._cache[event_id]
        images = self._ds.images[self._ds.images["related_event_id"] == event_id]
        value = None
        if not images.empty:
            image_id = images.iloc[0]["image_id"]
            value = ocr_extract_amount(image_id)
        self._cache[event_id] = value
        return value


class Pipeline:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.converter = CurrencyConverter(ds)
        self.resolver = ImageAmountResolver(ds)
        self.engine = EventEngine(ds, self.converter, self.resolver)

    def process_request(self, request: pd.Series) -> dict:
        user_id = request["user_id"]
        profile = self.ds.profile(user_id)
        forecast = self.engine.build_forecast(user_id, request["request_date"])
        options_df = self.ds.request_options(request["request_id"])

        requested_amount = round(float(request["requested_amount"]), 2)
        plan = build_plan(forecast, request, profile, options_df)

        ok, problems = verify(
            forecast, requested_amount, plan, _to_date(request["desired_completion_date"]), options_df
        )
        if not ok:
            # earliest_date_for_full_payment is computed independently of the
            # chosen method/plan (see planner.py) -- a verification failure on
            # the *chosen* plan doesn't invalidate that independent figure.
            plan = PlanResult(
                status="not_affordable",
                method="not_recommended",
                amount_safe_to_pay=plan.amount_safe_to_pay,
                earliest_date_for_full_payment=plan.earliest_date_for_full_payment,
                payments=[],
                changes=[],
            )

        explanation = build_explanation(forecast.home_currency, requested_amount, forecast.minimum_balance, plan)

        return {
            "request_id": request["request_id"],
            "amount_safe_to_pay": _fmt_num(plan.amount_safe_to_pay),
            "affordability_status": plan.status,
            "recommended_payment_method": plan.method,
            "payment_plan": _fmt_plan(plan.payments),
            "earliest_date_for_full_payment": (
                plan.earliest_date_for_full_payment.isoformat() if plan.earliest_date_for_full_payment else ""
            ),
            "spending_changes_needed": _fmt_changes(plan.changes),
            "decision_explanation": explanation,
        }

    def run(self, requests_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, request in requests_df.iterrows():
            try:
                rows.append(self.process_request(request))
            except Exception as exc:  # pragma: no cover - defensive fallback
                rows.append(
                    {
                        "request_id": request["request_id"],
                        "amount_safe_to_pay": "0",
                        "affordability_status": "not_affordable",
                        "recommended_payment_method": "not_recommended",
                        "payment_plan": "none",
                        "earliest_date_for_full_payment": "",
                        "spending_changes_needed": "none",
                        "decision_explanation": f"Unable to safely evaluate this request ({exc}).",
                    }
                )
        return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
