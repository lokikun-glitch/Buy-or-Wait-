"""Small shared data structures used across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class ForecastItem:
    """One cash-flow item in a user's 90-day forward projection.

    amount is signed in home currency: positive = credit (income),
    negative = debit (expense). `series_key` groups an item with the
    recurring series it belongs to so spending changes can target it.
    """

    date: date
    amount: float
    category: str
    series_key: str
    source: str  # "explicit" (a real financial_events.csv row) or "projected"
    event_id: Optional[str] = None
    can_stop: bool = False
    can_reduce: bool = False
    minimum_allowed_amount: Optional[float] = None
    anchor_event_id: Optional[str] = None  # representative event_id for spending-change refs


@dataclass
class SpendingChangeOption:
    """A candidate flexible-spending change (stop or reduce) for a category series."""

    series_key: str
    category: str
    anchor_event_id: str
    action: str  # "stop" or "reduce_to"
    reduce_to_amount: Optional[float]
    savings_total: float  # total cash freed across the 90-day forecast window
