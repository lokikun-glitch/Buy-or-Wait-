"""Load and lightly normalize all dataset CSVs.

Everything is read once and indexed by user_id / request_id so the rest of
the pipeline never has to scan the full 25k-row events table per request.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
)


@dataclass
class Dataset:
    requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    rates: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame

    profiles_by_user: Dict[str, pd.Series] = field(default_factory=dict)
    events_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    options_by_request: Dict[str, pd.DataFrame] = field(default_factory=dict)
    messages_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    images_by_user: Dict[str, pd.DataFrame] = field(default_factory=dict)
    rate_lookup: Dict[tuple, float] = field(default_factory=dict)

    def profile(self, user_id: str) -> pd.Series:
        return self.profiles_by_user[user_id]

    def user_events(self, user_id: str) -> pd.DataFrame:
        return self.events_by_user.get(user_id, self.events.iloc[0:0])

    def request_options(self, request_id: str) -> pd.DataFrame:
        return self.options_by_request.get(request_id, self.payment_options.iloc[0:0])

    def user_messages(self, user_id: str) -> pd.DataFrame:
        return self.messages_by_user.get(user_id, self.messages.iloc[0:0])

    def user_images(self, user_id: str) -> pd.DataFrame:
        return self.images_by_user.get(user_id, self.images.iloc[0:0])

    def fx_rate(self, rate_date: str, from_currency: str, to_currency: str) -> Optional[float]:
        return self.rate_lookup.get((rate_date, from_currency, to_currency))


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATASET_DIR, name))


def load_dataset(dataset_dir: Optional[str] = None) -> Dataset:
    global DATASET_DIR
    if dataset_dir:
        DATASET_DIR = dataset_dir

    requests = _read_csv("requests.csv")
    profiles = _read_csv("financial_profiles.csv")
    events = _read_csv("financial_events.csv")
    rates = _read_csv("exchange_rates.csv")
    payment_options = _read_csv("request_payment_options.csv")
    messages = _read_csv("messages.csv")
    images = _read_csv("images.csv")

    ds = Dataset(
        requests=requests,
        profiles=profiles,
        events=events,
        rates=rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
    )

    ds.profiles_by_user = {row.user_id: row for _, row in profiles.iterrows()}
    ds.events_by_user = {uid: grp.copy() for uid, grp in events.groupby("user_id")}
    ds.options_by_request = {
        rid: grp.copy() for rid, grp in payment_options.groupby("request_id")
    }
    ds.messages_by_user = {uid: grp.copy() for uid, grp in messages.groupby("user_id")}
    ds.images_by_user = {uid: grp.copy() for uid, grp in images.groupby("user_id")}
    ds.rate_lookup = {
        (row.rate_date, row.from_currency, row.to_currency): float(row.rate)
        for _, row in rates.iterrows()
    }

    return ds
