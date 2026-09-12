"""Parse request_payment_options.csv rows into installment schedules."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd


def _to_date(d) -> date:
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


@dataclass
class InstallmentOption:
    payment_option_id: str
    request_id: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    frequency_days: int
    financing_fee: float
    total_payable_amount: float

    @property
    def schedule(self) -> List[tuple]:
        out = []
        for i in range(self.number_of_payments):
            d = self.first_payment_date + timedelta(days=self.frequency_days * i)
            out.append((d, self.payment_amount))
        return out

    @property
    def last_payment_date(self) -> date:
        return self.schedule[-1][0]


def load_installment_options(options_df: pd.DataFrame) -> List[InstallmentOption]:
    out = []
    for _, row in options_df.iterrows():
        if row["payment_method"] != "installments":
            continue
        freq = row["payment_frequency_days"]
        freq = int(freq) if not pd.isna(freq) else 30
        out.append(
            InstallmentOption(
                payment_option_id=row["payment_option_id"],
                request_id=row["request_id"],
                payment_amount=round(float(row["payment_amount"]), 2),
                number_of_payments=int(row["number_of_payments"]),
                first_payment_date=_to_date(row["first_payment_date"]),
                frequency_days=freq,
                financing_fee=float(row["financing_fee"]),
                total_payable_amount=round(float(row["total_payable_amount"]), 2),
            )
        )
    return out
