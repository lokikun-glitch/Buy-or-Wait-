"""Currency conversion using only the fixed, dated rates in exchange_rates.csv.

Rule from the dataset contract: for a foreign-currency cash event, use the
row for its settlement date and the stated from_currency -> to_currency
direction. For dates that fall between two published (monthly) rate rows
(e.g. a forecast date that isn't exactly the 15th), we use the most recent
published rate on or before that date -- treating each published rate as
"fixed until the next update", never an invented/interpolated value.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from .data_loader import Dataset


def _to_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d)[:10], "%Y-%m-%d").date()


class CurrencyConverter:
    def __init__(self, ds: Dataset):
        self._ds = ds
        # Build sorted rate rows per (from, to) pair for nearest-on-or-before lookups.
        self._pairs: dict[tuple, list[tuple[date, float]]] = {}
        for _, row in ds.rates.iterrows():
            key = (row.from_currency, row.to_currency)
            self._pairs.setdefault(key, []).append((_to_date(row.rate_date), float(row.rate)))
        for key in self._pairs:
            self._pairs[key].sort()

    def _rate_on_or_before(self, from_currency: str, to_currency: str, when: date) -> Optional[float]:
        rows = self._pairs.get((from_currency, to_currency))
        if not rows:
            return None
        best = None
        for d, r in rows:
            if d <= when:
                best = r
            else:
                break
        if best is not None:
            return best
        # No rate published yet on/before this date -- fall back to the earliest known rate
        # rather than inventing a number.
        return rows[0][1]

    def _edge_rate(self, from_currency: str, to_currency: str, when: date) -> Optional[float]:
        direct = self._rate_on_or_before(from_currency, to_currency, when)
        if direct is not None:
            return direct
        inverse = self._rate_on_or_before(to_currency, from_currency, when)
        if inverse is not None and inverse != 0:
            return 1.0 / inverse
        return None

    def rate(self, from_currency: str, to_currency: str, when) -> float:
        if from_currency == to_currency:
            return 1.0
        when = _to_date(when)

        direct = self._edge_rate(from_currency, to_currency, when)
        if direct is not None:
            return direct

        # BFS over the small currency graph, chaining published (or inverted) rates.
        currencies = {c for pair in self._pairs for c in pair} | {from_currency, to_currency}
        frontier = {from_currency: 1.0}
        visited = {from_currency}
        for _ in range(len(currencies)):
            next_frontier = {}
            for node, acc_rate in frontier.items():
                for other in currencies:
                    if other in visited:
                        continue
                    edge = self._edge_rate(node, other, when)
                    if edge is None:
                        continue
                    combined = acc_rate * edge
                    if other == to_currency:
                        return combined
                    next_frontier[other] = combined
                    visited.add(other)
            if not next_frontier:
                break
            frontier = next_frontier
        raise ValueError(f"No exchange rate path {from_currency}->{to_currency} on/before {when}")

    def convert(self, amount: float, from_currency: str, to_currency: str, when) -> float:
        if from_currency == to_currency:
            return amount
        return amount * self.rate(from_currency, to_currency, when)
