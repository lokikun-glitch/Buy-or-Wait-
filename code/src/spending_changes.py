"""Search for a minimal (<=3) combination of flexible spending changes that
makes a target payment (or payment schedule) safe."""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional

from .events import UserForecast
from .models import SpendingChangeOption
from .simulator import build_trajectory, is_amount_safe_on_date

MAX_CHANGES = 3


def _search(
    forecast: UserForecast, is_safe_with
) -> Optional[List[SpendingChangeOption]]:
    """Shared minimal-combination search: `is_safe_with(changes)` reports
    whether that combination of changes makes the target payment(s) safe."""
    options = sorted(forecast.change_options, key=lambda o: -o.savings_total)
    if not options:
        return None

    for size in range(1, MAX_CHANGES + 1):
        best_combo = None
        best_savings = -1.0
        for combo in combinations(options, size):
            if is_safe_with(list(combo)):
                total_savings = sum(c.savings_total for c in combo)
                if total_savings > best_savings:
                    best_savings = total_savings
                    best_combo = list(combo)
        if best_combo is not None:
            return best_combo
    return None


def find_minimal_changes(
    forecast: UserForecast, amount: float, pay_date, protect_none: bool = False
) -> Optional[List[SpendingChangeOption]]:
    def is_safe_with(changes):
        traj = build_trajectory(forecast, changes=changes)
        return is_amount_safe_on_date(forecast, amount, pay_date, trajectory=traj)

    return _search(forecast, is_safe_with)


def find_minimal_changes_for_payments(
    forecast: UserForecast, payments: List[tuple]
) -> Optional[List[SpendingChangeOption]]:
    """Same search, but against an arbitrary (date, amount) payment schedule
    -- e.g. a supplied installment option or a partial-payment pair -- rather
    than a single day-0 payment. Spending changes are not restricted to
    full_payment: the problem statement ties them to keeping the 90-day
    forecast safe, independent of which method is paying it off."""

    def is_safe_with(changes):
        extra = [(d, -amt) for d, amt in payments]
        traj = build_trajectory(forecast, changes=changes, extra_payments=extra)
        return traj.min_over_window() >= forecast.minimum_balance - 1e-6

    return _search(forecast, is_safe_with)
