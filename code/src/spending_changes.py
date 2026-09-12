"""Search for a minimal (<=3) combination of flexible spending changes that
makes a target payment safe on a given date."""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional

from .events import UserForecast
from .models import SpendingChangeOption
from .simulator import build_trajectory, is_amount_safe_on_date

MAX_CHANGES = 3


def find_minimal_changes(
    forecast: UserForecast, amount: float, pay_date, protect_none: bool = False
) -> Optional[List[SpendingChangeOption]]:
    options = sorted(forecast.change_options, key=lambda o: -o.savings_total)
    if not options:
        return None

    for size in range(1, MAX_CHANGES + 1):
        best_combo = None
        best_savings = -1.0
        for combo in combinations(options, size):
            traj = build_trajectory(forecast, changes=list(combo))
            if is_amount_safe_on_date(forecast, amount, pay_date, trajectory=traj):
                total_savings = sum(c.savings_total for c in combo)
                if total_savings > best_savings:
                    best_savings = total_savings
                    best_combo = list(combo)
        if best_combo is not None:
            return best_combo
    return None
