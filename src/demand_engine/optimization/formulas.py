"""Classical inventory formulas as small, pure, vectorised functions.

Units: demand in units/day, costs in currency. ``protection_days`` is the period
the inventory position must cover after an order decision: ``L + R - 1`` for a
lead time of ``L`` days and a review period of ``R`` days, given that an order
placed at the end of day ``t`` arrives at the start of day ``t + L``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from demand_engine.types import FloatArray

DAYS_PER_YEAR = 365.0


def protection_days(lead_time: int, review_period: int) -> int:
    """Days of demand the inventory position must cover: ``L + R - 1``."""
    if lead_time < 1 or review_period < 1:
        raise ValueError("lead_time and review_period must be >= 1 day")
    return lead_time + review_period - 1


def z_score(service_level: float) -> float:
    """Standard-normal quantile for a target cycle service level (e.g. 0.95 -> 1.645)."""
    if not 0.0 < service_level < 1.0:
        raise ValueError("service_level must be in (0, 1)")
    return float(norm.ppf(service_level))


def safety_stock(sigma_protection: FloatArray, service_level: float) -> FloatArray:
    """Safety stock ``z * sigma`` (``sigma``: forecast-error spread over the protection period)."""
    if np.any(sigma_protection < 0):
        raise ValueError("sigma must be non-negative")
    return z_score(service_level) * sigma_protection


def reorder_point(expected_demand: FloatArray, safety: FloatArray) -> FloatArray:
    """Reorder point = expected demand over the protection period + safety stock."""
    return expected_demand + safety


def eoq(
    annual_demand: FloatArray, ordering_cost: float, annual_holding_cost: FloatArray
) -> FloatArray:
    """Economic order quantity ``sqrt(2 D K / H)`` (0 when there is no demand).

    Args:
        annual_demand: Expected demand per year (units).
        ordering_cost: Fixed cost per order.
        annual_holding_cost: Cost of holding one unit for a year.
    """
    if np.any(annual_holding_cost <= 0):
        raise ValueError("annual holding cost must be positive")
    if ordering_cost < 0 or np.any(annual_demand < 0):
        raise ValueError("ordering cost and demand must be non-negative")
    return np.sqrt(2.0 * annual_demand * ordering_cost / annual_holding_cost)


def daily_holding_cost(price: FloatArray, annual_rate: float) -> FloatArray:
    """Cost of holding one unit for one day: ``price * rate / 365``."""
    return price * annual_rate / DAYS_PER_YEAR


def critical_ratio(underage_cost: float, overage_cost: float) -> float:
    """Newsvendor optimal service level ``Cu / (Cu + Co)``."""
    if underage_cost < 0 or overage_cost < 0 or underage_cost + overage_cost == 0:
        raise ValueError("costs must be non-negative and not both zero")
    return underage_cost / (underage_cost + overage_cost)
