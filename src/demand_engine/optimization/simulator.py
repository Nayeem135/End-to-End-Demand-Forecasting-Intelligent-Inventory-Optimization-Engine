"""Deterministic, vectorised daily inventory simulator (lost-sales model).

All series are simulated in parallel as rows of ``(n_series, n_days)`` arrays.
Order of events within day ``t``:

1. Receive replenishment scheduled to arrive on day ``t``.
2. Serve demand: ``sold = min(on_hand, demand)``; unmet demand is lost.
3. Review (if ``t % review_period == 0``): when the inventory position
   (on hand + on order) is at or below the reorder point ``s_t``, place an order that
   arrives at the start of day ``t + lead_time``.
4. Pay holding cost on the end-of-day on-hand stock.

Policies:

* ``"sQ"``: order the smallest multiple of ``Q_t`` that lifts the position above ``s_t``.
* ``"sS"``: order up to ``S_t`` (min/max rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from demand_engine.types import FloatArray

PolicyKind = Literal["sQ", "sS"]


@dataclass(frozen=True)
class Policy:
    """Replenishment policy parameters per series and day.

    Attributes:
        name: Display name.
        kind: ``"sQ"`` (reorder point, order quantity) or ``"sS"`` (min/max).
        reorder_point: ``s`` with shape ``(n, T)``.
        quantity: ``Q`` (for ``sQ``) or order-up-to level ``S`` (for ``sS``), shape ``(n, T)``.
    """

    name: str
    kind: PolicyKind
    reorder_point: FloatArray
    quantity: FloatArray


@dataclass(frozen=True)
class CostModel:
    """Per-series unit costs (arrays of shape ``(n,)``) and fixed ordering cost."""

    holding_per_unit_day: FloatArray
    stockout_per_unit: FloatArray
    ordering_cost: float


@dataclass(frozen=True)
class SimulationOutput:
    """Daily trajectories ``(n, T)`` and per-series totals."""

    policy: str
    on_hand: FloatArray
    inventory_position: FloatArray
    orders: FloatArray
    sold: FloatArray
    lost: FloatArray
    summary: pd.DataFrame


def simulate(
    demand: FloatArray,
    policy: Policy,
    costs: CostModel,
    lead_time: int,
    review_period: int,
    initial_on_hand: FloatArray,
    unique_ids: list[str] | None = None,
) -> SimulationOutput:
    """Simulate a replenishment policy against actual demand.

    Args:
        demand: Actual demand, shape ``(n, T)``, non-negative.
        policy: Policy parameters (same shape as ``demand``).
        costs: Cost model.
        lead_time: Days between order placement and arrival (>= 1).
        review_period: Review every ``R`` days (>= 1).
        initial_on_hand: Starting stock per series, shape ``(n,)``.
        unique_ids: Optional series ids used as the summary index.

    Returns:
        :class:`SimulationOutput` with trajectories and a per-series summary.
    """
    _validate(demand, policy, lead_time, review_period, initial_on_hand)
    n, horizon = demand.shape
    on_hand = initial_on_hand.astype("float64").copy()
    arrivals = np.zeros((n, horizon + lead_time))
    on_order = np.zeros(n)
    oh_hist, ip_hist, ord_hist, sold_hist, lost_hist = (np.zeros((n, horizon)) for _ in range(5))

    for t in range(horizon):
        on_hand += arrivals[:, t]
        on_order -= arrivals[:, t]
        sold = np.minimum(on_hand, demand[:, t])
        on_hand -= sold
        order: FloatArray = np.zeros(n)
        if t % review_period == 0:
            position = on_hand + on_order
            order = _order_quantity(
                policy.kind, position, policy.reorder_point[:, t], policy.quantity[:, t]
            )
            arrivals[:, t + lead_time] += order
            on_order += order
        oh_hist[:, t], ip_hist[:, t], ord_hist[:, t] = on_hand, on_hand + on_order, order
        sold_hist[:, t], lost_hist[:, t] = sold, demand[:, t] - sold

    summary = summarize_run(demand, oh_hist, ord_hist, sold_hist, lost_hist, costs, unique_ids)
    return SimulationOutput(policy.name, oh_hist, ip_hist, ord_hist, sold_hist, lost_hist, summary)


def _order_quantity(
    kind: PolicyKind, position: FloatArray, s: FloatArray, q: FloatArray
) -> FloatArray:
    trigger = position <= s
    if kind == "sS":
        return np.where(trigger, np.ceil(np.maximum(q - position, 0.0)), 0.0)
    q_safe = np.maximum(q, 1.0)
    multiples = np.floor((s - position) / q_safe) + 1.0  # smallest n with position + n*Q > s
    return np.where(trigger, np.ceil(multiples * q_safe), 0.0)


def summarize_run(
    demand: FloatArray,
    on_hand: FloatArray,
    orders: FloatArray,
    sold: FloatArray,
    lost: FloatArray,
    costs: CostModel,
    unique_ids: list[str] | None,
) -> pd.DataFrame:
    """Per-series cost and service KPIs of a simulation run."""
    holding = (on_hand * costs.holding_per_unit_day[:, None]).sum(axis=1)
    n_orders = (orders > 0).sum(axis=1)
    ordering = n_orders * costs.ordering_cost
    stockout = lost.sum(axis=1) * costs.stockout_per_unit
    total_demand = demand.sum(axis=1)
    fill_rate = np.divide(
        sold.sum(axis=1), total_demand, out=np.ones(len(demand)), where=total_demand > 0
    )
    index = unique_ids if unique_ids is not None else list(range(len(demand)))
    return pd.DataFrame(
        {
            "holding_cost": holding,
            "ordering_cost": ordering,
            "stockout_cost": stockout,
            "total_cost": holding + ordering + stockout,
            "fill_rate": fill_rate,
            "stockout_days": (lost > 0).sum(axis=1),
            "avg_on_hand": on_hand.mean(axis=1),
            "n_orders": n_orders,
            "units_ordered": orders.sum(axis=1),
            "units_demanded": total_demand,
            "units_lost": lost.sum(axis=1),
        },
        index=pd.Index(index, name="unique_id"),
    )


def _validate(
    demand: FloatArray,
    policy: Policy,
    lead_time: int,
    review_period: int,
    initial_on_hand: FloatArray,
) -> None:
    if demand.ndim != 2:
        raise ValueError("demand must be a 2-D array (n_series, n_days)")
    if policy.reorder_point.shape != demand.shape or policy.quantity.shape != demand.shape:
        raise ValueError("policy arrays must have the same shape as demand")
    if initial_on_hand.shape != (demand.shape[0],):
        raise ValueError("initial_on_hand must have shape (n_series,)")
    if lead_time < 1 or review_period < 1:
        raise ValueError("lead_time and review_period must be >= 1")
    if np.any(demand < 0) or np.any(initial_on_hand < 0):
        raise ValueError("demand and initial stock must be non-negative")
