"""Construct the replenishment policies compared in the simulation.

Information rule (no look-ahead): on day ``t`` a policy may only use the forecast
issued at the latest cut-off on or before ``t``. The test period is a sequence of
forecast *windows* (one per cut-off); ``window_id[t]`` says which window day ``t``
belongs to. Beyond the end of its own window a forecast is extended by repeating
its final week (keeping the weekday pattern) - never by peeking at the next
window's forecast.

1. ``rule_of_thumb``: typical manual min/max rule - reorder when the position falls
   below ``L`` days of recent average sales, fill up to ``N`` days of cover. The
   "recent average" is refreshed at every cut-off. No safety-stock maths.
2. ``ma_safety_stock``: (s, Q) policy - reorder point = forecast over the protection
   period + safety stock from measured errors - fed with the moving-average forecast.
3. ``forecast_driven``: the *identical* (s, Q) policy fed with the champion
   model's daily forecast and its own measured errors.

The order quantity ``Q`` is the EOQ of the *long-run* demand rate (last year of
history) and is shared by policies 2 and 3: lot sizing is a long-run economic
decision, while the forecast's job is timing (when to reorder). Using a 28-day
forecast for EOQ would let a zero forecast for a slow mover collapse ``Q`` to one
unit and generate absurd ordering costs. Policies 2 and 3 therefore differ *only*
in the forecast behind the reorder point, so their cost gap isolates the value
of forecast accuracy.
"""

from __future__ import annotations

import numpy as np

from demand_engine.optimization.formulas import DAYS_PER_YEAR, eoq, safety_stock
from demand_engine.optimization.simulator import Policy
from demand_engine.types import FloatArray, IntArray

_SEASON = 7


def _blocks(window_id: IntArray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` index ranges of each forecast window."""
    if window_id.ndim != 1 or window_id.size == 0:
        raise ValueError("window_id must be a non-empty 1-D array")
    if np.any(np.diff(window_id) < 0):
        raise ValueError("window_id must be non-decreasing")
    edges = np.flatnonzero(np.diff(window_id)) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, window_id.size]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def window_mean(forecast: FloatArray, window_id: IntArray) -> FloatArray:
    """Each day's value replaced by the mean forecast of its window, shape ``(n, T)``."""
    out = np.empty_like(forecast)
    for start, end in _blocks(window_id):
        out[:, start:end] = forecast[:, start:end].mean(axis=1, keepdims=True)
    return out


def forward_sum(forecast: FloatArray, days: int, window_id: IntArray) -> FloatArray:
    """For each day ``t`` the forecast total over ``t+1 .. t+days`` using only ``t``'s window.

    Beyond the window end the forecast is continued by cycling its last week.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    n = forecast.shape[0]
    out = np.empty_like(forecast)
    for start, end in _blocks(window_id):
        block = forecast[:, start:end]
        tail = block[:, -min(_SEASON, block.shape[1]) :]
        reps = int(np.ceil(days / tail.shape[1])) + 1
        padded = np.concatenate([block, np.tile(tail, (1, reps))], axis=1)
        csum = np.concatenate([np.zeros((n, 1)), np.cumsum(padded, axis=1)], axis=1)
        t = np.arange(end - start)
        out[:, start:end] = csum[:, t + 1 + days] - csum[:, t + 1]
    return out


def rule_of_thumb_policy(recent_mean: FloatArray, lead_time: int, cover_days: int) -> Policy:
    """Min/max rule from a daily demand rate ``(n, T)``: ``s = L x rate``, ``S = N x rate``."""
    s = recent_mean * lead_time
    big_s = np.maximum(recent_mean * cover_days, s + 1.0)
    return Policy(name="rule_of_thumb", kind="sS", reorder_point=s, quantity=big_s)


def order_quantity(
    mean_daily: FloatArray, ordering_cost: float, annual_holding: FloatArray
) -> FloatArray:
    """EOQ from mean daily demand (broadcastable to ``annual_holding``), at least one unit."""
    return np.maximum(eoq(mean_daily * DAYS_PER_YEAR, ordering_cost, annual_holding), 1.0)


def forecast_policy(
    name: str,
    forecast: FloatArray,
    window_id: IntArray,
    bias: FloatArray,
    sigma: FloatArray,
    protection: int,
    service_level: float,
    lot_size: FloatArray,
) -> Policy:
    """(s, Q) policy whose reorder point follows a forecast, re-planned at every cut-off.

    Args:
        name: Policy name.
        forecast: Daily forecasts ``(n, T)`` issued at the window cut-offs.
        window_id: Forecast window of each day ``(T,)``.
        bias: Mean cumulative forecast error over the protection period ``(n,)``.
        sigma: Std of the cumulative forecast error over the protection period ``(n,)``.
        protection: Protection period ``P`` in days.
        service_level: Target cycle service level.
        lot_size: Order quantity per series ``(n,)`` (e.g. from :func:`order_quantity`).

    Returns:
        Policy with ``s_t = forecast over (t, t+P] + bias + z x sigma`` and constant ``Q``.
    """
    buffer = bias + safety_stock(sigma, service_level)
    s = forward_sum(forecast, protection, window_id) + buffer[:, None]
    q = np.repeat(lot_size[:, None], forecast.shape[1], axis=1)
    return Policy(name=name, kind="sQ", reorder_point=s, quantity=q)
