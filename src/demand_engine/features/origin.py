"""Horizon-aware "forecast origin" features.

The fixed lags in :mod:`demand_engine.features.lags` must be ``>= horizon`` so one
model can forecast every day of the window. That throws away the most recent
weeks for short horizons. Origin features fix this: each row carries its forecast
horizon ``h`` (days between the forecast origin and the target day) and statistics
of demand *at the origin* ``t - h``:

* ``horizon``          - days ahead (1..H);
* ``origin_last``      - demand on the origin day;
* ``origin_mean_7/28`` - mean demand over the 7/28 days ending at the origin;
* ``origin_dow_mean``  - mean of the 4 most recent same-weekday observations known
  at the origin (a seasonal-naive average).

Leakage guarantee: every value only reads demand at or before ``t - h``. During
training each row is assigned a random horizon (seeded), so the model learns all
horizons without replicating the data; at prediction time ``h`` is the true
distance from the cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_engine.data.schema import UNIQUE_ID, Y
from demand_engine.types import FloatArray, IntArray

ORIGIN_FEATURES = ("horizon", "origin_last", "origin_mean_7", "origin_mean_28", "origin_dow_mean")
_DOW_WEEKS = 4


def _gather(values: FloatArray, positions: IntArray, offsets: IntArray) -> FloatArray:
    """``values[i - offsets[i]]`` within the same series, NaN when it falls before the start."""
    idx = np.arange(values.size) - offsets
    ok = positions >= offsets
    out = np.full(values.size, np.nan)
    out[ok] = values[idx[ok]]
    return out


def add_origin_features(df: pd.DataFrame, horizons: IntArray) -> pd.DataFrame:
    """Return a copy of ``df`` with origin features for the given per-row horizons.

    Args:
        df: Long table sorted by ``unique_id, ds`` with contiguous daily rows.
        horizons: Forecast horizon (>= 1) of every row, aligned with ``df``.

    Raises:
        ValueError: If shapes mismatch or a horizon is < 1.
    """
    if horizons.shape != (len(df),):
        raise ValueError("horizons must have one entry per row")
    if horizons.size and horizons.min() < 1:
        raise ValueError("horizons must be >= 1")
    out = df.copy()
    ids = out[UNIQUE_ID]
    y = out[Y].to_numpy(dtype="float64")
    positions = np.asarray(out.groupby(UNIQUE_ID, sort=False).cumcount(), dtype=np.int64)
    h = horizons.astype("int64")

    out["horizon"] = h
    out["origin_last"] = _gather(y, positions, h)
    by_id = out[Y].groupby(ids, sort=False)
    for w in (7, 28):
        roll = by_id.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        out[f"origin_mean_{w}"] = _gather(roll.to_numpy(dtype="float64"), positions, h)

    first_week = np.ceil(h / 7).astype("int64")
    same_dow = np.vstack([_gather(y, positions, 7 * (first_week + k)) for k in range(_DOW_WEEKS)])
    with np.errstate(invalid="ignore"):
        counts = np.sum(~np.isnan(same_dow), axis=0)
        sums = np.nansum(same_dow, axis=0)
    out["origin_dow_mean"] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def training_horizons(n_rows: int, max_horizon: int, seed: int) -> IntArray:
    """Seeded random horizons ``1..max_horizon`` for training rows."""
    rng = np.random.default_rng(seed)
    return rng.integers(1, max_horizon + 1, size=n_rows, dtype="int64")


def forecast_horizons(df: pd.DataFrame, cutoff: pd.Timestamp) -> IntArray:
    """Days between ``cutoff`` and each row's date (history rows get 1; they are not predicted)."""
    days = (df["ds"] - cutoff).dt.days.to_numpy(dtype="int64")
    return np.maximum(days, 1)
