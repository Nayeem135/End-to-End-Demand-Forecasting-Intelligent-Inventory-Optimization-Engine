"""Target-derived (lag / rolling) features with an explicit leakage guarantee.

We use *direct* multi-horizon forecasting with a single model: every target-based
feature for day ``t`` only uses demand from day ``t - horizon`` or earlier. When we
forecast the window ``(cutoff, cutoff + horizon]``, all features are therefore
computable from history up to the cut-off and no recursive feeding of predictions
is needed. ``tests/unit/features/test_lags.py`` verifies this empirically.
"""

from __future__ import annotations

import pandas as pd

from demand_engine.data.schema import UNIQUE_ID, Y


def lag_feature_names(lags: list[int], windows: list[int], horizon: int) -> list[str]:
    """Names of the features produced by :func:`add_lag_features`."""
    names = [f"lag_{lag}" for lag in lags]
    for w in windows:
        names += [f"roll_mean_{horizon}_{w}", f"roll_std_{horizon}_{w}"]
    names.append(f"nonzero_rate_{horizon}_28")
    return names


def add_lag_features(
    df: pd.DataFrame, lags: list[int], windows: list[int], horizon: int
) -> pd.DataFrame:
    """Return a copy of ``df`` with lag and rolling-window demand features.

    Args:
        df: Long table sorted by ``unique_id, ds`` with contiguous daily rows; future
            rows may have ``y = NaN``.
        lags: Lags in days; each must be ``>= horizon``.
        windows: Rolling window lengths applied to demand shifted by ``horizon``.
        horizon: Forecast horizon in days.

    Raises:
        ValueError: If any lag is shorter than the horizon (would leak the future).
    """
    if min(lags) < horizon:
        raise ValueError(f"lags {lags} must all be >= horizon {horizon}")
    out = df.copy()
    ids = out[UNIQUE_ID]
    grouped = out.groupby(UNIQUE_ID, sort=False)[Y]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(horizon)
    by_id = shifted.groupby(ids, sort=False)
    for w in windows:
        roll = by_id.rolling(w, min_periods=1)
        out[f"roll_mean_{horizon}_{w}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"roll_std_{horizon}_{w}"] = roll.std().reset_index(level=0, drop=True)
    nonzero = (shifted > 0).astype("float64").where(shifted.notna())
    out[f"nonzero_rate_{horizon}_28"] = (
        nonzero.groupby(ids, sort=False)
        .rolling(28, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return out
