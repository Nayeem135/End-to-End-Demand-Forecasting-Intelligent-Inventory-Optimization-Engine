"""Forecast-error statistics over the protection period, estimated from backtests.

Instead of assuming i.i.d. daily errors (``sigma_P = sqrt(P) * sigma``) we measure the
error of *cumulative* demand over ``P`` consecutive days directly from
validation-fold forecasts. This captures error autocorrelation. Bias and spread
are estimated separately:

* ``bias``  = mean of ``sum(y - yhat)`` over P days (positive = the model under-forecasts);
* ``sigma`` = standard deviation of that cumulative error.

The reorder point then becomes ``forecast_P + bias + z * sigma``: a systematic
under-forecast is corrected once instead of being inflated by ``z``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_engine.data.schema import DS, UNIQUE_ID, Y

MIN_WINDOWS = 5


def protection_error_stats(preds: pd.DataFrame, days: int) -> pd.DataFrame:
    """Bias and standard deviation of ``days``-day cumulative forecast errors per series.

    Args:
        preds: Backtest forecasts with ``unique_id, cutoff, ds, y, yhat``.
        days: Protection period length ``P`` in days.

    Returns:
        Frame indexed by ``unique_id`` with ``bias`` and ``sigma``. Series with fewer
        than :data:`MIN_WINDOWS` complete windows fall back to the i.i.d. scaling of
        daily errors (``P * mean``, ``sqrt(P) * std``).
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    df = preds.sort_values([UNIQUE_ID, "cutoff", DS])
    err = (df[Y] - df["yhat"]).astype("float64")
    blocks = [df[UNIQUE_ID], df["cutoff"]]
    cum = err.groupby(blocks, sort=False).rolling(days).sum().reset_index(level=[0, 1], drop=True)
    windows = pd.DataFrame({UNIQUE_ID: df[UNIQUE_ID], "cum": cum}).dropna()
    grouped = windows.groupby(UNIQUE_ID)["cum"]
    stats = pd.DataFrame(
        {"bias": grouped.mean(), "sigma": grouped.std(ddof=0), "n": grouped.size()}
    )

    daily = err.groupby(df[UNIQUE_ID])
    fallback = pd.DataFrame(
        {"bias": days * daily.mean(), "sigma": np.sqrt(days) * daily.std(ddof=0)}
    )
    stats = stats.reindex(fallback.index)
    short = stats["n"].fillna(0) < MIN_WINDOWS
    stats.loc[short, ["bias", "sigma"]] = fallback.loc[short, ["bias", "sigma"]]
    return stats[["bias", "sigma"]].fillna(0.0)
