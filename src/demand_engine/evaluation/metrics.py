"""Point and probabilistic forecast accuracy metrics.

Scale-free metrics follow the M5 competition: RMSSE scales each series' RMSE by
the in-sample RMSE of a one-step naive forecast, and WRMSSE weights series by
their recent dollar sales, so accuracy on high-revenue items matters most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_engine.data.schema import DS, PRICE, UNIQUE_ID, Y
from demand_engine.types import FloatArray

_EPS = 1e-9


def mae(y: FloatArray, yhat: FloatArray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: FloatArray, yhat: FloatArray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y: FloatArray, yhat: FloatArray) -> float:
    """Mean forecast error ``yhat - y`` (positive = over-forecasting)."""
    return float(np.mean(yhat - y))


def naive_scale(y_train: FloatArray) -> float:
    """Mean squared one-step naive error on the training data (RMSSE denominator)."""
    if y_train.size < 2:
        return float("nan")
    return float(np.mean(np.diff(y_train) ** 2))


def rmsse(y: FloatArray, yhat: FloatArray, y_train: FloatArray) -> float:
    """Root mean squared scaled error; ``nan`` when the series is constant in training."""
    scale = naive_scale(y_train)
    if not np.isfinite(scale) or scale < _EPS:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2) / scale))


def pinball(y: FloatArray, q: FloatArray, tau: float) -> float:
    """Pinball (quantile) loss of quantile forecast ``q`` at level ``tau``."""
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def coverage(y: FloatArray, lo: FloatArray, hi: FloatArray) -> float:
    """Share of actuals falling inside ``[lo, hi]``."""
    return float(np.mean((y >= lo) & (y <= hi)))


def revenue_weights(history: pd.DataFrame, days: int = 28) -> pd.Series:
    """Series weights proportional to dollar sales over the last ``days`` days of ``history``."""
    start = history[DS].max() - pd.Timedelta(days=days - 1)
    recent = history[history[DS] >= start]
    revenue = (recent[Y] * recent[PRICE]).groupby(recent[UNIQUE_ID]).sum()
    total = revenue.sum()
    if total <= 0:
        return pd.Series(1.0 / len(revenue), index=revenue.index)
    return pd.Series(revenue / total, index=revenue.index)


def per_series_metrics(preds: pd.DataFrame, history: pd.DataFrame, level: int) -> pd.DataFrame:
    """Accuracy metrics per series, pooled over all folds in ``preds``.

    Args:
        preds: Forecasts with ``unique_id, ds, y, yhat, yhat_lo, yhat_hi``.
        history: Demand history strictly before the earliest forecast date
            (used for RMSSE scaling).
        level: Interval level in percent used for pinball loss quantiles.

    Returns:
        One row per series with mae, rmse, bias, rmsse, coverage and pinball.
    """
    tail = (1.0 - level / 100.0) / 2.0
    scales = history.groupby(UNIQUE_ID)[Y].apply(lambda s: naive_scale(s.to_numpy(dtype="float64")))
    rows = []
    for uid, g in preds.groupby(UNIQUE_ID, sort=True):
        y = g[Y].to_numpy(dtype="float64")
        yhat = g["yhat"].to_numpy(dtype="float64")
        lo = g["yhat_lo"].to_numpy(dtype="float64")
        hi = g["yhat_hi"].to_numpy(dtype="float64")
        scale = float(scales.get(uid, float("nan")))
        mse = float(np.mean((y - yhat) ** 2))
        rows.append(
            {
                UNIQUE_ID: uid,
                "mae": mae(y, yhat),
                "rmse": float(np.sqrt(mse)),
                "bias": bias(y, yhat),
                "rmsse": float(np.sqrt(mse / scale)) if scale > _EPS else float("nan"),
                "coverage": coverage(y, lo, hi),
                "pinball": 0.5 * (pinball(y, lo, tail) + pinball(y, hi, 1.0 - tail)),
            }
        )
    return pd.DataFrame(rows)


def summarize(per_series: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    """Aggregate per-series metrics into a single scorecard.

    ``wrmsse`` is the revenue-weighted RMSSE (weights renormalised over series
    with a finite RMSSE); other metrics are simple means across series.
    """
    df = per_series.set_index(UNIQUE_ID)
    valid = df["rmsse"].notna()
    w = weights.reindex(df.index).fillna(0.0)[valid]
    wrmsse = float((df.loc[valid, "rmsse"] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return {
        "wrmsse": wrmsse,
        "rmsse": float(df["rmsse"].mean()),
        "mae": float(df["mae"].mean()),
        "rmse": float(df["rmse"].mean()),
        "bias": float(df["bias"].mean()),
        "coverage": float(df["coverage"].mean()),
        "pinball": float(df["pinball"].mean()),
    }
