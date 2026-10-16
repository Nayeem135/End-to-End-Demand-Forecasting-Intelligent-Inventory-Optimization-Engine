"""SHAP explanations for the LightGBM forecaster.

Values come from ``shap.TreeExplainer`` (exact TreeSHAP). For the Tweedie/Poisson
objectives the model works on a log link, so contributions are additive in
*log-demand*: ``exp(base + sum(contributions)) = forecast``. A contribution of
+0.2 therefore means "this feature lifts the forecast by about 22%".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from demand_engine.data.schema import UNIQUE_ID
from demand_engine.models.lgbm import LightGBMForecaster
from demand_engine.types import FloatArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Explanation:
    """Global and per-series SHAP summaries."""

    global_importance: pd.DataFrame  # feature, mean_abs_shap
    local: pd.DataFrame  # unique_id, feature, shap_value, feature_value (top-k per series)
    base_value: float


def shap_values(model: LightGBMForecaster, x: pd.DataFrame) -> tuple[FloatArray, float]:
    """Exact TreeSHAP values of the point model for rows ``x``."""
    explainer = shap.TreeExplainer(model.boosters["yhat"])
    values = np.asarray(explainer.shap_values(x), dtype="float64")
    base = float(np.ravel(explainer.expected_value)[0])
    return values, base


def explain(
    model: LightGBMForecaster,
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    top_k: int = 8,
    max_rows: int = 10_000,
    seed: int = 42,
) -> Explanation:
    """Explain forecasts for ``rows`` (``unique_id, ds`` keys within ``frame``).

    Args:
        model: A fitted :class:`LightGBMForecaster`.
        frame: History plus forecast-window rows (``y`` may be NaN in the window).
        rows: Keys of the rows to explain (typically the forecast window).
        top_k: Number of strongest features kept per series.
        max_rows: Approximate cap on explained rows. Rows are sampled *per series*
            (at least one each), so every series gets an explanation.
        seed: Sampling seed.
    """
    feats = model.feature_matrix(frame)
    keyed = pd.concat([frame[[UNIQUE_ID, "ds"]], feats], axis=1)
    keyed = keyed.merge(rows[[UNIQUE_ID, "ds"]], on=[UNIQUE_ID, "ds"], how="inner")
    if len(keyed) > max_rows:
        per_series = max(1, max_rows // keyed[UNIQUE_ID].nunique())
        order = np.random.default_rng(seed).random(len(keyed))
        rank = pd.Series(order, index=keyed.index).groupby(keyed[UNIQUE_ID]).rank(method="first")
        keyed = keyed[rank <= per_series].sort_values([UNIQUE_ID, "ds"])
    x = keyed[feats.columns]
    values, base = shap_values(model, x)
    names = list(feats.columns)

    global_imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
    global_imp = global_imp.sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    contrib = pd.DataFrame(values, columns=names)
    contrib[UNIQUE_ID] = keyed[UNIQUE_ID].to_numpy()
    mean_contrib = contrib.groupby(UNIQUE_ID).mean()
    numeric_x = x.apply(lambda c: c.cat.codes if isinstance(c.dtype, pd.CategoricalDtype) else c)
    numeric_x = numeric_x.astype("float64")
    numeric_x[UNIQUE_ID] = keyed[UNIQUE_ID].to_numpy()
    mean_value = numeric_x.groupby(UNIQUE_ID).mean()

    local_rows = []
    for uid, series in mean_contrib.iterrows():
        top = series.abs().sort_values(ascending=False).index[:top_k]
        for feat in top:
            local_rows.append(
                {
                    UNIQUE_ID: uid,
                    "feature": feat,
                    "shap_value": float(series[feat]),
                    "feature_value": float(mean_value.loc[uid, feat]),
                }
            )
    logger.info("SHAP explained %d rows; top feature: %s", len(x), global_imp["feature"].iloc[0])
    return Explanation(global_imp, pd.DataFrame(local_rows), base)
