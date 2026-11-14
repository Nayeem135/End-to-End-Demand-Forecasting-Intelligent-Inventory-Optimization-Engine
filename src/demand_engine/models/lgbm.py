"""Global LightGBM forecaster (one model across all SKU-store series).

* Point forecast: Tweedie objective, well suited to zero-inflated, right-skewed
  retail demand.
* Uncertainty: two additional quantile models give ``yhat_lo``/``yhat_hi`` for the
  configured central interval (e.g. P10/P90 for an 80% interval).
* Direct multi-horizon: all target-derived features use lags >= horizon, so one
  model forecasts the full window without recursion (see :mod:`demand_engine.features.lags`).
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from demand_engine.config import LightGBMSettings
from demand_engine.data.schema import DS, UNIQUE_ID, Y
from demand_engine.features.builder import FeatureSpec, build_features, learn_categories
from demand_engine.features.origin import forecast_horizons, training_horizons
from demand_engine.models.base import Forecaster, window_length

logger = logging.getLogger(__name__)


class LightGBMForecaster(Forecaster):
    """Global gradient-boosted tree forecaster with quantile intervals."""

    name = "lightgbm"

    def __init__(
        self,
        settings: LightGBMSettings,
        horizon: int,
        seed: int,
        level: int = 80,
        quantile_models: bool = True,
    ) -> None:
        """Create the forecaster.

        Args:
            settings: LightGBM hyper-parameters and feature lags/windows.
            horizon: Forecast horizon in days (lags must be >= horizon).
            seed: Random seed for bagging / feature sampling.
            level: Central prediction-interval level in percent.
            quantile_models: Train the P-lo/P-hi quantile models. ``False`` trains only the
                point model (intervals collapse to the point forecast) — used for fast
                hyper-parameter tuning, where only point accuracy is scored.
        """
        super().__init__(level=level)
        if min(settings.lags) < horizon:
            raise ValueError("all lags must be >= horizon for direct forecasting")
        self.settings = settings
        self.horizon = horizon
        self.seed = seed
        self.quantile_models = quantile_models
        self.spec: FeatureSpec | None = None
        self.boosters: dict[str, lgb.Booster] = {}
        self._history: pd.DataFrame | None = None

    def _params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        s = self.settings
        params: dict[str, Any] = {
            "objective": objective,
            "learning_rate": s.learning_rate,
            "num_leaves": s.num_leaves,
            "min_data_in_leaf": s.min_data_in_leaf,
            "feature_fraction": s.feature_fraction,
            "bagging_fraction": s.bagging_fraction,
            "bagging_freq": s.bagging_freq,
            "lambda_l2": s.lambda_l2,
            "seed": self.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 0,
            "verbose": -1,
        }
        if objective == "tweedie":
            params["tweedie_variance_power"] = s.tweedie_variance_power
        if alpha is not None:
            params["alpha"] = alpha
        return params

    @property
    def quantiles(self) -> tuple[float, float]:
        """Lower and upper quantile levels implied by :attr:`level`."""
        tail = (1.0 - self.level / 100.0) / 2.0
        return tail, 1.0 - tail

    def _fit(self, history: pd.DataFrame) -> None:
        self._history = history
        self.spec = FeatureSpec(
            horizon=self.horizon,
            lags=self.settings.lags,
            windows=self.settings.rolling_windows,
            categories=learn_categories(history),
            target_encoding=self.settings.target_encoding,
        )
        horizons = training_horizons(len(history), self.horizon, self.seed)
        feats = build_features(history, self.spec, horizons)
        # Rows without even the shortest lag carry almost no signal; drop them.
        train = feats[feats[f"lag_{min(self.settings.lags)}"].notna()]
        x, y = train[self.spec.feature_names], train[Y]
        logger.info("Training LightGBM on %d rows x %d features", len(x), x.shape[1])
        lo, hi = self.quantiles
        objectives: dict[str, tuple[str, float | None, int]] = {
            "yhat": (self.settings.objective, None, self.settings.n_estimators),
        }
        if self.quantile_models:
            objectives["yhat_lo"] = ("quantile", lo, max(1, self.settings.n_estimators // 2))
            objectives["yhat_hi"] = ("quantile", hi, max(1, self.settings.n_estimators // 2))
        for key, (objective, alpha, rounds) in objectives.items():
            dataset = lgb.Dataset(x, label=y, free_raw_data=False)
            self.boosters[key] = lgb.train(
                self._params(objective, alpha), dataset, num_boost_round=rounds
            )

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        assert self._history is not None
        assert self.spec is not None
        window_length(future)
        frame = pd.concat([self._history, future.assign(**{Y: float("nan")})], ignore_index=True)
        frame = frame.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
        cutoff = self._history[DS].max()
        feats = build_features(frame, self.spec, forecast_horizons(frame, cutoff))
        rows = feats.merge(future[[UNIQUE_ID, DS]], on=[UNIQUE_ID, DS], how="inner")
        x = rows[self.spec.feature_names]
        out = rows[[UNIQUE_ID, DS]].copy()
        for key, booster in self.boosters.items():
            out[key] = np.asarray(booster.predict(x), dtype="float64")
        if not self.quantile_models:
            out["yhat_lo"] = out["yhat_hi"] = out["yhat"]
        return out

    def params(self) -> dict[str, object]:
        """All LightGBM hyper-parameters, feature settings, horizon and seed."""
        return (
            super().params()
            | self.settings.model_dump()
            | {"horizon": self.horizon, "seed": self.seed, "quantile_models": self.quantile_models}
        )

    def feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Model inputs for ``frame`` (history + window, used for SHAP explanations).

        The forecast origin is the last date with a known target in ``frame``.
        """
        if self.spec is None:
            raise RuntimeError("model is not fitted")
        cutoff = frame.loc[frame[Y].notna(), DS].max()
        feats = build_features(frame, self.spec, forecast_horizons(frame, cutoff))
        return feats[self.spec.feature_names]

    def feature_importance(self) -> pd.DataFrame:
        """Gain-based importance of the point model, normalised to sum to 1."""
        booster = self.boosters["yhat"]
        gain = booster.feature_importance(importance_type="gain")
        imp = pd.DataFrame({"feature": booster.feature_name(), "gain": gain})
        imp["gain"] = imp["gain"] / imp["gain"].sum()
        return imp.sort_values("gain", ascending=False, ignore_index=True)
