"""Baseline and statistical forecasters backed by Nixtla ``statsforecast``.

All models get distribution-free *conformal* prediction intervals, so even point
methods such as Croston or a moving average produce calibrated uncertainty bands.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    TSB,
    AutoETS,
    CrostonOptimized,
    Naive,
    SeasonalNaive,
    WindowAverage,
)
from statsforecast.utils import ConformalIntervals

from demand_engine.config import CONFORMAL_WINDOWS
from demand_engine.data.schema import DS, UNIQUE_ID, Y
from demand_engine.models.base import Forecaster, window_length

logger = logging.getLogger(__name__)


class StatsForecastModel(Forecaster):
    """Wraps one ``statsforecast`` model behind the :class:`Forecaster` interface."""

    def __init__(
        self,
        name: str,
        model_factory: Callable[[], Any],
        level: int = 80,
        conformal: bool = True,
        hyperparams: dict[str, object] | None = None,
    ) -> None:
        """Create the wrapper.

        Args:
            name: Public model name (used in reports and the API).
            model_factory: Zero-argument callable returning a fresh statsforecast model.
            level: Prediction interval level in percent.
            conformal: Use conformal intervals; ``False`` uses the model's native
                (likelihood-based) intervals, which is much faster for ETS.
            hyperparams: The model's settings, reported by :meth:`params`.
        """
        super().__init__(level=level)
        self.name = name
        self._factory = model_factory
        self._conformal = conformal
        self._hyperparams = dict(hyperparams or {})
        self._history: pd.DataFrame | None = None

    def params(self) -> dict[str, object]:
        """Base parameters plus the statsforecast model settings and interval method."""
        intervals = f"conformal({CONFORMAL_WINDOWS} windows)" if self._conformal else "native"
        return super().params() | self._hyperparams | {"intervals": intervals}

    def _fit(self, history: pd.DataFrame) -> None:
        # statsforecast fits and forecasts in one call; we keep the history and run it in predict.
        self._history = history[[UNIQUE_ID, DS, Y]].copy()

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        assert self._history is not None  # guaranteed by Forecaster.predict
        h = window_length(future)
        history = self._history[self._history[UNIQUE_ID].isin(future[UNIQUE_ID].unique())]
        model = self._factory()
        # n_jobs=1: process pools are fragile on Windows (spawn) and the models are fast.
        sf = StatsForecast(models=[model], freq="D", n_jobs=1, fallback_model=Naive())
        intervals = (
            ConformalIntervals(h=h, n_windows=CONFORMAL_WINDOWS) if self._conformal else None
        )
        fc = sf.forecast(h=h, df=history, level=[self.level], prediction_intervals=intervals)
        alias = str(getattr(model, "alias", type(model).__name__))
        fc = fc.rename(
            columns={
                alias: "yhat",
                f"{alias}-lo-{self.level}": "yhat_lo",
                f"{alias}-hi-{self.level}": "yhat_hi",
            }
        )
        fc[DS] = pd.to_datetime(fc[DS])
        return pd.DataFrame(fc)


def make_naive(level: int = 80) -> Forecaster:
    """Last observed value carried forward."""
    return StatsForecastModel("naive", Naive, level=level)


def make_seasonal_naive(season_length: int = 7, level: int = 80) -> Forecaster:
    """Value from the same weekday last week."""
    return StatsForecastModel(
        "seasonal_naive",
        lambda: SeasonalNaive(season_length),
        level=level,
        hyperparams={"season_length": season_length},
    )


def make_moving_average(window: int = 28, level: int = 80) -> Forecaster:
    """Flat forecast equal to the mean of the last ``window`` days."""
    return StatsForecastModel(
        "moving_average", lambda: WindowAverage(window), level=level, hyperparams={"window": window}
    )


def make_auto_ets(season_length: int = 7, level: int = 80) -> Forecaster:
    """Exponential smoothing with additive error and weekly seasonality (ETS(A,N,A)).

    The model form is fixed rather than searched: daily SKU demand rarely shows a
    trend over a 28-day horizon, and fixing it is ~7x faster than a full search.
    """
    return StatsForecastModel(
        "ets",
        lambda: AutoETS(season_length=season_length, model="ANA"),
        level=level,
        conformal=False,
        hyperparams={"season_length": season_length, "ets_model": "ANA"},
    )


def make_croston(level: int = 80) -> Forecaster:
    """Croston's method with optimised smoothing, designed for intermittent demand."""
    return StatsForecastModel(
        "croston", CrostonOptimized, level=level, hyperparams={"variant": "optimized"}
    )


def make_tsb(level: int = 80) -> Forecaster:
    """Teunter-Syntetos-Babai: intermittent demand with obsolescence-aware probability."""
    return StatsForecastModel(
        "tsb",
        lambda: TSB(alpha_d=0.1, alpha_p=0.1),
        level=level,
        hyperparams={"alpha_d": 0.1, "alpha_p": 0.1},
    )
