"""Common forecaster interface.

Every model follows the same two-step contract so that backtesting, the
inventory optimizer and the API can treat them interchangeably:

* ``fit(history)`` with the long demand table up to a cut-off;
* ``predict(future)`` with the exogenous rows (calendar, prices) of the forecast
  window. Any ``y`` column in ``future`` is dropped *before* the model sees it.

The output always has the columns ``unique_id, ds, yhat, yhat_lo, yhat_hi`` with
one row per requested ``(unique_id, ds)``, non-negative values and
``yhat_lo <= yhat <= yhat_hi``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from demand_engine.data.schema import DS, FORECAST_COLUMNS, UNIQUE_ID, Y


class ForecastContractError(RuntimeError):
    """Raised when a model returns output that violates the forecast contract."""


class Forecaster(ABC):
    """Abstract base class for all forecasting models."""

    name: str = "forecaster"

    def __init__(self, level: int = 80) -> None:
        """Create the model.

        Args:
            level: Central prediction-interval coverage in percent (e.g. 80 => P10..P90).
        """
        self.level = level
        self._fitted = False

    def fit(self, history: pd.DataFrame) -> Self:
        """Fit on the long demand table (rows up to the cut-off)."""
        if history.empty:
            raise ValueError("cannot fit on an empty history")
        if history[Y].isna().any():
            raise ValueError("history contains missing target values")
        self._fit(history.sort_values([UNIQUE_ID, DS]).reset_index(drop=True))
        self._fitted = True
        return self

    def predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Forecast every ``(unique_id, ds)`` row of ``future``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ForecastContractError: If the model output is malformed.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted before predict")
        request = future.drop(columns=[Y], errors="ignore").sort_values([UNIQUE_ID, DS])
        request = request.reset_index(drop=True)
        raw = self._predict(request)
        return self._enforce_contract(request, raw)

    def params(self) -> dict[str, object]:
        """Model name and hyper-parameters (logged to MLflow with every run)."""
        return {"model": self.name, "interval_level": self.level}

    @abstractmethod
    def _fit(self, history: pd.DataFrame) -> None:
        """Model-specific fitting."""

    @abstractmethod
    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        """Model-specific prediction returning at least the forecast columns."""

    def _enforce_contract(self, request: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FORECAST_COLUMNS if c not in raw.columns]
        if missing:
            raise ForecastContractError(f"{self.name}: output lacks columns {missing}")
        keys = request[[UNIQUE_ID, DS]]
        out = keys.merge(
            raw[list(FORECAST_COLUMNS)], on=[UNIQUE_ID, DS], how="left", validate="one_to_one"
        )
        values = out[["yhat", "yhat_lo", "yhat_hi"]]
        if values.isna().any().any():
            raise ForecastContractError(f"{self.name}: missing or NaN forecasts for some rows")
        arr = np.clip(values.to_numpy(dtype="float64"), 0.0, None)
        out["yhat"] = arr[:, 0]
        out["yhat_lo"] = np.minimum(arr[:, 1], arr[:, 0])
        out["yhat_hi"] = np.maximum(arr[:, 2], arr[:, 0])
        return out


def window_length(future: pd.DataFrame) -> int:
    """Number of forecast days, verifying all series request the same dates."""
    per_series = future.groupby(UNIQUE_ID)[DS].nunique()
    if per_series.nunique() != 1:
        raise ValueError("all series must request the same number of forecast days")
    return int(per_series.iloc[0])
