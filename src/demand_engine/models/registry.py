"""Build forecasters by name from configuration."""

from __future__ import annotations

from collections.abc import Callable

from demand_engine.config import ModelName, Settings
from demand_engine.models import statistical as st
from demand_engine.models.base import Forecaster
from demand_engine.models.lgbm import LightGBMForecaster

ForecasterFactory = Callable[[], Forecaster]


def model_factory(name: ModelName, settings: Settings) -> ForecasterFactory:
    """Return a zero-argument factory producing a fresh, unfitted model.

    Args:
        name: Registered model name.
        settings: Global settings (horizon, seeds, hyper-parameters).

    Raises:
        KeyError: If ``name`` is not a registered model.
    """
    m = settings.model
    level = m.interval_level
    factories: dict[str, ForecasterFactory] = {
        "naive": lambda: st.make_naive(level),
        "seasonal_naive": lambda: st.make_seasonal_naive(m.season_length, level),
        "moving_average": lambda: st.make_moving_average(m.moving_average_window, level),
        "croston": lambda: st.make_croston(level),
        "tsb": lambda: st.make_tsb(level),
        "ets": lambda: st.make_auto_ets(m.season_length, level),
        "lightgbm": lambda: LightGBMForecaster(
            m.lgbm, horizon=settings.backtest.horizon, seed=settings.seed, level=level
        ),
    }
    if name not in factories:
        raise KeyError(f"unknown model '{name}'; choose from {sorted(factories)}")
    return factories[name]
