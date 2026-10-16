"""Shared pytest fixtures: tiny synthetic M5 data and settings pointing at tmp dirs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demand_engine.config import (
    BacktestSettings,
    DataSettings,
    LightGBMSettings,
    ModelSettings,
    PathsSettings,
    Settings,
)
from demand_engine.data.synthetic import SyntheticM5, generate_synthetic_m5, write_synthetic_m5
from demand_engine.data.transform import to_long

N_ITEMS = 12
N_DAYS = 420


@pytest.fixture(scope="session")
def synthetic_raw() -> SyntheticM5:
    """Synthetic raw M5 tables for one state (4 stores)."""
    return generate_synthetic_m5(n_items=N_ITEMS, n_days=N_DAYS, states=("CA",), seed=7)


@pytest.fixture(scope="session")
def long_df(synthetic_raw: SyntheticM5) -> pd.DataFrame:
    """Long demand table derived from the synthetic raw tables."""
    return to_long(synthetic_raw.sales, synthetic_raw.calendar, synthetic_raw.prices, state="CA")


@pytest.fixture
def small_long(long_df: pd.DataFrame) -> pd.DataFrame:
    """A copy of a handful of series, cheap enough for model tests."""
    ids = sorted(long_df["unique_id"].unique())[:6]
    return long_df[long_df["unique_id"].isin(ids)].reset_index(drop=True).copy()


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings sized for fast tests, writing into a temp directory."""
    return Settings(
        paths=PathsSettings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts"),
        data=DataSettings(state="CA", n_items=8, history_days=None),
        backtest=BacktestSettings(horizon=14, n_folds=2, n_test_windows=2, min_train_days=84),
        model=ModelSettings(
            moving_average_window=14,
            lgbm=LightGBMSettings(
                n_estimators=30, num_leaves=15, min_data_in_leaf=10, lags=[14, 21, 28]
            ),
        ),
        mlflow_tracking_uri="sqlite:///" + (tmp_path / "mlflow.db").as_posix(),
    )


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A raw directory filled with synthetic M5 CSV files."""
    target = tmp_path / "data" / "raw"
    write_synthetic_m5(target, n_items=N_ITEMS, n_days=N_DAYS, states=("CA",), seed=7)
    return target
