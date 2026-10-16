import pandas as pd
import pytest

from demand_engine.config import LightGBMSettings
from demand_engine.data.splits import split_at
from demand_engine.models.lgbm import LightGBMForecaster

H = 14
SETTINGS = LightGBMSettings(
    n_estimators=20, num_leaves=7, min_data_in_leaf=5, lags=[14, 21], rolling_windows=[7]
)


@pytest.fixture
def split(small_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = small_long["ds"].max() - pd.Timedelta(days=H)
    return split_at(small_long, cutoff, H)


def test_fit_predict_contract(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, window = split
    model = LightGBMForecaster(SETTINGS, horizon=H, seed=1).fit(train)
    out = model.predict(window)
    assert len(out) == len(window)
    assert (out["yhat_lo"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["yhat_hi"]).all()
    assert set(model.boosters) == {"yhat", "yhat_lo", "yhat_hi"}


def test_deterministic_given_seed(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, window = split
    a = LightGBMForecaster(SETTINGS, horizon=H, seed=3).fit(train).predict(window)
    b = LightGBMForecaster(SETTINGS, horizon=H, seed=3).fit(train).predict(window)
    pd.testing.assert_frame_equal(a, b)


def test_prediction_ignores_future_actuals(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, window = split
    model = LightGBMForecaster(SETTINGS, horizon=H, seed=1).fit(train)
    a = model.predict(window)
    b = model.predict(window.assign(y=1e6))
    pd.testing.assert_frame_equal(a, b)


def test_quantiles_from_level() -> None:
    lo, hi = LightGBMForecaster(SETTINGS, horizon=H, seed=1, level=80).quantiles
    assert (lo, hi) == pytest.approx((0.1, 0.9))


def test_feature_importance_normalised(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    model = LightGBMForecaster(SETTINGS, horizon=H, seed=1).fit(split[0])
    imp = model.feature_importance()
    assert imp["gain"].sum() == pytest.approx(1.0)
    assert imp["gain"].is_monotonic_decreasing


def test_rejects_short_lags() -> None:
    with pytest.raises(ValueError, match="horizon"):
        LightGBMForecaster(SETTINGS, horizon=28, seed=1)


def test_feature_matrix_requires_fit(small_long: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        LightGBMForecaster(SETTINGS, horizon=H, seed=1).feature_matrix(small_long)
