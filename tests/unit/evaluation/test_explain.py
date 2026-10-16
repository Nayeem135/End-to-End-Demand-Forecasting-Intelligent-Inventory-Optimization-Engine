import numpy as np
import pandas as pd
import pytest

from demand_engine.config import LightGBMSettings
from demand_engine.data.splits import split_at
from demand_engine.evaluation.explain import explain, shap_values
from demand_engine.models.lgbm import LightGBMForecaster

H = 14


@pytest.fixture
def fitted(small_long: pd.DataFrame) -> tuple[LightGBMForecaster, pd.DataFrame, pd.DataFrame]:
    cutoff = small_long["ds"].max() - pd.Timedelta(days=H)
    train, window = split_at(small_long, cutoff, H)
    settings = LightGBMSettings(
        n_estimators=20, num_leaves=7, min_data_in_leaf=5, lags=[14, 21], rolling_windows=[7]
    )
    model = LightGBMForecaster(settings, horizon=H, seed=1).fit(train)
    frame = (
        pd.concat([train, window.assign(y=np.nan)])
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return model, frame, window


def test_shap_is_additive_in_raw_score(
    fitted: tuple[LightGBMForecaster, pd.DataFrame, pd.DataFrame],
) -> None:
    model, frame, _ = fitted
    x = model.feature_matrix(frame).tail(50)
    values, base = shap_values(model, x)
    raw = model.boosters["yhat"].predict(x, raw_score=True)
    assert values.sum(axis=1) + base == pytest.approx(raw, abs=1e-6)


def test_explain_outputs(fitted: tuple[LightGBMForecaster, pd.DataFrame, pd.DataFrame]) -> None:
    model, frame, window = fitted
    ex = explain(model, frame, window, top_k=3)
    assert ex.global_importance["mean_abs_shap"].is_monotonic_decreasing
    assert set(ex.global_importance["feature"]) == set(model.spec.feature_names)  # type: ignore[union-attr]
    counts = ex.local.groupby("unique_id").size()
    assert (counts == 3).all()
    assert len(counts) == window["unique_id"].nunique()


def test_explain_sampling_cap(
    fitted: tuple[LightGBMForecaster, pd.DataFrame, pd.DataFrame],
) -> None:
    model, frame, window = fitted
    ex = explain(model, frame, window, max_rows=10, seed=0)
    # one row per series at minimum: every series is still explained
    assert ex.local["unique_id"].nunique() == window["unique_id"].nunique()
