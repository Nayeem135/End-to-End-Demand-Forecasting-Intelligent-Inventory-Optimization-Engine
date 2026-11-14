"""Leakage tests for the complete feature pipeline (every feature family at once).

Rule under test: a feature for target date ``t`` may only use demand observed up to
``t - horizon``. Fixed lags, rolling windows and target encodings use the model horizon
``H``; origin features use each row's own horizon ``h <= H`` (the distance from its
forecast origin), so the binding limit for a row is ``t - h``.
"""

import numpy as np
import pandas as pd
import pytest

from demand_engine.config import LightGBMSettings
from demand_engine.data.splits import split_at
from demand_engine.features.builder import FeatureSpec, build_features, learn_categories
from demand_engine.features.lags import lag_feature_names
from demand_engine.features.origin import ORIGIN_FEATURES, forecast_horizons, training_horizons
from demand_engine.features.target_encoding import TE_FEATURES
from demand_engine.models.lgbm import LightGBMForecaster

H = 14


def _spec(df: pd.DataFrame, lags: list[int] | None = None, horizon: int = H) -> FeatureSpec:
    return FeatureSpec(
        horizon=horizon,
        lags=lags or [14, 21, 28],
        windows=[7, 28],
        categories=learn_categories(df),
        origin=True,
        target_encoding=True,
    )


def _target_columns(spec: FeatureSpec) -> list[str]:
    """Features derived from the target (calendar/price/ids are exogenous by construction)."""
    return [
        *lag_feature_names(spec.lags, spec.windows, spec.horizon),
        *ORIGIN_FEATURES,
        *TE_FEATURES,
    ]


def test_forecast_window_features_ignore_all_future_targets(small_long: pd.DataFrame) -> None:
    """Randomising every future target leaves every feature of the forecast window unchanged."""
    df = small_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    cutoff = df["ds"].max() - pd.Timedelta(days=H)
    future = df["ds"] > cutoff
    spec = _spec(df)
    horizons = forecast_horizons(df, cutoff)

    masked = df.assign(y=df["y"].where(~future))
    noisy = df.assign(y=df["y"].where(~future, np.random.default_rng(0).uniform(0, 1e4, len(df))))
    a = build_features(masked, spec, horizons).loc[future, spec.feature_names]
    b = build_features(noisy, spec, horizons).loc[future, spec.feature_names]

    pd.testing.assert_frame_equal(a, b)
    assert (
        a[_target_columns(spec)].notna().any().all()
    )  # the features are populated, not trivially NaN


@pytest.mark.parametrize("seed", range(4))
def test_training_features_use_only_data_up_to_t_minus_horizon(
    small_long: pd.DataFrame, seed: int
) -> None:
    """Change the target of ONE series on ONE day s: no feature dated before s + h may change.

    Checked for every row of every series (target encodings pool across series), with
    the row-specific limit h = min(H, origin horizon of the row).
    """
    rng = np.random.default_rng(seed)
    df = small_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    spec = _spec(df)
    horizons = training_horizons(len(df), H, seed)
    cols = _target_columns(spec)
    base = build_features(df, spec, horizons)[cols]

    victim = rng.choice(df["unique_id"].unique())
    days = df.loc[df["unique_id"] == victim, "ds"]
    s = days.iloc[rng.integers(30, len(days) - 30)]
    changed = df.copy()
    changed.loc[(changed["unique_id"] == victim) & (changed["ds"] == s), "y"] += 500.0
    after = build_features(changed, spec, horizons)[cols]

    limit = np.minimum(H, horizons)  # per-row look-back limit
    safe = (df["ds"] - s).dt.days.to_numpy() < limit
    pd.testing.assert_frame_equal(after[safe], base[safe])
    # Sensitivity check: the perturbation does reach features dated after s + h,
    # so the test is not passing because nothing depends on the target.
    later = ~safe & (df["ds"] > s).to_numpy()
    assert not after[later].fillna(-1).equals(base[later].fillna(-1))


def test_short_lags_are_leakage_free_when_horizon_allows(small_long: pd.DataFrame) -> None:
    """Lags 7/14/28 are valid for a 7-day horizon and pass the same perturbation test."""
    df = small_long.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    spec = _spec(df, lags=[7, 14, 28], horizon=7)
    horizons = training_horizons(len(df), 7, 1)
    cols = _target_columns(spec)
    base = build_features(df, spec, horizons)[cols]
    s = df["ds"].iloc[len(df) // 2]
    changed = df.assign(y=df["y"].where(df["ds"] != s, df["y"] + 500.0))
    after = build_features(changed, spec, horizons)[cols]
    safe = (df["ds"] - s).dt.days.to_numpy() < np.minimum(7, horizons)
    pd.testing.assert_frame_equal(after[safe], base[safe])


def test_lags_shorter_than_horizon_are_rejected() -> None:
    """A 7-day lag with a 28-day horizon would read the future: refused at every entry point."""
    with pytest.raises(ValueError, match="horizon"):
        LightGBMForecaster(LightGBMSettings(lags=[7, 14, 28]), horizon=28, seed=0)


def test_model_predictions_do_not_depend_on_future_targets(small_long: pd.DataFrame) -> None:
    """End to end: a fitted LightGBM (with target encodings) predicts identically whatever
    the true future values are."""
    cutoff = small_long["ds"].max() - pd.Timedelta(days=H)
    train, window = split_at(small_long, cutoff, H)
    settings = LightGBMSettings(n_estimators=15, num_leaves=7, min_data_in_leaf=5, lags=[14, 21],
                                rolling_windows=[7], target_encoding=True)  # fmt: skip
    model = LightGBMForecaster(settings, horizon=H, seed=1).fit(train)
    a = model.predict(window)
    b = model.predict(window.assign(y=1e6))
    pd.testing.assert_frame_equal(a, b)
    assert "te_item" in model.spec.feature_names  # type: ignore[union-attr]
