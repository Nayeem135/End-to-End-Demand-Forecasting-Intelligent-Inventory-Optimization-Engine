import numpy as np
import pandas as pd
import pytest

from demand_engine.features.origin import (
    ORIGIN_FEATURES,
    add_origin_features,
    forecast_horizons,
    training_horizons,
)


def _frame(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["a"] * n + ["b"] * n,
            "ds": list(pd.date_range("2020-01-01", periods=n)) * 2,
            "y": np.r_[np.arange(n, dtype=float), 1000 + np.arange(n, dtype=float)],
        }
    )


def test_values_hand_checked() -> None:
    df = _frame()
    h = np.full(len(df), 3, dtype="int64")
    out = add_origin_features(df, h)
    a = out[out["unique_id"] == "a"].reset_index(drop=True)
    assert a.loc[20, "horizon"] == 3
    assert a.loc[20, "origin_last"] == 17.0  # y at t-3
    assert a.loc[20, "origin_mean_7"] == pytest.approx(np.mean(np.arange(11, 18)))
    # same weekday, first week >= h: t-7, t-14, t-21, t-28 -> only t-7 and t-14 exist for t=20
    assert a.loc[20, "origin_dow_mean"] == pytest.approx(np.mean([13.0, 6.0]))
    assert set(ORIGIN_FEATURES) <= set(out.columns)


def test_no_cross_series_bleed() -> None:
    out = add_origin_features(_frame(), np.full(120, 5, dtype="int64"))
    b = out[out["unique_id"] == "b"].reset_index(drop=True)
    assert b.loc[:4, "origin_last"].isna().all()
    assert b.loc[5, "origin_last"] == 1000.0


def test_no_leakage_window_features_ignore_future_targets() -> None:
    df = _frame()
    cutoff = pd.Timestamp("2020-02-10")
    future = df["ds"] > cutoff
    horizons = forecast_horizons(df, cutoff)
    masked = df.copy()
    masked.loc[future, "y"] = np.nan
    perturbed = df.copy()
    perturbed.loc[future, "y"] = np.random.default_rng(1).uniform(0, 1e6, future.sum())
    cols = list(ORIGIN_FEATURES)
    a = add_origin_features(masked, horizons).loc[future, cols]
    b = add_origin_features(perturbed, horizons).loc[future, cols]
    pd.testing.assert_frame_equal(a, b)
    assert a["origin_last"].notna().all()


def test_forecast_and_training_horizons() -> None:
    df = _frame(10)
    h = forecast_horizons(df, pd.Timestamp("2020-01-05"))
    assert h[:6].tolist() == [1, 1, 1, 1, 1, 1]  # history rows clipped to 1
    assert h[6:10].tolist() == [2, 3, 4, 5]
    t = training_horizons(1000, 28, seed=3)
    assert t.min() >= 1
    assert t.max() <= 28
    assert (t == training_horizons(1000, 28, seed=3)).all()


def test_validation() -> None:
    with pytest.raises(ValueError, match="one entry per row"):
        add_origin_features(_frame(), np.ones(3, dtype="int64"))
    with pytest.raises(ValueError, match=">= 1"):
        add_origin_features(_frame(), np.zeros(120, dtype="int64"))
