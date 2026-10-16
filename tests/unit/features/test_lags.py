import numpy as np
import pandas as pd
import pytest

from demand_engine.features.lags import add_lag_features, lag_feature_names

H = 7
LAGS = [7, 14]
WINDOWS = [3, 7]


def _frame() -> pd.DataFrame:
    n = 40
    return pd.DataFrame(
        {
            "unique_id": ["a"] * n + ["b"] * n,
            "ds": list(pd.date_range("2020-01-01", periods=n)) * 2,
            "y": np.r_[np.arange(n, dtype=float), np.arange(n, dtype=float) * 10],
        }
    )


def test_lag_values_exact() -> None:
    out = add_lag_features(_frame(), LAGS, WINDOWS, H)
    a = out[out["unique_id"] == "a"].reset_index(drop=True)
    assert np.isnan(a.loc[6, "lag_7"])
    assert a.loc[7, "lag_7"] == 0.0
    assert a.loc[20, "lag_14"] == 6.0
    # rolling mean over y shifted by H: day 20 -> mean(y[11..13]) for window 3
    assert a.loc[20, f"roll_mean_{H}_3"] == pytest.approx(np.mean([11, 12, 13]))


def test_no_cross_series_bleed() -> None:
    out = add_lag_features(_frame(), LAGS, WINDOWS, H)
    b = out[out["unique_id"] == "b"].reset_index(drop=True)
    assert b.loc[:6, "lag_7"].isna().all()


def test_feature_names_match_columns() -> None:
    out = add_lag_features(_frame(), LAGS, WINDOWS, H)
    assert set(lag_feature_names(LAGS, WINDOWS, H)) <= set(out.columns)


def test_rejects_lag_shorter_than_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_lag_features(_frame(), [3], WINDOWS, H)


def test_no_leakage_future_targets_do_not_change_window_features() -> None:
    """Features for the forecast window must be identical whatever the future holds."""
    df = _frame()
    cutoff = pd.Timestamp("2020-01-30")
    in_window = (df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=H))
    future = df["ds"] > cutoff

    masked = df.copy()
    masked.loc[future, "y"] = np.nan
    perturbed = df.copy()
    perturbed.loc[future, "y"] = np.random.default_rng(0).uniform(0, 1e6, future.sum())

    cols = lag_feature_names(LAGS, WINDOWS, H)
    feats_masked = add_lag_features(masked, LAGS, WINDOWS, H).loc[in_window, cols]
    feats_perturbed = add_lag_features(perturbed, LAGS, WINDOWS, H).loc[in_window, cols]
    pd.testing.assert_frame_equal(feats_masked, feats_perturbed)
    assert feats_masked.notna().all().all()
