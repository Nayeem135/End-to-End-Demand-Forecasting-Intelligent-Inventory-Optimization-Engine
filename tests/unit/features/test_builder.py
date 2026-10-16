import numpy as np
import pandas as pd
import pytest

from demand_engine.features.builder import FeatureSpec, build_features, learn_categories


def test_build_features_has_all_columns_and_stable_codes(small_long: pd.DataFrame) -> None:
    spec = FeatureSpec(
        horizon=14, lags=[14, 28], windows=[7], categories=learn_categories(small_long)
    )
    out = build_features(small_long, spec, np.ones(len(small_long), dtype="int64"))
    assert set(spec.feature_names) <= set(out.columns)
    assert len(spec.feature_names) == len(set(spec.feature_names))
    assert isinstance(out["item_id"].dtype, pd.CategoricalDtype)

    # Same vocabulary on a subset -> identical integer codes for the same item.
    subset = small_long[small_long["unique_id"] == small_long["unique_id"].iloc[-1]]
    out_sub = build_features(
        subset.reset_index(drop=True), spec, np.ones(len(subset), dtype="int64")
    )
    item = subset["item_id"].iloc[0]
    code_full = out.loc[out["item_id"] == item, "item_id"].cat.codes.iloc[0]
    assert out_sub["item_id"].cat.codes.iloc[0] == code_full


def test_learn_categories_includes_none_event(small_long: pd.DataFrame) -> None:
    vocab = learn_categories(small_long)
    assert "none" in vocab["event_type"]
    assert vocab["store_id"] == sorted(small_long["store_id"].unique())


def test_origin_features_need_horizons(small_long: pd.DataFrame) -> None:
    spec = FeatureSpec(horizon=14, lags=[14], windows=[7], categories=learn_categories(small_long))
    with pytest.raises(ValueError, match="horizons"):
        build_features(small_long, spec)
    no_origin = FeatureSpec(
        horizon=14, lags=[14], windows=[7], categories=spec.categories, origin=False
    )
    assert "horizon" not in no_origin.feature_names
    assert "origin_last" not in build_features(small_long, no_origin).columns
