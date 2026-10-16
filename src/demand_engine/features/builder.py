"""Assemble the full model feature matrix."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from demand_engine.features.calendar import CALENDAR_FEATURES, add_calendar_features
from demand_engine.features.lags import add_lag_features, lag_feature_names
from demand_engine.features.origin import ORIGIN_FEATURES, add_origin_features
from demand_engine.features.price import PRICE_FEATURES, add_price_features
from demand_engine.types import IntArray

CATEGORICAL_FEATURES = ("item_id", "dept_id", "cat_id", "store_id", "event_type")


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration of the feature set (all fixed lags must be >= horizon)."""

    horizon: int
    lags: list[int]
    windows: list[int]
    categories: dict[str, list[str]] = field(default_factory=dict)
    origin: bool = True

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of model input columns (guaranteed unique)."""
        names = [
            *CATEGORICAL_FEATURES,
            *CALENDAR_FEATURES,
            *PRICE_FEATURES,
            *lag_feature_names(self.lags, self.windows, self.horizon),
            *(ORIGIN_FEATURES if self.origin else ()),
        ]
        if len(names) != len(set(names)):
            raise ValueError("duplicate feature names in feature specification")
        return names


def learn_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    """Collect category vocabularies so training and prediction share one encoding."""
    vocab = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES[:-1]}
    vocab["event_type"] = sorted({*df["event_type_1"].astype(str).unique(), "none"})
    return vocab


def build_features(
    df: pd.DataFrame, spec: FeatureSpec, horizons: IntArray | None = None
) -> pd.DataFrame:
    """Return ``df`` plus all model features, with categoricals encoded consistently.

    Args:
        df: Long demand table (history and optionally future rows with ``y = NaN``),
            sorted by ``unique_id, ds``.
        spec: Feature configuration, including category vocabularies.
        horizons: Per-row forecast horizon; required when ``spec.origin`` is set.
    """
    out = add_calendar_features(df)
    out = add_price_features(out)
    out = add_lag_features(out, spec.lags, spec.windows, spec.horizon)
    if spec.origin:
        if horizons is None:
            raise ValueError("origin features need per-row horizons")
        out = add_origin_features(out, horizons)
    for col in CATEGORICAL_FEATURES:
        cats = spec.categories.get(col)
        out[col] = pd.Categorical(out[col].astype(str), categories=cats)
    return out
