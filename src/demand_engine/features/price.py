"""Price features. Prices are set by the retailer in advance, so they are known exogenously."""

from __future__ import annotations

import pandas as pd

from demand_engine.data.schema import PRICE, UNIQUE_ID

PRICE_FEATURES = ("sell_price", "price_norm", "price_change_7", "price_rel_store_dept")


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with relative-price features.

    * ``price_norm``: price divided by the series' mean price (promo depth).
    * ``price_change_7``: relative change vs. 7 days earlier (promotion start/end).
    * ``price_rel_store_dept``: price relative to the department mean in the same store/day.

    Expects rows sorted by ``unique_id, ds``.
    """
    out = df.copy()
    by_series = out.groupby(UNIQUE_ID, sort=False)[PRICE]
    out["price_norm"] = out[PRICE] / by_series.transform("mean")
    out["price_change_7"] = (out[PRICE] / by_series.shift(7) - 1.0).fillna(0.0)
    dept_mean = out.groupby(["store_id", "dept_id", "ds"], sort=False)[PRICE].transform("mean")
    out["price_rel_store_dept"] = out[PRICE] / dept_mean
    return out
