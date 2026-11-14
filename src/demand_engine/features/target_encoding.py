"""Leakage-free target encodings: expanding group means lagged by the horizon.

A target encoding summarises a categorical group by its average demand (e.g. "how much
does this item sell per day across all stores"). Computed naively over the whole history
it leaks: a row's own target and future targets enter its feature. Here each encoding at
date ``t`` is the mean of ``y`` over **all rows of the group dated on or before
``t - horizon``** — an expanding mean, so it uses as much history as is legitimately
available and nothing after the cut-off. Rows with no history that old get ``NaN``
(LightGBM handles missing values natively).

Groups (all observable for future dates):

* ``te_item``        - the item across all stores (shares information between stores);
* ``te_store_dept``  - store x department (local level of a product family);
* ``te_dept_wday``   - department x weekday (a weekly profile per product family).
"""

from __future__ import annotations

import pandas as pd

from demand_engine.data.schema import DS, Y

TE_GROUPS: dict[str, tuple[str, ...]] = {
    "te_item": ("item_id",),
    "te_store_dept": ("store_id", "dept_id"),
    "te_dept_wday": ("dept_id", "wday"),
}
TE_FEATURES = tuple(TE_GROUPS)


def _expanding_group_mean(df: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """Per group and date: mean of ``y`` over all known rows of the group up to that date."""
    known = df.loc[df[Y].notna(), [*keys, DS, Y]]
    daily = (
        known.groupby([*keys, DS], sort=True, observed=True)[Y].agg(["sum", "size"]).reset_index()
    )
    cum = daily.groupby(list(keys), sort=False, observed=True)[["sum", "size"]].cumsum()
    daily["te"] = cum["sum"] / cum["size"]
    return daily[[*keys, DS, "te"]]


def add_target_encodings(
    df: pd.DataFrame, horizon: int, groups: dict[str, tuple[str, ...]] | None = None
) -> pd.DataFrame:
    """Return a copy of ``df`` with target-encoding columns.

    Args:
        df: Long table with ``ds``, ``y`` (``NaN`` for unknown future rows) and the
            grouping columns.
        horizon: Forecast horizon ``h``; the encoding at date ``t`` uses data up to ``t - h``.
        groups: Mapping feature name -> grouping columns (defaults to :data:`TE_GROUPS`).

    Raises:
        ValueError: If ``horizon`` < 1 or a grouping column is missing.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    groups = TE_GROUPS if groups is None else groups
    missing = sorted({c for keys in groups.values() for c in keys} - set(df.columns))
    if missing:
        raise ValueError(f"target encoding needs columns {missing}")

    out = df.copy()
    out["_row"] = range(len(out))
    out["_asof"] = out[DS] - pd.Timedelta(days=horizon)
    for name, keys in groups.items():
        table = _expanding_group_mean(out, keys).rename(columns={DS: "_asof"})
        left = out[[*keys, "_asof", "_row"]].sort_values("_asof")
        merged = pd.merge_asof(left, table.sort_values("_asof"), on="_asof", by=list(keys))
        out[name] = merged.set_index("_row")["te"].reindex(out["_row"]).to_numpy()
    return out.drop(columns=["_row", "_asof"])
