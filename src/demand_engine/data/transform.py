"""Convert raw wide M5 tables into the long demand table used downstream."""

from __future__ import annotations

import logging

import pandas as pd

from demand_engine.data.schema import DS, ID_COLUMNS, LONG_COLUMNS, PRICE, UNIQUE_ID, Y
from demand_engine.types import FloatArray

logger = logging.getLogger(__name__)


def day_columns(sales: pd.DataFrame) -> list[str]:
    """Return the ``d_*`` columns of a wide sales table in chronological order."""
    cols = [c for c in sales.columns if c.startswith("d_")]
    return sorted(cols, key=lambda c: int(c[2:]))


def to_long(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    state: str,
    history_days: int | None = None,
) -> pd.DataFrame:
    """Reshape wide sales to one row per series-day and attach calendar + price data.

    Leading zeros before an item's first sale are removed (pre-launch periods are
    not demand). Missing prices inside the active period are forward/back filled
    per series.

    Args:
        sales: Wide sales table (already restricted to the chosen items/stores).
        calendar: M5 calendar table.
        prices: M5 weekly prices table.
        state: State code, used to pick the matching SNAP column.
        history_days: If given, keep only the last N days.

    Returns:
        Long table with columns :data:`~demand_engine.data.schema.LONG_COLUMNS`,
        sorted by ``unique_id, ds``.
    """
    days = day_columns(sales)
    if history_days is not None:
        days = days[-history_days:]
    wide = sales[[*ID_COLUMNS, *days]]
    long = wide.melt(id_vars=list(ID_COLUMNS), value_vars=days, var_name="d", value_name=Y)

    cal = calendar.rename(columns={f"snap_{state}": "snap"})
    cal = cal[
        ["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1", "event_type_1", "snap"]
    ]
    long = long.merge(cal, on="d", how="left", validate="many_to_one")
    if long["date"].isna().any():
        raise ValueError("calendar does not cover all sales day columns")
    long = long.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )
    long = long.rename(columns={"date": DS}).drop(columns="d")
    long[DS] = pd.to_datetime(long[DS])
    long[Y] = long[Y].astype("float64")
    long[UNIQUE_ID] = long["item_id"].astype(str) + "_" + long["store_id"].astype(str)
    long = long.sort_values([UNIQUE_ID, DS], ignore_index=True)

    long = trim_leading_zeros(long)
    long[PRICE] = long.groupby(UNIQUE_ID)[PRICE].transform(lambda s: s.ffill().bfill())
    long = long[long[PRICE].notna()]
    for col in ("event_name_1", "event_type_1"):
        long[col] = long[col].fillna("none").astype(str)
    for col in (UNIQUE_ID, *ID_COLUMNS):
        long[col] = long[col].astype(str)
    long[PRICE] = long[PRICE].astype("float64")
    result = long[list(LONG_COLUMNS)].reset_index(drop=True)
    logger.info("Long table: %d rows, %d series", len(result), result[UNIQUE_ID].nunique())
    return result


def trim_leading_zeros(long: pd.DataFrame) -> pd.DataFrame:
    """Drop rows before each series' first non-zero sale (expects sorting by id, date)."""
    started = long.groupby(UNIQUE_ID, sort=False)[Y].cummax() > 0
    return long[started.to_numpy()].reset_index(drop=True)


def filter_short_series(
    long: pd.DataFrame, first_cutoff: pd.Timestamp, min_days: int
) -> pd.DataFrame:
    """Remove series with fewer than ``min_days`` observations up to ``first_cutoff``.

    Such series cannot be modelled or evaluated reliably in the earliest fold.
    """
    history = long[long[DS] <= first_cutoff].groupby(UNIQUE_ID).size()
    keep = history[history >= min_days].index
    dropped = long[UNIQUE_ID].nunique() - len(keep)
    if dropped:
        logger.info("Dropping %d series with < %d days of history", dropped, min_days)
    return long[long[UNIQUE_ID].isin(keep)].reset_index(drop=True)


ADI_CUT = 1.32
CV2_CUT = 0.49


def demand_profile(y: FloatArray) -> tuple[float, float]:
    """Average inter-demand interval (ADI) and squared CV of non-zero demand sizes.

    ``ADI = periods / periods with demand``; ``CV² = (std / mean)²`` of the non-zero
    values (0 when fewer than two non-zero periods). A series with no demand returns
    ``(inf, 0.0)``.
    """
    nonzero = y[y > 0]
    if nonzero.size == 0:
        return float("inf"), 0.0
    adi = y.size / nonzero.size
    cv2 = float((nonzero.std() / nonzero.mean()) ** 2) if nonzero.size > 1 else 0.0
    return float(adi), cv2


def classify_profile(
    adi: float, cv2: float, adi_cut: float = ADI_CUT, cv2_cut: float = CV2_CUT
) -> str:
    """Syntetos-Boylan class for given ADI and CV² (see :func:`classify_intermittency`)."""
    if adi < adi_cut:
        return "smooth" if cv2 < cv2_cut else "erratic"
    return "intermittent" if cv2 < cv2_cut else "lumpy"


def classify_intermittency(
    y: FloatArray, adi_cut: float = ADI_CUT, cv2_cut: float = CV2_CUT
) -> str:
    """Classify a demand series using the Syntetos-Boylan ADI/CV² scheme.

    Args:
        y: Non-negative demand history.
        adi_cut: Average inter-demand interval threshold.
        cv2_cut: Squared coefficient of variation threshold (non-zero demand sizes).

    Returns:
        One of ``"smooth"``, ``"erratic"``, ``"intermittent"``, ``"lumpy"``; a series
        without any demand is ``"lumpy"``.
    """
    if not (y > 0).any():
        return "lumpy"
    return classify_profile(*demand_profile(y), adi_cut=adi_cut, cv2_cut=cv2_cut)
