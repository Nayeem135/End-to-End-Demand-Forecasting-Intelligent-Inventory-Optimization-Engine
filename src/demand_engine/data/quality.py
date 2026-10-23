"""Short data-quality summary of the validated demand table.

Computed once at ingestion, logged, and written next to the Parquet file as
``quality_report.json`` so the dataset behind every result is documented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from demand_engine.data.schema import DS, PRICE, UNIQUE_ID, Y
from demand_engine.data.transform import classify_intermittency


@dataclass(frozen=True)
class QualityReport:
    """Key facts about the demand table (counts, coverage, sparsity)."""

    n_rows: int
    n_series: int
    n_items: int
    n_stores: int
    start: str
    end: str
    n_days: int
    series_selected: int
    series_dropped_short_history: int
    zero_sales_share: float
    median_daily_units: float
    total_units: float
    price_min: float
    price_max: float
    event_day_share: float
    snap_day_share: float
    demand_classes: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable representation (floats rounded to 4 decimals)."""
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}

    def lines(self) -> list[str]:
        """Human-readable summary, one fact per line."""
        classes = ", ".join(f"{k} {v}" for k, v in sorted(self.demand_classes.items()))
        return [
            f"{self.n_series:,} series ({self.n_items} items x {self.n_stores} stores), "
            f"{self.n_rows:,} rows, {self.start}..{self.end} ({self.n_days} days)",
            f"selected {self.series_selected:,} series, dropped "
            f"{self.series_dropped_short_history:,} with too little history",
            f"zero-sales days {self.zero_sales_share:.1%}, median {self.median_daily_units:.2f} "
            f"units/day per series, {self.total_units:,.0f} units in total",
            f"prices ${self.price_min:.2f}..${self.price_max:.2f}; event days "
            f"{self.event_day_share:.1%}, SNAP days {self.snap_day_share:.1%}",
            f"demand classes: {classes}",
        ]


def summarize_quality(df: pd.DataFrame, series_selected: int) -> QualityReport:
    """Summarise a validated long demand table.

    Args:
        df: Long demand table (already validated).
        series_selected: Series chosen by subsetting, before short-history filtering.

    Raises:
        ValueError: If the table is empty or ``series_selected`` is smaller than the
            number of series present.
    """
    if df.empty:
        raise ValueError("cannot summarise an empty demand table")
    n_series = int(df[UNIQUE_ID].nunique())
    if series_selected < n_series:
        raise ValueError("series_selected cannot be smaller than the series in the table")
    per_day = df.drop_duplicates(DS)
    per_series = df.groupby(UNIQUE_ID)[Y]
    classes = per_series.agg(lambda s: classify_intermittency(s.to_numpy(dtype="float64")))
    return QualityReport(
        n_rows=len(df),
        n_series=n_series,
        n_items=int(df["item_id"].nunique()),
        n_stores=int(df["store_id"].nunique()),
        start=df[DS].min().date().isoformat(),
        end=df[DS].max().date().isoformat(),
        n_days=int(df[DS].nunique()),
        series_selected=series_selected,
        series_dropped_short_history=series_selected - n_series,
        zero_sales_share=float((df[Y] == 0).mean()),
        median_daily_units=float(np.median(per_series.mean())),
        total_units=float(df[Y].sum()),
        price_min=float(df[PRICE].min()),
        price_max=float(df[PRICE].max()),
        event_day_share=float((per_day["event_type_1"] != "none").mean()),
        snap_day_share=float(per_day["snap"].astype(bool).mean()),
        demand_classes={str(k): int(v) for k, v in classes.value_counts().items()},
    )
