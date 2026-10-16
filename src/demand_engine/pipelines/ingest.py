"""Ingestion stage: raw M5 CSVs -> validated long Parquet table for the configured subset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from demand_engine.config import Settings
from demand_engine.data.loader import load_m5
from demand_engine.data.schema import DS, UNIQUE_ID
from demand_engine.data.splits import make_split_plan
from demand_engine.data.subset import select_items
from demand_engine.data.transform import filter_short_series, to_long
from demand_engine.data.validation import validate_long

logger = logging.getLogger(__name__)

PROCESSED_FILE = "sales.parquet"


def build_demand_table(settings: Settings) -> pd.DataFrame:
    """Load, subset, reshape, filter and validate the demand table (pure w.r.t. disk writes)."""
    cfg = settings.data
    raw = load_m5(settings.paths.raw_dir, cfg.state, cfg.stores)
    items = select_items(raw.sales, cfg.n_items, settings.seed)
    sales = raw.sales[raw.sales["item_id"].isin(items)]
    long = to_long(sales, raw.calendar, raw.prices, cfg.state, cfg.history_days)
    bt = settings.backtest
    plan = make_split_plan(long[DS].max(), bt.horizon, bt.n_folds, bt.n_test_windows)
    long = filter_short_series(long, plan.first_cutoff, bt.min_train_days)
    if long.empty:
        raise ValueError(
            "no series left after filtering; reduce min_train_days or increase history"
        )
    validate_long(long)
    return long


def run_ingest(settings: Settings) -> Path:
    """Execute the ingestion stage and write ``data/processed/sales.parquet``.

    Returns:
        Path of the written Parquet file.
    """
    long = build_demand_table(settings)
    out_dir = settings.paths.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PROCESSED_FILE
    long.to_parquet(path, index=False)
    logger.info(
        "Ingested %d series (%d rows, %s..%s) -> %s",
        long[UNIQUE_ID].nunique(),
        len(long),
        long[DS].min().date(),
        long[DS].max().date(),
        path,
    )
    return path


def load_demand_table(settings: Settings) -> pd.DataFrame:
    """Read the processed demand table written by :func:`run_ingest`."""
    path = settings.paths.processed_dir / PROCESSED_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run 'demand-engine ingest' first")
    df = pd.read_parquet(path)
    validate_long(df)
    return df
