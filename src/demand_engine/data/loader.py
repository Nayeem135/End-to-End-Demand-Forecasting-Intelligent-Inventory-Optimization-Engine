"""Memory-conscious readers for the three raw M5 CSV files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from demand_engine.data.schema import CALENDAR_FILE, M5_FILES, PRICES_FILE, SALES_FILE

logger = logging.getLogger(__name__)

_CHUNK_ROWS = 2_000


class M5FilesMissingError(FileNotFoundError):
    """Raised when one or more M5 CSV files are not present in the raw directory."""


@dataclass(frozen=True)
class M5Raw:
    """Raw M5 tables restricted to one state."""

    sales: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame


def check_raw_files(raw_dir: Path) -> None:
    """Ensure all M5 files exist, otherwise raise with download instructions.

    Raises:
        M5FilesMissingError: If any required file is missing.
    """
    missing = [name for name in M5_FILES if not (raw_dir / name).is_file()]
    if missing:
        raise M5FilesMissingError(
            f"Missing M5 files in '{raw_dir}': {missing}. Download them from "
            "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data, or run "
            "'demand-engine synthetic' for an offline synthetic dataset."
        )


def read_sales(path: Path, state: str) -> pd.DataFrame:
    """Read the wide sales file in chunks, keeping only rows of ``state``.

    Args:
        path: Path to ``sales_train_evaluation.csv``.
        state: State code to keep, e.g. ``"CA"``.

    Returns:
        Wide sales table (id columns plus ``d_1 .. d_n`` as int32).
    """
    header = pd.read_csv(path, nrows=0).columns
    dtypes: dict[str, str] = {c: "int32" for c in header if c.startswith("d_")}
    chunks = [
        chunk[chunk["state_id"] == state]
        for chunk in pd.read_csv(path, dtype=dtypes, chunksize=_CHUNK_ROWS)
    ]
    sales = pd.concat(chunks, ignore_index=True)
    if sales.empty:
        raise ValueError(f"no sales rows found for state '{state}' in {path}")
    logger.info("Loaded %d sales series for state %s", len(sales), state)
    return sales


def read_calendar(path: Path) -> pd.DataFrame:
    """Read ``calendar.csv`` with a parsed ``date`` column.

    Some public M5 mirrors omit the ``d`` key column; it is then derived from the
    (validated contiguous) date order, ``d_1`` being the first date.
    """
    calendar = pd.read_csv(path, parse_dates=["date"])
    if "d" not in calendar.columns:
        calendar = calendar.sort_values("date", ignore_index=True)
        gaps = calendar["date"].diff().dropna().dt.days
        if not (gaps == 1).all():
            raise ValueError("calendar without a 'd' column must have contiguous daily dates")
        calendar["d"] = [f"d_{i + 1}" for i in range(len(calendar))]
    return calendar


def read_prices(path: Path, stores: list[str]) -> pd.DataFrame:
    """Read ``sell_prices.csv`` keeping only the given stores."""
    chunks = [
        chunk[chunk["store_id"].isin(stores)]
        for chunk in pd.read_csv(
            path,
            # float64: float32 would turn $9.58 into 9.5799999 once widened downstream.
            dtype={"wm_yr_wk": "int32", "sell_price": "float64"},
            chunksize=500_000,
        )
    ]
    return pd.concat(chunks, ignore_index=True)


def load_m5(raw_dir: Path, state: str, stores: list[str] | None = None) -> M5Raw:
    """Load all three M5 tables for one state (optionally a subset of its stores).

    Args:
        raw_dir: Directory containing the M5 CSV files.
        state: State code, e.g. ``"CA"``.
        stores: Optional explicit list of store ids within the state.

    Returns:
        :class:`M5Raw` with sales, calendar and prices.
    """
    check_raw_files(raw_dir)
    sales = read_sales(raw_dir / SALES_FILE, state)
    if stores:
        unknown = set(stores) - set(sales["store_id"].unique())
        if unknown:
            raise ValueError(f"stores not found in state {state}: {sorted(unknown)}")
        sales = sales[sales["store_id"].isin(stores)].reset_index(drop=True)
    store_list = sorted(sales["store_id"].unique().tolist())
    calendar = read_calendar(raw_dir / CALENDAR_FILE)
    prices = read_prices(raw_dir / PRICES_FILE, store_list)
    return M5Raw(sales=sales, calendar=calendar, prices=prices)
