"""Generator for synthetic data that exactly mimics the raw M5 file formats.

Used by the unit/integration tests (so CI never needs the 450 MB Kaggle download)
and by ``demand-engine synthetic`` for a fully offline demo. The generated demand
contains the phenomena that make M5 hard: weekly seasonality, SNAP and event
effects, price promotions with elasticity, intermittent (mostly-zero) items and
late product launches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand_engine.data.schema import (
    CALENDAR_FILE,
    CATEGORIES,
    PRICES_FILE,
    SALES_FILE,
    STATE_STORES,
)

logger = logging.getLogger(__name__)

M5_START = pd.Timestamp("2011-01-29")
_WEEKDAYS = ("Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_EVENTS: dict[tuple[int, int], tuple[str, str]] = {
    (1, 1): ("NewYear", "National"),
    (2, 14): ("ValentinesDay", "Cultural"),
    (7, 4): ("IndependenceDay", "National"),
    (10, 31): ("Halloween", "Cultural"),
    (11, 24): ("Thanksgiving", "National"),
    (12, 25): ("Christmas", "National"),
}
_SNAP_DAYS: dict[str, frozenset[int]] = {
    "CA": frozenset(range(1, 11)),
    "TX": frozenset({1, 3, 5, 6, 7, 9, 11, 12, 13, 15}),
    "WI": frozenset({2, 3, 5, 6, 8, 9, 11, 12, 14, 15}),
}
_WEEKLY_PROFILE = np.array([1.30, 1.35, 0.95, 0.85, 0.85, 0.90, 1.05])  # Sat..Fri


@dataclass(frozen=True)
class SyntheticM5:
    """The three raw M5 tables."""

    sales: pd.DataFrame
    calendar: pd.DataFrame
    prices: pd.DataFrame


def make_calendar(n_days: int) -> pd.DataFrame:
    """Build an M5-format ``calendar.csv`` table covering ``n_days`` days from 2011-01-29."""
    dates = pd.date_range(M5_START, periods=n_days, freq="D")
    day_index = np.arange(n_days)
    weekday_pos = day_index % 7  # 2011-01-29 is a Saturday, as in M5
    events = [_EVENTS.get((d.month, d.day), (None, None)) for d in dates]
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + day_index // 7,
            "weekday": [_WEEKDAYS[i] for i in weekday_pos],
            "wday": weekday_pos + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in day_index],
            "event_name_1": [e[0] for e in events],
            "event_type_1": [e[1] for e in events],
            "event_name_2": None,
            "event_type_2": None,
        }
    )
    for state, days in _SNAP_DAYS.items():
        calendar[f"snap_{state}"] = dates.day.isin(list(days)).astype(int)
    return calendar


def _item_catalogue(n_items: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create ``n_items`` items spread across the M5 departments."""
    depts = [(cat, dept) for cat, dept_list in CATEGORIES.items() for dept in dept_list]
    weights = np.array([0.15, 0.2, 0.3, 0.12, 0.1, 0.08, 0.05])
    choice = rng.choice(len(depts), size=n_items, p=weights / weights.sum())
    counters: dict[str, int] = {}
    rows = []
    for idx in choice:
        cat, dept = depts[idx]
        counters[dept] = counters.get(dept, 0) + 1
        rows.append({"item_id": f"{dept}_{counters[dept]:03d}", "dept_id": dept, "cat_id": cat})
    return pd.DataFrame(rows)


def generate_synthetic_m5(
    n_items: int = 30,
    n_days: int = 500,
    states: tuple[str, ...] = ("CA",),
    seed: int = 42,
) -> SyntheticM5:
    """Generate synthetic raw M5 tables.

    Args:
        n_items: Number of distinct items (each is sold in every store of each state).
        n_days: Number of daily observations (``d_1`` .. ``d_n``).
        states: State codes whose stores are generated.
        seed: Random seed; identical inputs always yield identical output.

    Returns:
        A :class:`SyntheticM5` with ``sales`` (wide), ``calendar`` and ``prices`` tables.
    """
    if n_items < 1 or n_days < 28:
        raise ValueError("n_items must be >= 1 and n_days >= 28")
    unknown = set(states) - set(STATE_STORES)
    if unknown:
        raise ValueError(f"unknown states: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    calendar = make_calendar(n_days)
    items = _item_catalogue(n_items, rng)
    stores = [(state, store) for state in states for store in STATE_STORES[state]]
    n_weeks = int(calendar["wm_yr_wk"].nunique())
    week_of_day = (calendar["wm_yr_wk"] - calendar["wm_yr_wk"].min()).to_numpy()

    base_rate = rng.lognormal(mean=0.0, sigma=1.1, size=n_items)  # many slow movers
    base_price = np.round(rng.uniform(1.0, 20.0, size=n_items), 2)
    launch_day = np.where(rng.random(n_items) < 0.2, rng.integers(0, n_days // 3, n_items), 0)
    trend = rng.normal(0.0, 0.0004, size=n_items)
    weekly = _WEEKLY_PROFILE[np.arange(n_days) % 7]
    yearly = 1.0 + 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    event_effect = np.where(
        calendar["event_name_1"].eq("Christmas"),
        0.0,
        np.where(calendar["event_name_1"].notna(), 1.2, 1.0),
    )

    sales_rows: list[dict[str, object]] = []
    price_frames: list[pd.DataFrame] = []
    t = np.arange(n_days)
    for state, store in stores:
        store_mult = rng.uniform(0.6, 1.5)
        snap = calendar[f"snap_{state}"].to_numpy()
        for i, item in enumerate(items.to_dict("records")):
            promo_weeks = rng.random(n_weeks) < 0.1
            weekly_price = base_price[i] * np.where(promo_weeks, 0.8, 1.0)
            day_price = weekly_price[week_of_day]
            snap_effect = 1.0 + 0.15 * snap if item["cat_id"] == "FOODS" else 1.0
            elasticity = (day_price / base_price[i]) ** -2.0
            lam = (
                base_rate[i]
                * store_mult
                * weekly
                * yearly
                * event_effect
                * snap_effect
                * elasticity
                * np.exp(trend[i] * t)
            )
            gamma = rng.gamma(shape=2.0, scale=0.5, size=n_days)  # NB over-dispersion
            demand = rng.poisson(lam * gamma).astype(int)
            demand[t < launch_day[i]] = 0
            row: dict[str, object] = {
                "id": f"{item['item_id']}_{store}_evaluation",
                "item_id": item["item_id"],
                "dept_id": item["dept_id"],
                "cat_id": item["cat_id"],
                "store_id": store,
                "state_id": state,
            }
            row.update({f"d_{k + 1}": int(v) for k, v in enumerate(demand)})
            sales_rows.append(row)
            first_week = int(week_of_day[launch_day[i]])
            weeks = np.arange(first_week, n_weeks)
            price_frames.append(
                pd.DataFrame(
                    {
                        "store_id": store,
                        "item_id": item["item_id"],
                        "wm_yr_wk": 11101 + weeks,
                        "sell_price": np.round(weekly_price[weeks], 2),
                    }
                )
            )

    sales = pd.DataFrame(sales_rows)
    prices = pd.concat(price_frames, ignore_index=True)
    logger.info("Generated synthetic M5: %d series x %d days", len(sales), n_days)
    return SyntheticM5(sales=sales, calendar=calendar, prices=prices)


def write_synthetic_m5(
    raw_dir: Path,
    n_items: int = 30,
    n_days: int = 500,
    states: tuple[str, ...] = ("CA",),
    seed: int = 42,
) -> None:
    """Generate synthetic M5 tables and write them as the three M5 CSV files.

    Args:
        raw_dir: Destination directory (created if missing).
        n_items: See :func:`generate_synthetic_m5`.
        n_days: See :func:`generate_synthetic_m5`.
        states: See :func:`generate_synthetic_m5`.
        seed: See :func:`generate_synthetic_m5`.
    """
    data = generate_synthetic_m5(n_items=n_items, n_days=n_days, states=states, seed=seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    data.sales.to_csv(raw_dir / SALES_FILE, index=False)
    data.calendar.to_csv(raw_dir / CALENDAR_FILE, index=False)
    data.prices.to_csv(raw_dir / PRICES_FILE, index=False)
    logger.info("Wrote synthetic M5 CSVs to %s", raw_dir)
