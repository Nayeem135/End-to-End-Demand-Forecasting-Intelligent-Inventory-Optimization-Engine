"""Column names and constants shared across the code base.

The long ("tidy") demand table is the contract between the data layer and every
downstream component. It follows the Nixtla convention ``unique_id, ds, y``.
"""

from __future__ import annotations

from typing import Final

UNIQUE_ID: Final = "unique_id"
DS: Final = "ds"
Y: Final = "y"
PRICE: Final = "sell_price"

ID_COLUMNS: Final = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
CALENDAR_COLUMNS: Final = (
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "snap",
)
LONG_COLUMNS: Final = (UNIQUE_ID, *ID_COLUMNS, DS, Y, PRICE, *CALENDAR_COLUMNS)

FORECAST_COLUMNS: Final = (UNIQUE_ID, DS, "yhat", "yhat_lo", "yhat_hi")

SALES_FILE: Final = "sales_train_evaluation.csv"
CALENDAR_FILE: Final = "calendar.csv"
PRICES_FILE: Final = "sell_prices.csv"
M5_FILES: Final = (SALES_FILE, CALENDAR_FILE, PRICES_FILE)

STATE_STORES: Final[dict[str, tuple[str, ...]]] = {
    "CA": ("CA_1", "CA_2", "CA_3", "CA_4"),
    "TX": ("TX_1", "TX_2", "TX_3"),
    "WI": ("WI_1", "WI_2", "WI_3"),
}
CATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "FOODS": ("FOODS_1", "FOODS_2", "FOODS_3"),
    "HOUSEHOLD": ("HOUSEHOLD_1", "HOUSEHOLD_2"),
    "HOBBIES": ("HOBBIES_1", "HOBBIES_2"),
}


def make_unique_id(item_id: str, store_id: str) -> str:
    """Build the series key used everywhere, e.g. ``FOODS_3_090_CA_1``."""
    return f"{item_id}_{store_id}"
