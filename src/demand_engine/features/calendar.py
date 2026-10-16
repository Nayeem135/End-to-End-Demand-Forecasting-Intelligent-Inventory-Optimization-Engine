"""Calendar features: known in advance for any future date, hence leakage-free."""

from __future__ import annotations

import pandas as pd

from demand_engine.data.schema import DS

CALENDAR_FEATURES = (
    "wday",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "is_event",
    "snap",
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with calendar features added.

    Requires the ``ds``, ``wday``, ``month``, ``event_type_1`` and ``snap`` columns
    of the long demand table.
    """
    out = df.copy()
    out["day_of_month"] = out[DS].dt.day.astype("int16")
    out["week_of_year"] = out[DS].dt.isocalendar().week.astype("int16").to_numpy()
    out["is_weekend"] = out["wday"].isin([1, 2]).astype("int8")  # M5: 1=Saturday, 2=Sunday
    out["is_event"] = (out["event_type_1"] != "none").astype("int8")
    out["event_type"] = out["event_type_1"]  # encoded as categorical by the builder
    return out
