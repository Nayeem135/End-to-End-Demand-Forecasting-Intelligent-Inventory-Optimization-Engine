import pandas as pd
import pytest

from demand_engine.features.calendar import CALENDAR_FEATURES, add_calendar_features
from demand_engine.features.price import PRICE_FEATURES, add_price_features


def test_calendar_features(small_long: pd.DataFrame) -> None:
    out = add_calendar_features(small_long)
    assert set(CALENDAR_FEATURES) <= set(out.columns)
    sat = out[out["wday"] == 1]
    assert (sat["is_weekend"] == 1).all()
    assert (out.loc[out["event_type_1"] == "none", "is_event"] == 0).all()
    assert out["day_of_month"].between(1, 31).all()
    assert "day_of_month" not in small_long.columns  # input not mutated


def test_price_features() -> None:
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 8,
            "store_id": "S",
            "dept_id": "D",
            "ds": pd.date_range("2020-01-01", periods=8),
            "sell_price": [2.0] * 7 + [1.0],
        }
    )
    out = add_price_features(df)
    assert set(PRICE_FEATURES) <= set(out.columns)
    assert out["price_change_7"].iloc[-1] == pytest.approx(-0.5)
    assert out["price_change_7"].iloc[0] == 0.0
    assert out["price_norm"].mean() == pytest.approx(1.0)
    assert (out["price_rel_store_dept"] == 1.0).all()
