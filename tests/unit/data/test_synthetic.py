from pathlib import Path

import pandas as pd
import pytest

from demand_engine.data.schema import CALENDAR_FILE, PRICES_FILE, SALES_FILE
from demand_engine.data.synthetic import generate_synthetic_m5, make_calendar, write_synthetic_m5


def test_calendar_matches_m5_format() -> None:
    cal = make_calendar(14)
    assert cal.loc[0, "date"] == "2011-01-29"
    assert cal.loc[0, "weekday"] == "Saturday"
    assert cal.loc[0, "wday"] == 1
    assert cal["d"].tolist()[:2] == ["d_1", "d_2"]
    assert cal["wm_yr_wk"].nunique() == 2
    assert {"snap_CA", "snap_TX", "snap_WI"} <= set(cal.columns)


def test_generation_is_deterministic() -> None:
    a = generate_synthetic_m5(n_items=3, n_days=60, seed=1)
    b = generate_synthetic_m5(n_items=3, n_days=60, seed=1)
    pd.testing.assert_frame_equal(a.sales, b.sales)
    pd.testing.assert_frame_equal(a.prices, b.prices)


def test_shapes_and_non_negative() -> None:
    data = generate_synthetic_m5(n_items=4, n_days=60, states=("CA", "TX"), seed=3)
    assert len(data.sales) == 4 * (4 + 3)
    d_cols = [c for c in data.sales.columns if c.startswith("d_")]
    assert len(d_cols) == 60
    assert (data.sales[d_cols] >= 0).all().all()
    assert (data.prices["sell_price"] > 0).all()


def test_christmas_has_zero_sales() -> None:
    data = generate_synthetic_m5(n_items=3, n_days=400, seed=2)
    xmas = data.calendar.loc[data.calendar["event_name_1"] == "Christmas", "d"].iloc[0]
    assert (data.sales[xmas] == 0).all()


@pytest.mark.parametrize(
    ("kwargs", "msg"), [({"n_items": 0}, "n_items"), ({"states": ("NY",)}, "unknown")]
)
def test_invalid_arguments(kwargs: dict[str, object], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        generate_synthetic_m5(**kwargs)


def test_write_creates_three_files(tmp_path: Path) -> None:
    target = tmp_path / "raw"
    write_synthetic_m5(target, n_items=2, n_days=30)
    for name in (SALES_FILE, CALENDAR_FILE, PRICES_FILE):
        assert (target / name).is_file()
