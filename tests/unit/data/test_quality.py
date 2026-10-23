import json

import pandas as pd
import pytest

from demand_engine.data.quality import summarize_quality


def _table() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=4)
    base = {"store_id": "CA_1", "event_type_1": ["none", "Cultural", "none", "none"]}
    a = pd.DataFrame({"unique_id": "A_CA_1", "item_id": "A", "ds": ds, "y": [0.0, 2.0, 0.0, 2.0],
                      "sell_price": 1.0, "snap": [1, 1, 0, 0], **base})  # fmt: skip
    b = pd.DataFrame({"unique_id": "B_CA_1", "item_id": "B", "ds": ds, "y": [4.0, 4.0, 4.0, 4.0],
                      "sell_price": 3.0, "snap": [1, 1, 0, 0], **base})  # fmt: skip
    return pd.concat([a, b], ignore_index=True)


def test_hand_computed_summary() -> None:
    r = summarize_quality(_table(), series_selected=3)
    assert (r.n_rows, r.n_series, r.n_items, r.n_stores, r.n_days) == (8, 2, 2, 1, 4)
    assert (r.start, r.end) == ("2020-01-01", "2020-01-04")
    assert r.series_dropped_short_history == 1
    assert r.zero_sales_share == pytest.approx(2 / 8)
    assert r.median_daily_units == pytest.approx((1.0 + 4.0) / 2)
    assert r.total_units == 20.0
    assert (r.price_min, r.price_max) == (1.0, 3.0)
    assert r.event_day_share == pytest.approx(0.25)
    assert r.snap_day_share == pytest.approx(0.5)
    assert r.demand_classes == {"intermittent": 1, "smooth": 1}


def test_report_serialises_and_reads_well() -> None:
    r = summarize_quality(_table(), series_selected=2)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["n_series"] == 2
    assert d["zero_sales_share"] == 0.25
    lines = r.lines()
    assert len(lines) == 5
    assert lines[0].startswith("2 series (2 items x 1 stores)")


def test_summary_on_synthetic_fixture(long_df: pd.DataFrame) -> None:
    r = summarize_quality(long_df, series_selected=long_df["unique_id"].nunique())
    assert r.n_series == long_df["unique_id"].nunique()
    assert sum(r.demand_classes.values()) == r.n_series
    assert 0.0 <= r.zero_sales_share < 1.0


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_quality(_table().iloc[0:0], series_selected=0)
    with pytest.raises(ValueError, match="series_selected"):
        summarize_quality(_table(), series_selected=1)
