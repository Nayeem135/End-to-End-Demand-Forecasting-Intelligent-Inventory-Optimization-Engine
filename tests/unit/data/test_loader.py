from pathlib import Path

import pandas as pd
import pytest

from demand_engine.data.loader import M5FilesMissingError, check_raw_files, load_m5, read_calendar


def test_missing_files_raise_with_instructions(tmp_path: Path) -> None:
    with pytest.raises(M5FilesMissingError, match="kaggle"):
        check_raw_files(tmp_path)


def test_load_m5_filters_state(raw_dir: Path) -> None:
    raw = load_m5(raw_dir, state="CA")
    assert set(raw.sales["state_id"]) == {"CA"}
    assert set(raw.prices["store_id"]) <= {"CA_1", "CA_2", "CA_3", "CA_4"}
    assert str(raw.calendar["date"].dtype).startswith("datetime64")


def test_load_m5_store_subset(raw_dir: Path) -> None:
    raw = load_m5(raw_dir, state="CA", stores=["CA_2"])
    assert set(raw.sales["store_id"]) == {"CA_2"}
    assert set(raw.prices["store_id"]) == {"CA_2"}


def test_load_m5_unknown_store(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="stores not found"):
        load_m5(raw_dir, state="CA", stores=["TX_1"])


def test_load_m5_unknown_state(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="no sales rows"):
        load_m5(raw_dir, state="WI")


def test_calendar_without_d_column_is_derived(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    dates = ["2011-01-30", "2011-01-29", "2011-01-31"]
    pd.DataFrame({"date": dates, "wm_yr_wk": 1}).to_csv(path, index=False)
    cal = read_calendar(path)
    assert cal["d"].tolist() == ["d_1", "d_2", "d_3"]
    assert cal["date"].iloc[0] == pd.Timestamp("2011-01-29")


def test_calendar_without_d_column_rejects_gaps(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    pd.DataFrame({"date": ["2011-01-29", "2011-02-02"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="contiguous"):
        read_calendar(path)
