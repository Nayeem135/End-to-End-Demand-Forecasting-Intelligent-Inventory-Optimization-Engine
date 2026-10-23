import numpy as np
import pandas as pd
import pytest

from demand_engine.data.schema import LONG_COLUMNS
from demand_engine.data.synthetic import SyntheticM5
from demand_engine.data.transform import (
    classify_intermittency,
    classify_profile,
    day_columns,
    demand_profile,
    filter_short_series,
    to_long,
    trim_leading_zeros,
)
from demand_engine.data.validation import validate_long


def test_day_columns_sorted_numerically() -> None:
    df = pd.DataFrame(columns=["id", "d_10", "d_2", "d_1"])
    assert day_columns(df) == ["d_1", "d_2", "d_10"]


def test_to_long_contract(long_df: pd.DataFrame) -> None:
    assert list(long_df.columns) == list(LONG_COLUMNS)
    validate_long(long_df)  # raises on any contract violation
    assert long_df["unique_id"].iloc[0].endswith(long_df["store_id"].iloc[0])


def test_to_long_values_round_trip(synthetic_raw: SyntheticM5, long_df: pd.DataFrame) -> None:
    row = synthetic_raw.sales.iloc[0]
    uid = f"{row['item_id']}_{row['store_id']}"
    series = long_df[long_df["unique_id"] == uid].set_index("ds")["y"]
    cal = synthetic_raw.calendar.set_index("d")
    last_d = day_columns(synthetic_raw.sales)[-1]
    assert series.loc[pd.Timestamp(cal.loc[last_d, "date"])] == row[last_d]


def test_history_days_limits_range(synthetic_raw: SyntheticM5) -> None:
    long = to_long(
        synthetic_raw.sales, synthetic_raw.calendar, synthetic_raw.prices, "CA", history_days=60
    )
    assert long["ds"].nunique() <= 60


def test_leading_zeros_trimmed(long_df: pd.DataFrame) -> None:
    first = long_df.groupby("unique_id")["y"].first()
    assert (first > 0).all()


def test_trim_leading_zeros_keeps_inner_zeros() -> None:
    df = pd.DataFrame({"unique_id": ["a"] * 5, "y": [0.0, 0.0, 2.0, 0.0, 1.0]})
    assert trim_leading_zeros(df)["y"].tolist() == [2.0, 0.0, 1.0]


def test_filter_short_series() -> None:
    dates = pd.date_range("2020-01-01", periods=10)
    df = pd.DataFrame(
        {"unique_id": ["a"] * 10 + ["b"] * 3, "ds": list(dates) + list(dates[-3:]), "y": 1.0}
    )
    out = filter_short_series(df, first_cutoff=dates[-1], min_days=5)
    assert out["unique_id"].unique().tolist() == ["a"]


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        ([5, 5, 6, 5, 5, 6], "smooth"),
        ([1, 30, 2, 40, 1, 50], "erratic"),
        ([0, 0, 3, 0, 0, 3], "intermittent"),
        ([0, 0, 1, 0, 0, 40], "lumpy"),
        ([0, 0, 0], "lumpy"),
    ],
)
def test_classify_intermittency(y: list[int], expected: str) -> None:
    assert classify_intermittency(np.array(y, dtype=float)) == expected


def test_demand_profile_hand_computed() -> None:
    adi, cv2 = demand_profile(np.array([0.0, 2.0, 0.0, 4.0]))
    assert adi == 2.0
    assert cv2 == pytest.approx((1.0 / 3.0) ** 2)  # std 1, mean 3
    assert demand_profile(np.array([0.0, 5.0])) == (2.0, 0.0)
    assert demand_profile(np.zeros(3)) == (float("inf"), 0.0)


@pytest.mark.parametrize(
    ("adi", "cv2", "expected"),
    [(1.0, 0.1, "smooth"), (1.0, 0.9, "erratic"), (2.0, 0.1, "intermittent"), (2.0, 0.9, "lumpy")],
)
def test_classify_profile_quadrants(adi: float, cv2: float, expected: str) -> None:
    assert classify_profile(adi, cv2) == expected
