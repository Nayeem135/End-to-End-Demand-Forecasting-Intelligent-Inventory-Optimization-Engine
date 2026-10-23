from collections.abc import Callable

import pandas as pd
import pytest

from demand_engine.data.validation import DEMAND_SCHEMA, DataValidationError, validate_long


def _set(col: str, idx: int, value: object) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def mutate(df: pd.DataFrame) -> pd.DataFrame:
        df.loc[idx, col] = value
        return df

    return mutate


@pytest.mark.parametrize(
    ("mutate", "msg"),
    [
        (lambda df: df.drop(columns="sell_price"), "missing columns"),
        (lambda df: df.iloc[0:0], "empty"),
        (_set("y", 0, None), "nulls"),
        (_set("y", 0, -1.0), "negative"),
        (_set("sell_price", 0, 0.0), "non-positive"),
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate"),
        (lambda df: df.drop(index=5), "missing dates"),
        (lambda df: df.assign(ds=df["ds"].astype(str)), "datetime"),
    ],
)
def test_validation_failures(
    small_long: pd.DataFrame, mutate: Callable[[pd.DataFrame], pd.DataFrame], msg: str
) -> None:
    with pytest.raises(DataValidationError, match=msg):
        validate_long(mutate(small_long.copy()))


def test_valid_table_passes(small_long: pd.DataFrame) -> None:
    validate_long(small_long)


def test_valid_table_with_duplicate_index_labels_passes(small_long: pd.DataFrame) -> None:
    half = len(small_long) // 2
    shuffled_index = pd.concat([small_long.iloc[:half], small_long.iloc[half:]])
    shuffled_index.index = [0] * len(shuffled_index)  # labels carry no meaning
    validate_long(shuffled_index)


def test_schema_is_exposed_for_documentation() -> None:
    assert {"unique_id", "ds", "y", "sell_price"} <= set(DEMAND_SCHEMA.columns)
