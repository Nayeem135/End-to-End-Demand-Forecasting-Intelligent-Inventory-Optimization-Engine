"""Data-quality contract for the long demand table, declared as a pandera schema.

Column-level rules (presence, nullability, value ranges, datetime type) and the
``(unique_id, ds)`` uniqueness key are expressed declaratively in
:data:`DEMAND_SCHEMA`; the two table-level rules (non-empty, contiguous daily dates
per series) are dataframe checks on the same schema. Failures are re-raised as
:class:`DataValidationError` with a short, human-readable message.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrorReason

from demand_engine.data.schema import DS, LONG_COLUMNS, PRICE, UNIQUE_ID, Y


class DataValidationError(ValueError):
    """Raised when the demand table violates its contract."""


def _is_datetime(series: pd.Series) -> bool:
    return bool(pd.api.types.is_datetime64_any_dtype(series))


def _non_empty(df: pd.DataFrame) -> bool:
    return len(df) > 0


def _contiguous_dates(df: pd.DataFrame) -> bool:
    span = df.groupby(UNIQUE_ID)[DS].agg(["min", "max", "size"])
    expected = (span["max"] - span["min"]).dt.days + 1
    return bool((expected == span["size"]).all())


_KEY_COLUMNS = (UNIQUE_ID, DS, Y, PRICE)

DEMAND_SCHEMA = pa.DataFrameSchema(
    {
        UNIQUE_ID: pa.Column(nullable=False),
        DS: pa.Column(
            nullable=False,
            checks=pa.Check(_is_datetime, element_wise=False, error="column 'ds' must be datetime"),
        ),
        Y: pa.Column(float, pa.Check.ge(0, error="negative demand values found"), nullable=False),
        PRICE: pa.Column(float, pa.Check.gt(0, error="non-positive prices found"), nullable=False),
        **{c: pa.Column(nullable=True) for c in LONG_COLUMNS if c not in _KEY_COLUMNS},
    },
    checks=[
        pa.Check(_non_empty, error="demand table is empty"),
        pa.Check(_contiguous_dates, error="some series have missing dates"),
    ],
    unique=[UNIQUE_ID, DS],
    strict=False,
    coerce=False,
    name="demand_table",
)


def _message(exc: SchemaError) -> str:
    reason = exc.reason_code
    if reason == SchemaErrorReason.SERIES_CONTAINS_NULLS:
        column = exc.schema.name if exc.schema is not None else "?"
        return f"column '{column}' contains nulls"
    if reason == SchemaErrorReason.DUPLICATES:
        return "duplicate (unique_id, ds) rows found"
    if exc.check is not None and getattr(exc.check, "error", None):
        return str(exc.check.error)
    return str(exc)


def validate_long(df: pd.DataFrame) -> None:
    """Validate the long demand table against :data:`DEMAND_SCHEMA`.

    Checks: required columns, non-empty, no nulls in key columns, datetime ``ds``,
    non-negative demand, positive price, unique ``(unique_id, ds)`` keys, and
    contiguous daily dates within each series.

    Raises:
        DataValidationError: Describing the first violated rule.
    """
    missing = [c for c in LONG_COLUMNS if c not in df.columns]  # report all, not just the first
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    try:
        # A fresh RangeIndex: pandera's failure-case report breaks on duplicate index labels
        # (pandas 3), and the row index carries no meaning in this table.
        DEMAND_SCHEMA.validate(df.reset_index(drop=True), lazy=False)
    except SchemaError as exc:
        raise DataValidationError(_message(exc)) from exc
