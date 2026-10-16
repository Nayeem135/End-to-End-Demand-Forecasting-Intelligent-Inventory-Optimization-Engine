"""Fast DataFrame -> table loading (PostgreSQL ``COPY``, generic INSERT elsewhere)."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from sqlalchemy import Connection, Table


def to_records(df: pd.DataFrame, table: Table) -> list[dict[str, Any]]:
    """Select the table's columns as plain Python values (NaN -> None, datetimes -> dates)."""
    cols = [c.name for c in table.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table.name}: data frame lacks columns {missing}")
    frame = df[cols].copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.date
    frame = frame.astype(object).where(frame.notna(), None)
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def bulk_insert(connection: Connection, table: Table, df: pd.DataFrame) -> int:
    """Insert all rows of ``df`` into ``table`` and return the row count."""
    if df.empty:
        return 0
    records = to_records(df, table)
    cols = [c.name for c in table.columns]
    if connection.dialect.name == "postgresql":
        raw = connection.connection.driver_connection
        if raw is None:
            raise RuntimeError("database connection is closed")
        with (
            raw.cursor() as cur,
            cur.copy(f"COPY {table.name} ({', '.join(cols)}) FROM STDIN") as copy,
        ):
            for rec in records:
                copy.write_row([rec[c] for c in cols])
    else:
        connection.execute(table.insert(), records)
    return len(records)
