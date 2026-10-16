"""Shared type aliases."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


def cell(df: pd.DataFrame, row: Hashable, column: str) -> float:
    """Read one numeric cell as ``float`` (raises ``KeyError`` if absent)."""
    value = df.at[row, column]
    if not isinstance(value, int | float | np.number):
        raise TypeError(f"cell ({row!r}, {column!r}) is not numeric: {value!r}")
    return float(value)
