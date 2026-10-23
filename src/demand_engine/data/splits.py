"""Time-based split logic: rolling-origin validation folds and held-out test windows.

Timeline (``h`` = horizon)::

    ... train ... | val fold 1 | ... | val fold k | test window 1 | ... | test window m |
                                                                                    last date

Each window is ``h`` days long and starts the day after its cut-off. Validation
folds are used for model selection and to estimate forecast-error distributions
(safety stock). Test windows are only touched for the final report and the
inventory simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from demand_engine.data.schema import DS


@dataclass(frozen=True)
class SplitPlan:
    """Cut-off dates of all validation folds and test windows (chronological)."""

    validation_cutoffs: tuple[pd.Timestamp, ...]
    test_cutoffs: tuple[pd.Timestamp, ...]
    horizon: int

    @property
    def first_cutoff(self) -> pd.Timestamp:
        """Earliest cut-off of the whole plan."""
        return self.validation_cutoffs[0]

    @property
    def test_start(self) -> pd.Timestamp:
        """First day of the held-out test period."""
        return self.test_cutoffs[0] + pd.Timedelta(days=1)


def make_split_plan(last_date: pd.Timestamp, horizon: int, n_folds: int, n_test: int) -> SplitPlan:
    """Compute consecutive, non-overlapping cut-offs ending at ``last_date``.

    Args:
        last_date: Last observed date of the data set.
        horizon: Window length in days.
        n_folds: Number of validation folds.
        n_test: Number of test windows.

    Returns:
        A :class:`SplitPlan`.
    """
    if horizon < 1 or n_folds < 1 or n_test < 1:
        raise ValueError("horizon, n_folds and n_test must all be >= 1")
    total = n_folds + n_test
    cutoffs = [last_date - pd.Timedelta(days=horizon * (total - i)) for i in range(total)]
    return SplitPlan(
        validation_cutoffs=tuple(cutoffs[:n_folds]),
        test_cutoffs=tuple(cutoffs[n_folds:]),
        horizon=horizon,
    )


def split_at(
    df: pd.DataFrame, cutoff: pd.Timestamp, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into history (``ds <= cutoff``) and the following ``horizon`` days.

    Returns:
        ``(train, window)`` data frames.
    """
    end = cutoff + pd.Timedelta(days=horizon)
    train = df[df[DS] <= cutoff]
    window = df[(df[DS] > cutoff) & (df[DS] <= end)]
    return train.reset_index(drop=True), window.reset_index(drop=True)
