from itertools import pairwise

import pandas as pd
import pytest

from demand_engine.data.splits import make_split_plan, split_at


def test_split_plan_layout() -> None:
    last = pd.Timestamp("2016-05-22")
    plan = make_split_plan(last, horizon=28, n_folds=3, n_test=2)
    cutoffs = [*plan.validation_cutoffs, *plan.test_cutoffs]
    assert len(cutoffs) == 5
    assert all((b - a).days == 28 for a, b in pairwise(cutoffs))
    assert plan.test_cutoffs[-1] + pd.Timedelta(days=28) == last
    assert plan.first_cutoff == cutoffs[0]
    assert plan.test_start == plan.test_cutoffs[0] + pd.Timedelta(days=1)


def test_split_plan_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        make_split_plan(pd.Timestamp("2020-01-01"), horizon=0, n_folds=1, n_test=1)


def test_split_at_is_disjoint_and_complete() -> None:
    df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=30), "y": range(30)})
    cutoff = pd.Timestamp("2020-01-20")
    train, window = split_at(df, cutoff, horizon=7)
    assert train["ds"].max() == cutoff
    assert window["ds"].min() == cutoff + pd.Timedelta(days=1)
    assert len(window) == 7
    assert set(train["ds"]).isdisjoint(window["ds"])
