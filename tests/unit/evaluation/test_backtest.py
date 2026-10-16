import pandas as pd
import pytest

from demand_engine.data.splits import make_split_plan
from demand_engine.evaluation.backtest import backtest, leaderboard, rolling_forecasts
from demand_engine.models.statistical import make_moving_average, make_naive

H = 14


def test_rolling_forecasts_shape_and_no_overlap(small_long: pd.DataFrame) -> None:
    plan = make_split_plan(small_long["ds"].max(), H, n_folds=2, n_test=1)
    preds = rolling_forecasts(small_long, make_naive, plan.validation_cutoffs, H)
    n_series = small_long["unique_id"].nunique()
    assert len(preds) == n_series * H * 2
    for cutoff, g in preds.groupby("cutoff"):
        assert (g["ds"] > cutoff).all()
        assert (g["ds"] <= cutoff + pd.Timedelta(days=H)).all()
    assert preds["y"].notna().all()


def test_rolling_forecasts_rejects_empty_window(small_long: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no data"):
        rolling_forecasts(small_long, make_naive, [small_long["ds"].max()], H)


def test_backtest_and_leaderboard(small_long: pd.DataFrame) -> None:
    plan = make_split_plan(small_long["ds"].max(), H, n_folds=2, n_test=1)
    results = [
        backtest(small_long, "naive", make_naive, plan.validation_cutoffs, H, 80),
        backtest(small_long, "ma", lambda: make_moving_average(14), plan.validation_cutoffs, H, 80),
    ]
    board = leaderboard(results)
    assert board["rank"].tolist() == [1, 2]
    assert board["wrmsse"].is_monotonic_increasing
    assert (results[0].predictions["model"] == "naive").all()
    assert len(results[0].per_series) == small_long["unique_id"].nunique()
