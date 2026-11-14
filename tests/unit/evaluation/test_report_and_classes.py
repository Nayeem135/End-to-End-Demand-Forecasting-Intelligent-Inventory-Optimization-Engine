import pandas as pd
import pytest

from demand_engine.data.splits import make_split_plan
from demand_engine.evaluation.backtest import backtest, leaderboard, leaderboard_by_class
from demand_engine.evaluation.report import leaderboard_report, markdown_table
from demand_engine.models.statistical import make_moving_average, make_naive

H = 14


@pytest.fixture
def results(small_long: pd.DataFrame) -> list:
    plan = make_split_plan(small_long["ds"].max(), H, n_folds=2, n_test=1)
    return [
        backtest(small_long, "naive", make_naive, plan.validation_cutoffs, H, 80),
        backtest(small_long, "ma", lambda: make_moving_average(14), plan.validation_cutoffs, H, 80),
    ]


def test_fold_summaries_one_per_cutoff(results: list) -> None:
    assert len(results[0].fold_summaries) == 2
    assert {"wrmsse", "wape", "smape"} <= set(results[0].fold_summaries[0])


def test_leaderboard_has_new_metrics(results: list) -> None:
    board = leaderboard(results)
    assert {"wape", "smape", "rmsse", "bias"} <= set(board.columns)


def test_leaderboard_by_class_ranks_within_each_class(
    results: list, small_long: pd.DataFrame
) -> None:
    ids = sorted(small_long["unique_id"].unique())
    classes = pd.Series({uid: ("smooth" if i % 2 else "lumpy") for i, uid in enumerate(ids)})
    board = leaderboard_by_class(results, classes)
    assert set(board["demand_class"]) == {"smooth", "lumpy"}
    for _, part in board.groupby("demand_class"):
        assert sorted(part["rank"]) == [1, 2]
        assert part.sort_values("rank")["wrmsse"].is_monotonic_increasing
        assert part["n_series"].iloc[0] == len(ids) // 2


def test_markdown_table_formats_percent_and_nan() -> None:
    df = pd.DataFrame({"model": ["a"], "wape": [0.1234], "rmsse": [float("nan")]})
    text = markdown_table(df, ["model", "wape", "rmsse"])
    assert text.splitlines()[0] == "| model | wape | rmsse |"
    assert text.splitlines()[1] == "|---|---:|---:|"
    assert "12.3%" in text
    assert "n/a" in text


def test_leaderboard_report_sections(results: list, small_long: pd.DataFrame) -> None:
    board = leaderboard(results)
    classes = pd.Series(dict.fromkeys(small_long["unique_id"].unique(), "intermittent"))
    by_class = leaderboard_by_class(results, classes).assign(split="test")
    report = leaderboard_report(board, board, by_class, champion="naive")
    assert "Champion (lowest validation WRMSSE): **naive**" in report
    assert "## Validation folds" in report
    assert "## Test windows by demand class" in report
    assert "### intermittent (6 series)" in report
