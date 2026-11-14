"""End-to-end pipeline behaviour on synthetic data (no database needed beyond fixtures)."""

import mlflow
import pandas as pd
import pytest

from demand_engine.pipelines.artifacts import RunArtifacts, latest_run
from demand_engine.pipelines.publish import ARTIFACT_TABLES

pytestmark = pytest.mark.integration


def test_all_artifacts_written(pipeline_run: RunArtifacts) -> None:
    for artifact in ARTIFACT_TABLES:
        assert not pipeline_run.load_frame(artifact).empty, artifact
    assert latest_run(pipeline_run.root.parent.parent).run_id == pipeline_run.run_id


def test_manifest_contents(pipeline_run: RunArtifacts) -> None:
    manifest = pipeline_run.load_json("manifest")
    assert manifest["champion"] in {"naive", "moving_average", "lightgbm"}
    assert len(manifest["validation_cutoffs"]) == 2
    assert len(manifest["test_cutoffs"]) == 2
    assert "database" not in manifest["settings"]  # credentials never leave the process
    assert set(manifest["headline"]) >= {"savings_vs_rule_pct", "champion_test_wrmsse"}


def test_champion_selected_on_validation_only(pipeline_run: RunArtifacts) -> None:
    board = pipeline_run.load_frame("leaderboard")
    val = board[board["split"] == "validation"].sort_values("rank")
    assert val["model"].iloc[0] == pipeline_run.load_json("manifest")["champion"]


def test_test_forecasts_never_overlap_validation(pipeline_run: RunArtifacts) -> None:
    val = pipeline_run.load_frame("val_forecasts")
    test = pipeline_run.load_frame("forecasts")
    assert val["ds"].max() < test["ds"].min()
    for cutoff, g in test.groupby("cutoff"):
        assert (g["ds"] > pd.Timestamp(cutoff)).all()


def test_simulation_results_are_consistent(pipeline_run: RunArtifacts) -> None:
    per_series = pipeline_run.load_frame("series_results")
    portfolio = pipeline_run.load_frame("portfolio").set_index("policy")
    totals = per_series.groupby("policy")["total_cost"].sum()
    for policy, total in totals.items():
        assert portfolio.loc[policy, "total_cost"] == pytest.approx(total)
    assert (per_series["fill_rate"].between(0, 1)).all()
    assert portfolio.loc["rule_of_thumb", "savings_vs_baseline_pct"] == 0.0


def test_mlflow_runs_logged(pipeline_run: RunArtifacts) -> None:
    assert pipeline_run.load_json("manifest")["mlflow_run_id"]


def test_leaderboard_reports_written(pipeline_run: RunArtifacts) -> None:
    by_class = pipeline_run.load_frame("leaderboard_by_class")
    assert {"validation", "test"} == set(by_class["split"])
    assert {"wape", "smape", "rmsse", "bias", "wrmsse"} <= set(by_class.columns)
    report = (pipeline_run.root / "leaderboard.md").read_text(encoding="utf-8")
    assert "## Test windows by demand class" in report
    assert (pipeline_run.root / "leaderboard.csv").is_file()


def test_mlflow_has_params_folds_and_class_metrics(pipeline_run: RunArtifacts) -> None:
    manifest = pipeline_run.load_json("manifest")
    parent = mlflow.get_run(manifest["mlflow_run_id"])
    children = mlflow.search_runs(
        experiment_ids=[parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    lgbm = children[children["tags.mlflow.runName"] == "lightgbm"].iloc[0]
    assert lgbm["params.num_leaves"] is not None
    assert lgbm["metrics.val_wape"] >= 0
    assert any(c.startswith("metrics.test_") and c.endswith("_wrmsse") and "fold" not in c
               and c not in ("metrics.test_wrmsse",) for c in children.columns)  # fmt: skip
    history = mlflow.tracking.MlflowClient().get_metric_history(lgbm["run_id"], "val_fold_wrmsse")
    assert [m.step for m in history] == [0, 1]
