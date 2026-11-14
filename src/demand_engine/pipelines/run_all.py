"""End-to-end pipeline: demand table -> forecasts -> inventory simulation -> artifacts."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from demand_engine.config import Settings
from demand_engine.data.schema import DS, PRICE, UNIQUE_ID, Y
from demand_engine.evaluation.report import leaderboard_report
from demand_engine.pipelines.artifacts import RunArtifacts, new_run
from demand_engine.pipelines.forecasting import ForecastingOutput, run_forecasting, series_catalogue
from demand_engine.pipelines.optimize import OptimizationOutput, run_optimization
from demand_engine.pipelines.tracking import Tracker
from demand_engine.types import cell

logger = logging.getLogger(__name__)

HISTORY_DAYS_FOR_CHARTS = 112


def run_pipeline(df: pd.DataFrame, settings: Settings) -> RunArtifacts:
    """Run forecasting and optimisation on ``df`` and persist all artifacts.

    Args:
        df: Validated long demand table (see :func:`demand_engine.pipelines.ingest.run_ingest`).
        settings: Global settings.

    Returns:
        Handle to the completed run's artifact directory.
    """
    start = time.perf_counter()
    store = new_run(settings.paths.artifacts_dir)
    tracker = Tracker(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        settings.paths.artifacts_dir / "mlflow",
    )
    with tracker.run(f"pipeline-{store.run_id}") as parent_id:
        tracker.log_params(settings.model_dump(mode="json", exclude={"database"}))
        fc = run_forecasting(df, settings, tracker)
        opt = run_optimization(df, fc, settings)
        manifest = _save_artifacts(store, df, settings, fc, opt, parent_id)
        tracker.set_tags({"champion": fc.champion, "artifacts_run_id": store.run_id})
        tracker.log_metrics(manifest["headline"])
        for name in ("leaderboard.md", "leaderboard.csv", "leaderboard_by_class.csv"):
            tracker.log_artifact(store.root / name)
        tracker.log_artifact(store.root / "portfolio.parquet")
    logger.info("Pipeline finished in %.1fs -> %s", time.perf_counter() - start, store.root)
    return store


def _save_artifacts(
    store: RunArtifacts,
    df: pd.DataFrame,
    settings: Settings,
    fc: ForecastingOutput,
    opt: OptimizationOutput,
    mlflow_run_id: str | None,
) -> dict[str, Any]:
    test_start = fc.plan.test_cutoffs[0]
    chart_start = test_start - pd.Timedelta(days=HISTORY_DAYS_FOR_CHARTS)
    store.save_frame("history", df.loc[df[DS] > chart_start, [UNIQUE_ID, DS, Y, PRICE]])

    champion_test = fc.test[fc.champion].predictions
    store.save_frame("series", series_catalogue(df, fc.plan, champion_test))

    boards = [
        fc.leaderboard_validation.assign(split="validation"),
        fc.leaderboard_test.assign(split="test"),
    ]
    board = pd.concat(boards, ignore_index=True)
    board["is_champion"] = board["model"] == fc.champion
    store.save_frame("leaderboard", board)
    store.save_frame("leaderboard_by_class", fc.leaderboard_by_class)
    board.to_csv(store.root / "leaderboard.csv", index=False)
    fc.leaderboard_by_class.to_csv(store.root / "leaderboard_by_class.csv", index=False)
    (store.root / "leaderboard.md").write_text(
        leaderboard_report(
            fc.leaderboard_validation, fc.leaderboard_test, fc.leaderboard_by_class, fc.champion
        ),
        encoding="utf-8",
    )

    keep = list(dict.fromkeys([fc.champion, fc.baseline]))
    forecast_cols = [UNIQUE_ID, DS, "cutoff", "model", "yhat", "yhat_lo", "yhat_hi"]
    store.save_frame("forecasts", pd.concat([fc.test[m].predictions[forecast_cols] for m in keep]))
    val_cols = [UNIQUE_ID, DS, "cutoff", "model", Y, "yhat"]
    store.save_frame(
        "val_forecasts", pd.concat([fc.validation[m].predictions[val_cols] for m in keep])
    )

    result = opt.result
    store.save_frame("policies", result.policy_table)
    store.save_frame("portfolio", result.portfolio)
    store.save_frame("sensitivity", opt.sensitivity)
    per_series = [
        out.summary.reset_index().assign(policy=name) for name, out in result.outputs.items()
    ]
    store.save_frame("series_results", pd.concat(per_series, ignore_index=True))

    if fc.explanation is not None:
        imp = fc.explanation.global_importance
        if fc.feature_gain is not None:
            imp = imp.merge(fc.feature_gain, on="feature", how="left")
        store.save_frame("shap_global", imp)
        store.save_frame("shap_local", fc.explanation.local)

    port = result.portfolio.set_index("policy")
    val = fc.leaderboard_validation.set_index("model")
    test = fc.leaderboard_test.set_index("model")
    ma_cost = cell(port, "ma_safety_stock", "total_cost")
    fd_cost = cell(port, "forecast_driven", "total_cost")
    headline = {
        "champion_test_wrmsse": cell(test, fc.champion, "wrmsse"),
        "baseline_test_wrmsse": float(fc.test[fc.baseline].summary["wrmsse"]),
        "champion_val_wrmsse": cell(val, fc.champion, "wrmsse"),
        "savings_vs_rule_pct": cell(port, "forecast_driven", "savings_vs_baseline_pct"),
        "savings_vs_ma_pct": 100.0 * (ma_cost - fd_cost) / ma_cost if ma_cost > 0 else 0.0,
        "fill_rate_forecast_driven": cell(port, "forecast_driven", "fill_rate"),
        "fill_rate_rule_of_thumb": cell(port, "rule_of_thumb", "fill_rate"),
    }
    manifest: dict[str, Any] = {
        "run_id": store.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "state": settings.data.state,
        "n_series": int(df[UNIQUE_ID].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "n_stores": int(df["store_id"].nunique()),
        "data_start": df[DS].min().date().isoformat(),
        "data_end": df[DS].max().date().isoformat(),
        "test_start": (test_start + pd.Timedelta(days=1)).date().isoformat(),
        "validation_cutoffs": [c.date().isoformat() for c in fc.plan.validation_cutoffs],
        "test_cutoffs": [c.date().isoformat() for c in fc.plan.test_cutoffs],
        "champion": fc.champion,
        "baseline_model": fc.baseline,
        "headline": headline,
        "mlflow_run_id": mlflow_run_id,
        "settings": settings.model_dump(mode="json", exclude={"database"}),
    }
    store.save_json("manifest", manifest)  # written last: marks the run as complete
    return manifest
