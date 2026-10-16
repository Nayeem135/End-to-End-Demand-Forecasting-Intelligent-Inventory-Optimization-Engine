"""Publish stage: load a run's artifacts into the relational database (one transaction)."""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import Connection, Engine, delete, insert, select

from demand_engine.api.db import models as m
from demand_engine.api.db.bulk import bulk_insert
from demand_engine.pipelines.artifacts import ArtifactNotFoundError, RunArtifacts

logger = logging.getLogger(__name__)

# artifact file name -> destination table name
ARTIFACT_TABLES: dict[str, str] = {
    "series": "series",
    "leaderboard": "model_scores",
    "history": "demand_history",
    "forecasts": "forecasts",
    "val_forecasts": "validation_forecasts",
    "policies": "inventory_policies",
    "portfolio": "portfolio_results",
    "series_results": "series_results",
    "sensitivity": "sensitivity_points",
    "shap_global": "feature_importance",
    "shap_local": "shap_local",
}


def publish_run(engine: Engine, run: RunArtifacts, keep_last: int = 3) -> dict[str, int]:
    """Insert (or replace) a pipeline run and all its result tables.

    Args:
        engine: Database engine (schema must already be migrated).
        run: Artifacts of a completed pipeline run.
        keep_last: Older runs beyond this number are deleted afterwards (cascade).

    Returns:
        Row counts inserted per table.
    """
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    manifest = run.load_json("manifest")
    run_id = str(manifest["run_id"])
    counts: dict[str, int] = {}
    tables = m.Base.metadata.tables
    with engine.begin() as conn:
        _delete_runs(conn, [run_id])
        conn.execute(
            insert(m.PipelineRun).values(
                run_id=run_id,
                state=manifest["state"],
                n_series=manifest["n_series"],
                n_items=manifest["n_items"],
                n_stores=manifest["n_stores"],
                data_start=date.fromisoformat(manifest["data_start"]),
                data_end=date.fromisoformat(manifest["data_end"]),
                test_start=date.fromisoformat(manifest["test_start"]),
                champion=manifest["champion"],
                baseline_model=manifest["baseline_model"],
                headline=manifest["headline"],
                settings=manifest["settings"],
                mlflow_run_id=manifest.get("mlflow_run_id"),
            )
        )
        for artifact, table_name in ARTIFACT_TABLES.items():
            try:
                df = run.load_frame(artifact)
            except ArtifactNotFoundError:
                logger.warning("Artifact '%s' missing; table %s left empty", artifact, table_name)
                continue
            counts[table_name] = bulk_insert(conn, tables[table_name], df.assign(run_id=run_id))
        ids = (
            conn.execute(select(m.PipelineRun.run_id).order_by(m.PipelineRun.run_id.desc()))
            .scalars()
            .all()
        )
        _delete_runs(conn, list(ids[keep_last:]))
    logger.info("Published run %s: %s", run_id, counts)
    return counts


def _delete_runs(conn: Connection, run_ids: list[str]) -> None:
    """Delete runs and their rows explicitly (does not rely on FK cascade support)."""
    if not run_ids:
        return
    for table in reversed(m.Base.metadata.sorted_tables):
        if "run_id" in table.c:
            conn.execute(delete(table).where(table.c.run_id.in_(run_ids)))
