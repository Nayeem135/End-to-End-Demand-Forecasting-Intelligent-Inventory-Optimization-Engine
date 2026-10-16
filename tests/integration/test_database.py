import json
import shutil
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect, text

from demand_engine.api.db.models import Base
from demand_engine.pipelines.artifacts import RunArtifacts
from demand_engine.pipelines.publish import ARTIFACT_TABLES, publish_run

pytestmark = pytest.mark.integration


def _count(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


def test_migrations_create_every_table(engine: Engine) -> None:
    assert set(Base.metadata.tables) <= set(inspect(engine).get_table_names())


def test_migrations_match_orm_models(engine: Engine) -> None:
    """Autogenerate must find nothing to change: models and migrations are in sync."""
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_publish_row_counts_match_artifacts(engine: Engine, published: RunArtifacts) -> None:
    for artifact, table in ARTIFACT_TABLES.items():
        assert _count(engine, table) == len(published.load_frame(artifact)), table
    assert _count(engine, "pipeline_runs") == 1


def test_publish_is_idempotent(engine: Engine, published: RunArtifacts) -> None:
    before = _count(engine, "forecasts")
    publish_run(engine, published)
    assert _count(engine, "forecasts") == before
    assert _count(engine, "pipeline_runs") == 1


def test_publish_prunes_old_runs(engine: Engine, published: RunArtifacts, tmp_path: Path) -> None:
    old_id = "00000000T000000"  # sorts before the real run => treated as oldest
    clone = tmp_path / old_id
    shutil.copytree(published.root, clone)
    manifest = json.loads((clone / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = old_id
    (clone / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    publish_run(engine, RunArtifacts(clone), keep_last=1)

    with engine.connect() as conn:
        ids = conn.execute(text("SELECT run_id FROM pipeline_runs")).scalars().all()
        orphans = conn.execute(
            text("SELECT count(*) FROM forecasts WHERE run_id = :r"), {"r": old_id}
        ).scalar_one()
    assert list(ids) == [published.run_id]
    assert orphans == 0


def test_publish_rejects_invalid_keep_last(engine: Engine, published: RunArtifacts) -> None:
    with pytest.raises(ValueError, match="keep_last"):
        publish_run(engine, published, keep_last=0)
