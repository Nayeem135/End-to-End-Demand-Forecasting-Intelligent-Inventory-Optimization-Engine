"""Integration fixtures: a real PostgreSQL, a full pipeline run on synthetic data, the API.

PostgreSQL comes from ``TEST_DATABASE_URL`` (CI service container) or, locally, from
an embedded server started with the ``pgserver`` dev dependency.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine

from demand_engine.api.db.migrate import downgrade, upgrade
from demand_engine.api.main import create_app
from demand_engine.config import (
    BacktestSettings,
    DatabaseSettings,
    DataSettings,
    LightGBMSettings,
    ModelSettings,
    PathsSettings,
    Settings,
)
from demand_engine.data.synthetic import write_synthetic_m5
from demand_engine.pipelines.artifacts import RunArtifacts
from demand_engine.pipelines.ingest import load_demand_table, run_ingest
from demand_engine.pipelines.publish import publish_run
from demand_engine.pipelines.run_all import run_pipeline

pytestmark = pytest.mark.integration

# pgserver logs from an atexit hook after pytest has closed its streams; keep it quiet.
logging.getLogger("pgserver").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def pg_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """URL of an empty PostgreSQL database for this test session."""
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="delete")
    yield server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)


@pytest.fixture(scope="session")
def engine(pg_url: str) -> Iterator[Engine]:
    """Engine on a freshly migrated schema."""
    downgrade(pg_url, "base")
    upgrade(pg_url)
    eng = create_engine(pg_url)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def integration_settings(tmp_path_factory: pytest.TempPathFactory, pg_url: str) -> Settings:
    """Small but complete configuration writing into a temp directory."""
    root = tmp_path_factory.mktemp("e2e")
    return Settings(
        paths=PathsSettings(data_dir=root / "data", artifacts_dir=root / "artifacts"),
        data=DataSettings(state="CA", n_items=8, history_days=None),
        backtest=BacktestSettings(horizon=14, n_folds=2, n_test_windows=2, min_train_days=84),
        model=ModelSettings(
            candidates=["naive", "moving_average", "lightgbm"],
            baseline="moving_average",
            moving_average_window=14,
            lgbm=LightGBMSettings(
                n_estimators=30, num_leaves=15, min_data_in_leaf=10, lags=[14, 21, 28]
            ),
        ),
        database=DatabaseSettings(url=pg_url),
        mlflow_tracking_uri="sqlite:///" + (root / "mlflow.db").as_posix(),
    )


@pytest.fixture(scope="session")
def pipeline_run(integration_settings: Settings) -> RunArtifacts:
    """Run ingest + full pipeline once on synthetic M5 data."""
    raw = integration_settings.paths.raw_dir
    write_synthetic_m5(raw, n_items=10, n_days=420, states=("CA",), seed=11)
    run_ingest(integration_settings)
    return run_pipeline(load_demand_table(integration_settings), integration_settings)


@pytest.fixture(scope="session")
def published(engine: Engine, pipeline_run: RunArtifacts) -> RunArtifacts:
    """The pipeline run loaded into PostgreSQL."""
    publish_run(engine, pipeline_run)
    return pipeline_run


@pytest.fixture(scope="session")
def client(integration_settings: Settings, published: RunArtifacts) -> Iterator[TestClient]:
    """API client against the published run."""
    with TestClient(create_app(integration_settings)) as c:
        yield c
