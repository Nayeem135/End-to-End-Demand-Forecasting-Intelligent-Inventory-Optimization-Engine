"""`demand-engine run-all` end to end: synthetic data -> pipeline -> migrate -> publish."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from demand_engine.config import get_settings
from demand_engine.data.synthetic import write_synthetic_m5
from demand_engine.pipelines.cli import main

pytestmark = pytest.mark.integration

DB_NAME = "cli_e2e"


@pytest.fixture
def cli_db(pg_url: str) -> Iterator[str]:
    """A dedicated database so this test cannot affect other integration tests."""
    admin = create_engine(pg_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {DB_NAME}"))
        conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
    yield make_url(pg_url).set(database=DB_NAME).render_as_string(hide_password=False)
    admin.dispose()


def test_run_all_publishes_a_run(
    cli_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {
        "PATHS__DATA_DIR": str(tmp_path / "data"),
        "PATHS__ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        "DATABASE__URL": cli_db,
        "MLFLOW_TRACKING_URI": "",
        "DATA__N_ITEMS": "6",
        "BACKTEST__HORIZON": "14",
        "BACKTEST__N_FOLDS": "2",
        "BACKTEST__N_TEST_WINDOWS": "1",
        "BACKTEST__MIN_TRAIN_DAYS": "84",
        "MODEL__CANDIDATES": '["naive", "moving_average"]',
        "MODEL__LGBM__LAGS": "[14, 28]",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        write_synthetic_m5(tmp_path / "data" / "raw", n_items=8, n_days=360, seed=5)
        main(["run-all"])
    finally:
        get_settings.cache_clear()

    engine = create_engine(cli_db)
    with engine.connect() as conn:
        runs = conn.execute(text("SELECT run_id, champion FROM pipeline_runs")).all()
        n_policies = conn.execute(text("SELECT count(*) FROM inventory_policies")).scalar_one()
    engine.dispose()
    assert len(runs) == 1
    assert runs[0].champion in {"naive", "moving_average"}
    assert n_policies > 0
