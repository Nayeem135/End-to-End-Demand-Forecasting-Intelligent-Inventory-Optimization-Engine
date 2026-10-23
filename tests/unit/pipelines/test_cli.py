import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from demand_engine.config import get_settings
from demand_engine.data.schema import SALES_FILE
from demand_engine.pipelines.cli import build_parser, main, start_embedded_postgres


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the CLI's settings at a temp directory."""
    monkeypatch.setenv("PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PATHS__ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BACKTEST__HORIZON", "14")
    monkeypatch.setenv("BACKTEST__N_FOLDS", "1")
    monkeypatch.setenv("BACKTEST__N_TEST_WINDOWS", "1")
    monkeypatch.setenv("BACKTEST__MIN_TRAIN_DAYS", "84")
    monkeypatch.setenv("MODEL__LGBM__LAGS", "[14, 28]")
    monkeypatch.setenv("DATA__N_ITEMS", "5")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_parser_lists_all_commands() -> None:
    parser = build_parser()
    for cmd in ["synthetic", "ingest", "run", "db-upgrade", "publish", "run-all", "serve", "demo"]:
        assert parser.parse_args([cmd]).command == cmd


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_synthetic_then_ingest(isolated_env: Path) -> None:
    main(["synthetic", "--n-items", "6", "--n-days", "300"])
    raw = isolated_env / "data" / "raw"
    assert (raw / SALES_FILE).is_file()
    with pytest.raises(SystemExit, match="--force"):
        main(["synthetic"])
    main(["ingest"])
    assert (isolated_env / "data" / "processed" / "sales.parquet").is_file()


def test_embedded_postgres_waits_through_crash_recovery(tmp_path: Path) -> None:
    calls: list[Path] = []
    sleeps: list[float] = []

    def flaky_get_server(pgdata: Path, cleanup_mode: str) -> str:
        calls.append(pgdata)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("pg_ctl", 10)  # crash recovery exceeds pg_ctl timeout
        if len(calls) == 2:
            raise AssertionError  # attached, but the server is not 'ready' yet
        return "server"

    server = start_embedded_postgres(
        tmp_path, flaky_get_server, attempts=5, delay_seconds=2, sleep=sleeps.append
    )
    assert server == "server"
    assert len(calls) == 3
    assert sleeps == [2, 2]


def test_embedded_postgres_gives_up_after_attempts(tmp_path: Path) -> None:
    def never(pgdata: Path, cleanup_mode: str) -> str:
        raise subprocess.TimeoutExpired("pg_ctl", 10)

    with pytest.raises(subprocess.TimeoutExpired):
        start_embedded_postgres(tmp_path, never, attempts=2, delay_seconds=0, sleep=lambda _: None)
