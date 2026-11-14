"""Command-line interface: ``demand-engine <command>``.

Commands
--------
synthetic   Write a synthetic M5-format dataset to data/raw (offline demo / CI).
ingest      Build the validated demand table for the configured subset.
run         Forecast, backtest, optimise and simulate; write run artifacts.
tune        Optuna tuning of LightGBM (small budget) on the first validation fold.
db-upgrade  Apply database migrations.
publish     Load the latest (or a given) run's artifacts into the database.
run-all     ingest -> run -> db-upgrade -> publish.
serve       Start the API (optionally serving the built dashboard).
demo        No Docker needed: embedded PostgreSQL + migrate + publish + serve.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from demand_engine.config import Settings, get_settings
from demand_engine.data.schema import SALES_FILE
from demand_engine.logging_config import configure_logging

logger = logging.getLogger("demand_engine.cli")


def _cmd_synthetic(settings: Settings, args: argparse.Namespace) -> None:
    from demand_engine.data.synthetic import write_synthetic_m5

    raw = settings.paths.raw_dir
    if (raw / SALES_FILE).exists() and not args.force:
        raise SystemExit(
            f"{raw / SALES_FILE} exists; pass --force to overwrite it with synthetic data"
        )
    write_synthetic_m5(
        raw,
        n_items=args.n_items,
        n_days=args.n_days,
        states=(settings.data.state,),
        seed=settings.seed,
    )


def _cmd_ingest(settings: Settings, _: argparse.Namespace) -> None:
    from demand_engine.pipelines.ingest import run_ingest

    run_ingest(settings)


def _cmd_run(settings: Settings, _: argparse.Namespace) -> None:
    from demand_engine.pipelines.ingest import load_demand_table
    from demand_engine.pipelines.run_all import run_pipeline

    run_pipeline(load_demand_table(settings), settings)


def _cmd_db_upgrade(settings: Settings, _: argparse.Namespace) -> None:
    from demand_engine.api.db.migrate import upgrade

    upgrade(settings.database.url)
    logger.info("Database schema is up to date")


def _cmd_tune(settings: Settings, args: argparse.Namespace) -> None:
    import json

    from demand_engine.pipelines.ingest import load_demand_table
    from demand_engine.pipelines.tracking import Tracker
    from demand_engine.pipelines.tuning import tune_lightgbm

    tracker = Tracker(
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment,
        settings.paths.artifacts_dir / "mlflow",
    )
    trials = args.trials or settings.model.tuning_trials
    result = tune_lightgbm(load_demand_table(settings), settings, trials, tracker)
    out = settings.paths.artifacts_dir / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_wrmsse": result.baseline_wrmsse,
        "best_wrmsse": result.best_wrmsse,
        "improvement_pct": result.improvement_pct,
        "best_params": result.best_params,
    }
    (out / "best_params.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    result.trials.to_csv(out / "trials.csv", index=False)
    logger.info(
        "Tuning: WRMSSE %.4f -> %.4f (%.2f %%); best params written to %s",
        result.baseline_wrmsse,
        result.best_wrmsse,
        result.improvement_pct,
        out / "best_params.json",
    )
    for key, value in result.best_params.items():
        logger.info("  MODEL__LGBM__%s=%s", key.upper(), value)


def _cmd_publish(settings: Settings, args: argparse.Namespace) -> None:
    from demand_engine.api.db.session import get_engine
    from demand_engine.pipelines.artifacts import RunArtifacts, latest_run
    from demand_engine.pipelines.publish import publish_run

    run = (
        RunArtifacts(settings.paths.artifacts_dir / "runs" / args.run_id)
        if args.run_id
        else latest_run(settings.paths.artifacts_dir)
    )
    db = settings.database
    publish_run(get_engine(db.url, db.echo, db.connect_timeout), run)


def _cmd_run_all(settings: Settings, args: argparse.Namespace) -> None:
    _cmd_ingest(settings, args)
    _cmd_run(settings, args)
    if args.no_publish:
        return
    _cmd_db_upgrade(settings, args)
    args.run_id = None
    _cmd_publish(settings, args)


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> None:
    import uvicorn

    from demand_engine.api.main import create_app

    if args.dashboard:
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(update={"static_dir": Path(args.dashboard)})}
        )
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


def start_embedded_postgres(
    pgdata: Path,
    get_server: Callable[..., Any],
    attempts: int = 6,
    delay_seconds: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Start (or attach to) the embedded PostgreSQL server, tolerating slow crash recovery.

    After an unclean shutdown PostgreSQL replays its log and syncs the data directory
    before accepting connections, which can exceed ``pgserver``'s fixed 10 s start
    timeout. The server keeps starting in the background, and ``pgserver`` attaches to
    an already-running server, so waiting and retrying is enough. While attached to a
    server that is still recovering, ``pgserver`` fails an internal ``assert status ==
    'ready'``; that ``AssertionError`` therefore also means "not ready yet".

    Raises:
        subprocess.TimeoutExpired | AssertionError: If the server is still not ready after
            ``attempts`` tries.
    """
    for attempt in range(1, attempts + 1):
        try:
            return get_server(pgdata, cleanup_mode="stop")
        except (subprocess.TimeoutExpired, AssertionError):
            if attempt == attempts:
                raise
            logger.warning(
                "PostgreSQL is still starting (recovering from an unclean shutdown?); "
                "retrying in %.0fs (%d/%d)",
                delay_seconds,
                attempt,
                attempts,
            )
            sleep(delay_seconds)
    raise RuntimeError("unreachable")  # pragma: no cover - loop always returns or raises


def _fresh_pgserver(pgdata: Path, cleanup_mode: str) -> Any:  # pragma: no cover - needs pgserver
    """``pgserver.get_server`` without its stale-handle cache.

    pgserver registers a server handle in a class-level cache *before* starting
    PostgreSQL, so a start that timed out leaves a broken handle that every later
    ``get_server`` call would return. Evicting it lets a retry attach to the server once
    crash recovery has finished.
    """
    from pgserver import postgres_server

    postgres_server.PostgresServer._instances.pop(pgdata, None)
    return postgres_server.get_server(pgdata, cleanup_mode=cleanup_mode)


def _cmd_demo(settings: Settings, args: argparse.Namespace) -> None:
    """Self-contained demo using an embedded PostgreSQL server (pip package ``pgserver``)."""
    try:
        import pgserver  # noqa: F401 - availability check; used via _fresh_pgserver
    except ImportError as exc:  # pragma: no cover - depends on optional dev dependency
        raise SystemExit(
            "the 'demo' command needs the dev dependency 'pgserver' (uv sync)"
        ) from exc

    pgdata = settings.paths.data_dir / "pgdata"
    pgdata.mkdir(parents=True, exist_ok=True)
    server = start_embedded_postgres(pgdata.resolve(), _fresh_pgserver)
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)
    logger.info("Embedded PostgreSQL running at %s", url)
    os.environ["DATABASE__URL"] = url
    settings = settings.model_copy(
        update={"database": settings.database.model_copy(update={"url": url})}
    )
    _cmd_db_upgrade(settings, args)
    args.run_id = None
    _cmd_publish(settings, args)
    if args.dashboard is None and Path("frontend/dist/index.html").is_file():
        args.dashboard = "frontend/dist"
    _cmd_serve(settings, args)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(prog="demand-engine", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synthetic", help="write a synthetic M5-format dataset")
    p.add_argument("--n-items", type=int, default=150)
    p.add_argument("--n-days", type=int, default=900)
    p.add_argument("--force", action="store_true", help="overwrite existing raw files")
    p.set_defaults(func=_cmd_synthetic)

    sub.add_parser("ingest", help="build the demand table").set_defaults(func=_cmd_ingest)
    sub.add_parser("run", help="forecast + optimise + simulate").set_defaults(func=_cmd_run)

    p = sub.add_parser("tune", help="Optuna tuning of LightGBM on the first validation fold")
    p.add_argument("--trials", type=int, default=None, help="trial budget (default: config)")
    p.set_defaults(func=_cmd_tune)
    sub.add_parser("db-upgrade", help="apply database migrations").set_defaults(
        func=_cmd_db_upgrade
    )

    p = sub.add_parser("publish", help="load run artifacts into the database")
    p.add_argument("--run-id", default=None, help="run directory name (default: latest)")
    p.set_defaults(func=_cmd_publish)

    p = sub.add_parser("run-all", help="ingest, run, migrate and publish")
    p.add_argument("--no-publish", action="store_true", help="skip database steps")
    p.set_defaults(func=_cmd_run_all)

    for name, func in (("serve", _cmd_serve), ("demo", _cmd_demo)):
        p = sub.add_parser(name, help="start the API" if name == "serve" else "embedded-DB demo")
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--dashboard", default=None, help="path to built frontend (frontend/dist)")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    args.func(settings, args)


if __name__ == "__main__":  # pragma: no cover
    main()
