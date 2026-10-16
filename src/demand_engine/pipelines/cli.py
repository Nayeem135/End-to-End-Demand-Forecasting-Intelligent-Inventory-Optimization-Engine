"""Command-line interface: ``demand-engine <command>``.

Commands
--------
synthetic   Write a synthetic M5-format dataset to data/raw (offline demo / CI).
ingest      Build the validated demand table for the configured subset.
run         Forecast, backtest, optimise and simulate; write run artifacts.
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
from collections.abc import Sequence
from pathlib import Path

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


def _cmd_demo(settings: Settings, args: argparse.Namespace) -> None:
    """Self-contained demo using an embedded PostgreSQL server (pip package ``pgserver``)."""
    try:
        import pgserver
    except ImportError as exc:  # pragma: no cover - depends on optional dev dependency
        raise SystemExit(
            "the 'demo' command needs the dev dependency 'pgserver' (uv sync)"
        ) from exc

    pgdata = settings.paths.data_dir / "pgdata"
    pgdata.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(pgdata.resolve(), cleanup_mode="stop")
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
    configure_logging(settings.log_level)
    args.func(settings, args)


if __name__ == "__main__":  # pragma: no cover
    main()
