"""FastAPI application factory.

Run with ``uvicorn demand_engine.api.main:create_app --factory`` or ``demand-engine serve``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from demand_engine import __version__
from demand_engine.api.db.session import get_engine, session_factory
from demand_engine.api.repository import NotFoundError, latest_run_id
from demand_engine.api.routes import router
from demand_engine.api.simulation_service import SimulationService
from demand_engine.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Optional explicit settings (tests); defaults to :func:`get_settings`.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        threading.Thread(target=_warm_cache, args=(app,), daemon=True).start()
        yield

    app = FastAPI(
        lifespan=lifespan,
        title="Demand Forecasting & Inventory Optimization API",
        version=__version__,
        description=(
            "Forecasts daily SKU-store demand (M5 / Walmart), converts forecasts into safety "
            "stock, reorder points and order quantities, and proves cost savings by simulation."
        ),
    )
    db = settings.database
    engine = get_engine(db.url, db.echo, db.connect_timeout)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory(engine)
    app.state.simulation = SimulationService()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": str(exc)}
        )

    app.include_router(router)
    _mount_frontend(app, settings)
    return app


def _warm_cache(app: FastAPI) -> None:
    """Pre-load the latest run's simulator inputs so the first what-if is fast (best effort)."""
    try:
        with app.state.session_factory() as session:
            run_id = latest_run_id(session)
            if run_id is not None:
                app.state.simulation.inputs(session, run_id)
    except Exception:
        logger.warning("Simulation cache warm-up skipped", exc_info=True)


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built React dashboard from the API process when available (single-URL demo).

    Real files are served as-is; any other non-API path returns ``index.html`` so that
    client-side routes such as ``/what-if`` survive a page refresh.
    """
    static_dir = settings.api.static_dir
    if static_dir is None:
        return
    if not (static_dir / "index.html").is_file():
        logger.warning("static_dir %s has no index.html; dashboard not mounted", static_dir)
        return
    root = static_dir.resolve()
    if (root / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def dashboard(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
        candidate = (root / path).resolve()
        if path and candidate.is_file() and _is_within(candidate, root):
            return FileResponse(candidate)
        return FileResponse(root / "index.html")

    logger.info("Serving dashboard from %s", static_dir)


def _is_within(path: Path, root: Path) -> bool:
    """True if ``path`` is inside ``root`` (guards against ``../`` traversal)."""
    return root == path or root in path.parents
