"""FastAPI dependencies: DB sessions, run resolution, shared services."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from demand_engine.api import repository as repo
from demand_engine.api.simulation_service import SimulationService


def get_session(request: Request) -> Iterator[Session]:
    """Request-scoped database session."""
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_simulation_service(request: Request) -> SimulationService:
    """Process-wide simulation service (holds the input cache)."""
    service: SimulationService = request.app.state.simulation
    return service


SessionDep = Annotated[Session, Depends(get_session)]
SimulationDep = Annotated[SimulationService, Depends(get_simulation_service)]


def resolve_run_id(
    session: SessionDep,
    run_id: Annotated[
        str | None, Query(description="Pipeline run id; defaults to the latest run.")
    ] = None,
) -> str:
    """Return the requested run id (validated) or the latest one."""
    if run_id is not None:
        try:
            repo.get_run(session, run_id)
        except repo.NotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        return run_id
    latest = repo.latest_run_id(session)
    if latest is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "no pipeline run published yet; run 'demand-engine run-all'"
        )
    return latest


RunIdDep = Annotated[str, Depends(resolve_run_id)]
