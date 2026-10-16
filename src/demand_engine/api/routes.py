"""HTTP endpoints: validate input, call the repository or service, map results to schemas."""

from __future__ import annotations

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query
from sqlalchemy import text

from demand_engine import __version__
from demand_engine.api import repository as repo
from demand_engine.api.db.models import PipelineRun
from demand_engine.api.deps import RunIdDep, SessionDep, SimulationDep
from demand_engine.api.schemas import (
    FeatureImportanceOut,
    Filters,
    ForecastPoint,
    ForecastResponse,
    Headline,
    Health,
    HistoryPoint,
    InventoryParams,
    ModelScoreOut,
    PolicyOut,
    PolicyPage,
    PolicyRow,
    PortfolioOut,
    RunSummary,
    SensitivityOut,
    SeriesDetail,
    SeriesOut,
    SeriesPage,
    SeriesResultOut,
    SeriesSimulation,
    ShapLocalOut,
    WhatIfRequest,
    WhatIfResponse,
)

router = APIRouter(prefix="/api")

UniqueId = Annotated[str, Path(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=200)]


@router.get("/health", response_model=Health, tags=["system"])
def health(session: SessionDep) -> Health:
    """Liveness and database connectivity."""
    try:
        session.execute(text("SELECT 1"))
        latest = repo.latest_run_id(session)
        return Health(status="ok", database=True, latest_run=latest, version=__version__)
    except Exception:
        return Health(status="degraded", database=False, latest_run=None, version=__version__)


def run_summary(run: PipelineRun) -> RunSummary:
    """Map a run row to its API representation (inventory parameters from its settings)."""
    inventory = run.settings["inventory"]
    return RunSummary(
        run_id=run.run_id,
        created_at=run.created_at,
        state=run.state,
        n_series=run.n_series,
        n_items=run.n_items,
        n_stores=run.n_stores,
        data_start=run.data_start,
        data_end=run.data_end,
        test_start=run.test_start,
        champion=run.champion,
        baseline_model=run.baseline_model,
        headline=Headline.model_validate(run.headline),
        inventory=InventoryParams.model_validate(
            {k: inventory[k] for k in InventoryParams.model_fields}
        ),
        mlflow_run_id=run.mlflow_run_id,
    )


@router.get("/runs", response_model=list[RunSummary], tags=["runs"])
def runs(session: SessionDep) -> list[RunSummary]:
    """All published pipeline runs, newest first."""
    return [run_summary(r) for r in repo.list_runs(session)]


@router.get("/overview", response_model=RunSummary, tags=["runs"])
def overview(session: SessionDep, run_id: RunIdDep) -> RunSummary:
    """Headline numbers of a run (default: latest)."""
    return run_summary(repo.get_run(session, run_id))


@router.get("/models/leaderboard", response_model=list[ModelScoreOut], tags=["models"])
def model_leaderboard(
    session: SessionDep, run_id: RunIdDep, split: Literal["validation", "test"] = "test"
) -> list[ModelScoreOut]:
    """Forecast accuracy of every model (lower WRMSSE is better)."""
    return [ModelScoreOut.model_validate(r) for r in repo.leaderboard(session, run_id, split)]


@router.get("/filters", response_model=Filters, tags=["series"])
def filters(session: SessionDep, run_id: RunIdDep) -> Filters:
    """Distinct values for dashboard filter controls."""
    return Filters(
        stores=repo.distinct_values(session, run_id, "store_id"),
        categories=repo.distinct_values(session, run_id, "cat_id"),
        departments=repo.distinct_values(session, run_id, "dept_id"),
        demand_classes=repo.distinct_values(session, run_id, "demand_class"),
    )


def _query(
    search: str | None,
    store_id: str | None,
    cat_id: str | None,
    demand_class: str | None,
    sort: repo.SortField,
    descending: bool,
    page: int,
    page_size: int,
) -> repo.SeriesQuery:
    return repo.SeriesQuery(
        search, store_id, cat_id, demand_class, sort, descending, page, page_size
    )


@router.get("/series", response_model=SeriesPage, tags=["series"])
def series_list(
    session: SessionDep,
    run_id: RunIdDep,
    search: Annotated[str | None, Query(max_length=64)] = None,
    store_id: str | None = None,
    cat_id: str | None = None,
    demand_class: str | None = None,
    sort: repo.SortField = "avg_daily_sales",
    descending: bool = True,
    page: Page = 1,
    page_size: PageSize = 25,
) -> SeriesPage:
    """Search and browse SKU-store series."""
    q = _query(search, store_id, cat_id, demand_class, sort, descending, page, page_size)
    rows, total = repo.list_series(session, run_id, q)
    return SeriesPage(
        items=[SeriesOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/series/{unique_id}", response_model=SeriesDetail, tags=["series"])
def series_detail(session: SessionDep, run_id: RunIdDep, unique_id: UniqueId) -> SeriesDetail:
    """Metadata, inventory recommendation and simulation KPIs of one series."""
    return SeriesDetail(
        series=SeriesOut.model_validate(repo.get_series(session, run_id, unique_id)),
        policy=PolicyOut.model_validate(repo.get_policy(session, run_id, unique_id)),
        results=[
            SeriesResultOut.model_validate(r)
            for r in repo.series_results(session, run_id, unique_id)
        ],
    )


@router.get("/series/{unique_id}/forecast", response_model=ForecastResponse, tags=["series"])
def series_forecast(session: SessionDep, run_id: RunIdDep, unique_id: UniqueId) -> ForecastResponse:
    """Recent actuals plus champion and baseline forecasts with intervals."""
    repo.get_series(session, run_id, unique_id)
    run = repo.get_run(session, run_id)
    grouped: dict[str, list[ForecastPoint]] = {}
    for row in repo.forecasts(session, run_id, unique_id):
        grouped.setdefault(row.model, []).append(
            ForecastPoint(ds=row.ds, yhat=row.yhat, yhat_lo=row.yhat_lo, yhat_hi=row.yhat_hi)
        )
    return ForecastResponse(
        unique_id=unique_id,
        champion=run.champion,
        baseline_model=run.baseline_model,
        test_start=run.test_start,
        history=[
            HistoryPoint(ds=h.ds, y=h.y, sell_price=h.sell_price)
            for h in repo.history(session, run_id, unique_id)
        ],
        forecasts=grouped,
    )


@router.get("/series/{unique_id}/simulation", response_model=SeriesSimulation, tags=["simulation"])
def series_simulation(
    session: SessionDep, run_id: RunIdDep, simulation: SimulationDep, unique_id: UniqueId
) -> SeriesSimulation:
    """Day-by-day simulated inventory of every policy for one series."""
    return SeriesSimulation(
        unique_id=unique_id, trajectories=simulation.trajectories(session, run_id, unique_id)
    )


@router.get(
    "/series/{unique_id}/explanation", response_model=list[ShapLocalOut], tags=["explainability"]
)
def series_explanation(
    session: SessionDep, run_id: RunIdDep, unique_id: UniqueId
) -> list[ShapLocalOut]:
    """Strongest SHAP drivers of this series' forecast (log-demand contributions)."""
    repo.get_series(session, run_id, unique_id)
    return [ShapLocalOut.model_validate(r) for r in repo.shap_local(session, run_id, unique_id)]


@router.get(
    "/explainability/global", response_model=list[FeatureImportanceOut], tags=["explainability"]
)
def global_explanation(
    session: SessionDep, run_id: RunIdDep, limit: Annotated[int, Query(ge=1, le=100)] = 20
) -> list[FeatureImportanceOut]:
    """Mean absolute SHAP value per feature across the last test window."""
    return [
        FeatureImportanceOut.model_validate(r)
        for r in repo.feature_importance(session, run_id, limit)
    ]


@router.get("/inventory/policies", response_model=PolicyPage, tags=["inventory"])
def inventory_policies(
    session: SessionDep,
    run_id: RunIdDep,
    search: Annotated[str | None, Query(max_length=64)] = None,
    store_id: str | None = None,
    cat_id: str | None = None,
    demand_class: str | None = None,
    sort: repo.SortField = "avg_daily_sales",
    descending: bool = True,
    page: Page = 1,
    page_size: PageSize = 25,
) -> PolicyPage:
    """Safety stock, reorder point and order quantity per series."""
    q = _query(search, store_id, cat_id, demand_class, sort, descending, page, page_size)
    rows, total = repo.list_policies(session, run_id, q)
    items = [
        PolicyRow(
            **PolicyOut.model_validate(p).model_dump(),
            item_id=s.item_id,
            store_id=s.store_id,
            cat_id=s.cat_id,
            demand_class=s.demand_class,
            avg_price=s.avg_price,
        )
        for p, s in rows
    ]
    return PolicyPage(items=items, total=total, page=page, page_size=page_size)


@router.get("/simulation/portfolio", response_model=list[PortfolioOut], tags=["simulation"])
def simulation_portfolio(session: SessionDep, run_id: RunIdDep) -> list[PortfolioOut]:
    """Portfolio cost and service KPIs per policy at the run's parameters."""
    return [PortfolioOut.model_validate(r) for r in repo.portfolio(session, run_id)]


@router.get("/simulation/sensitivity", response_model=list[SensitivityOut], tags=["simulation"])
def simulation_sensitivity(session: SessionDep, run_id: RunIdDep) -> list[SensitivityOut]:
    """Cost / fill-rate trade-off across target service levels."""
    return [SensitivityOut.model_validate(r) for r in repo.sensitivity(session, run_id)]


@router.post("/simulation/what-if", response_model=WhatIfResponse, tags=["simulation"])
def what_if(
    session: SessionDep, run_id: RunIdDep, simulation: SimulationDep, body: WhatIfRequest
) -> WhatIfResponse:
    """Re-run the inventory simulation live; unspecified parameters keep the run's values."""
    start = time.perf_counter()
    result = simulation.what_if(session, run_id, body)
    p = result.params
    params = InventoryParams(
        service_level=p.service_level,
        lead_time_days=p.lead_time_days,
        review_period_days=p.review_period_days,
        ordering_cost=p.ordering_cost,
        holding_cost_rate=p.holding_cost_rate,
        stockout_cost_multiplier=p.stockout_cost_multiplier,
    )
    portfolio = [PortfolioOut.model_validate(r) for r in result.portfolio.to_dict("records")]
    return WhatIfResponse(
        params=params, portfolio=portfolio, elapsed_ms=(time.perf_counter() - start) * 1000
    )
