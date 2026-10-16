"""Read-side data access. All SQL lives here; routers only orchestrate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import Executable

from demand_engine.api.db import models as m

SortField = Literal["unique_id", "avg_daily_sales", "avg_price", "champion_mae"]


class NotFoundError(LookupError):
    """Raised when a requested run or series does not exist."""


def latest_run_id(session: Session) -> str | None:
    """Most recent published run id, or ``None`` when the database is empty."""
    return session.execute(
        select(m.PipelineRun.run_id).order_by(m.PipelineRun.run_id.desc()).limit(1)
    ).scalar_one_or_none()


def get_run(session: Session, run_id: str) -> m.PipelineRun:
    """Load a run or raise :class:`NotFoundError`."""
    run = session.get(m.PipelineRun, run_id)
    if run is None:
        raise NotFoundError(f"run '{run_id}' not found")
    return run


def list_runs(session: Session) -> Sequence[m.PipelineRun]:
    """All runs, newest first."""
    return (
        session.execute(select(m.PipelineRun).order_by(m.PipelineRun.run_id.desc())).scalars().all()
    )


def leaderboard(session: Session, run_id: str, split: str) -> Sequence[m.ModelScore]:
    """Model scores for a split, ranked."""
    stmt = select(m.ModelScore).where(m.ModelScore.run_id == run_id, m.ModelScore.split == split)
    return session.execute(stmt.order_by(m.ModelScore.rank)).scalars().all()


@dataclass(frozen=True)
class SeriesQuery:
    """Filters and paging for the series catalogue."""

    search: str | None = None
    store_id: str | None = None
    cat_id: str | None = None
    demand_class: str | None = None
    sort: SortField = "avg_daily_sales"
    descending: bool = True
    page: int = 1
    page_size: int = 25


def _filtered(stmt: Select[m.SeriesMeta], q: SeriesQuery) -> Select[m.SeriesMeta]:
    if q.search:
        stmt = stmt.where(m.SeriesMeta.unique_id.ilike(f"%{q.search}%"))
    if q.store_id:
        stmt = stmt.where(m.SeriesMeta.store_id == q.store_id)
    if q.cat_id:
        stmt = stmt.where(m.SeriesMeta.cat_id == q.cat_id)
    if q.demand_class:
        stmt = stmt.where(m.SeriesMeta.demand_class == q.demand_class)
    return stmt


def list_series(
    session: Session, run_id: str, q: SeriesQuery
) -> tuple[Sequence[m.SeriesMeta], int]:
    """Filtered, sorted, paginated series plus the total match count."""
    base = _filtered(select(m.SeriesMeta).where(m.SeriesMeta.run_id == run_id), q)
    total = session.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    column = getattr(m.SeriesMeta, q.sort)
    order = column.desc() if q.descending else column.asc()
    stmt = (
        base.order_by(order, m.SeriesMeta.unique_id)
        .offset((q.page - 1) * q.page_size)
        .limit(q.page_size)
    )
    return session.execute(stmt).scalars().all(), int(total)


def get_series(session: Session, run_id: str, unique_id: str) -> m.SeriesMeta:
    """One series or :class:`NotFoundError`."""
    row = session.get(m.SeriesMeta, (run_id, unique_id))
    if row is None:
        raise NotFoundError(f"series '{unique_id}' not found")
    return row


def get_policy(session: Session, run_id: str, unique_id: str) -> m.InventoryPolicyRow:
    """Inventory recommendation for a series."""
    row = session.get(m.InventoryPolicyRow, (run_id, unique_id))
    if row is None:
        raise NotFoundError(f"no policy for series '{unique_id}'")
    return row


def series_results(session: Session, run_id: str, unique_id: str) -> Sequence[m.SeriesResult]:
    """Per-policy simulation KPIs of a series."""
    stmt = select(m.SeriesResult).where(
        m.SeriesResult.run_id == run_id, m.SeriesResult.unique_id == unique_id
    )
    return session.execute(stmt.order_by(m.SeriesResult.policy)).scalars().all()


def list_policies(
    session: Session, run_id: str, q: SeriesQuery
) -> tuple[list[tuple[m.InventoryPolicyRow, m.SeriesMeta]], int]:
    """Inventory recommendations joined with series metadata, filtered and paginated."""
    series_ids = _filtered(select(m.SeriesMeta).where(m.SeriesMeta.run_id == run_id), q).subquery()
    base = (
        select(m.InventoryPolicyRow, m.SeriesMeta)
        .join(
            m.SeriesMeta,
            (m.SeriesMeta.run_id == m.InventoryPolicyRow.run_id)
            & (m.SeriesMeta.unique_id == m.InventoryPolicyRow.unique_id),
        )
        .where(m.InventoryPolicyRow.run_id == run_id)
        .where(m.InventoryPolicyRow.unique_id.in_(select(series_ids.c.unique_id)))
    )
    total = session.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    sort_col = getattr(m.SeriesMeta, q.sort)
    order = sort_col.desc() if q.descending else sort_col.asc()
    stmt = (
        base.order_by(order, m.SeriesMeta.unique_id)
        .offset((q.page - 1) * q.page_size)
        .limit(q.page_size)
    )
    rows = [(p, s) for p, s in session.execute(stmt).tuples().all()]
    return rows, int(total)


def history(session: Session, run_id: str, unique_id: str) -> Sequence[m.DemandHistory]:
    """Stored demand history of a series, chronological."""
    stmt = select(m.DemandHistory).where(
        m.DemandHistory.run_id == run_id, m.DemandHistory.unique_id == unique_id
    )
    return session.execute(stmt.order_by(m.DemandHistory.ds)).scalars().all()


def forecasts(session: Session, run_id: str, unique_id: str) -> Sequence[m.ForecastRow]:
    """Test-period forecasts of a series for all stored models."""
    stmt = select(m.ForecastRow).where(
        m.ForecastRow.run_id == run_id, m.ForecastRow.unique_id == unique_id
    )
    return session.execute(stmt.order_by(m.ForecastRow.model, m.ForecastRow.ds)).scalars().all()


def portfolio(session: Session, run_id: str) -> Sequence[m.PortfolioResult]:
    """Stored portfolio KPIs per policy."""
    stmt = select(m.PortfolioResult).where(m.PortfolioResult.run_id == run_id)
    return session.execute(stmt.order_by(m.PortfolioResult.total_cost.desc())).scalars().all()


def sensitivity(session: Session, run_id: str) -> Sequence[m.SensitivityPoint]:
    """Stored service-level sensitivity curve."""
    stmt = select(m.SensitivityPoint).where(m.SensitivityPoint.run_id == run_id)
    return (
        session.execute(stmt.order_by(m.SensitivityPoint.policy, m.SensitivityPoint.service_level))
        .scalars()
        .all()
    )


def feature_importance(
    session: Session, run_id: str, limit: int
) -> Sequence[m.FeatureImportanceRow]:
    """Top global SHAP features."""
    stmt = select(m.FeatureImportanceRow).where(m.FeatureImportanceRow.run_id == run_id)
    return (
        session.execute(stmt.order_by(m.FeatureImportanceRow.mean_abs_shap.desc()).limit(limit))
        .scalars()
        .all()
    )


def shap_local(session: Session, run_id: str, unique_id: str) -> Sequence[m.ShapLocalRow]:
    """Top SHAP contributions for one series, strongest first."""
    stmt = select(m.ShapLocalRow).where(
        m.ShapLocalRow.run_id == run_id, m.ShapLocalRow.unique_id == unique_id
    )
    rows = session.execute(stmt).scalars().all()
    return sorted(rows, key=lambda r: abs(r.shap_value), reverse=True)


def distinct_values(session: Session, run_id: str, column: str) -> list[str]:
    """Sorted distinct values of a series metadata column."""
    col = getattr(m.SeriesMeta, column)
    stmt = select(col).where(m.SeriesMeta.run_id == run_id).distinct().order_by(col)
    return [str(v) for v in session.execute(stmt).scalars().all()]


def read_frame(session: Session, stmt: Executable) -> pd.DataFrame:
    """Execute a select and return a DataFrame (column names from the statement)."""
    result = session.execute(stmt)
    return pd.DataFrame(result.all(), columns=list(result.keys()))
