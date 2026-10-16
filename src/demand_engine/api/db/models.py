"""ORM models. Every result table is scoped to a pipeline run (``run_id``).

Deleting a run cascades to all of its rows, so old runs can be pruned safely.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, ClassVar

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base class for all tables."""

    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON}


class RunScoped:
    """Mixin adding the ``run_id`` primary-key column referencing :class:`PipelineRun`."""

    run_id: Mapped[str] = mapped_column(
        String(40),
        ForeignKey("pipeline_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
        sort_order=-1,  # first column => leading key of the composite primary-key index
    )


class PipelineRun(Base):
    """One execution of the end-to-end pipeline."""

    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    state: Mapped[str] = mapped_column(String(2))
    n_series: Mapped[int] = mapped_column(Integer)
    n_items: Mapped[int] = mapped_column(Integer)
    n_stores: Mapped[int] = mapped_column(Integer)
    data_start: Mapped[date] = mapped_column(Date)
    data_end: Mapped[date] = mapped_column(Date)
    test_start: Mapped[date] = mapped_column(Date)
    champion: Mapped[str] = mapped_column(String(40))
    baseline_model: Mapped[str] = mapped_column(String(40))
    headline: Mapped[dict[str, Any]]
    settings: Mapped[dict[str, Any]]
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))


class SeriesMeta(RunScoped, Base):
    """A SKU-store demand series."""

    __tablename__ = "series"

    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    item_id: Mapped[str] = mapped_column(String(40), index=True)
    dept_id: Mapped[str] = mapped_column(String(20))
    cat_id: Mapped[str] = mapped_column(String(20), index=True)
    store_id: Mapped[str] = mapped_column(String(10), index=True)
    state_id: Mapped[str] = mapped_column(String(2))
    avg_daily_sales: Mapped[float] = mapped_column(Float)
    long_run_daily_sales: Mapped[float] = mapped_column(Float)
    avg_price: Mapped[float] = mapped_column(Float)
    demand_class: Mapped[str] = mapped_column(String(16), index=True)
    champion_mae: Mapped[float] = mapped_column(Float)


class ModelScore(RunScoped, Base):
    """Aggregate accuracy of one model on one split."""

    __tablename__ = "model_scores"

    split: Mapped[str] = mapped_column(String(12), primary_key=True)
    model: Mapped[str] = mapped_column(String(40), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    wrmsse: Mapped[float] = mapped_column(Float)
    rmsse: Mapped[float] = mapped_column(Float)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    bias: Mapped[float] = mapped_column(Float)
    coverage: Mapped[float] = mapped_column(Float)
    pinball: Mapped[float] = mapped_column(Float)
    seconds: Mapped[float] = mapped_column(Float)
    is_champion: Mapped[bool] = mapped_column(Boolean)


class DemandHistory(RunScoped, Base):
    """Recent actual demand per series (for charts and the simulator)."""

    __tablename__ = "demand_history"

    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ds: Mapped[date] = mapped_column(Date, primary_key=True)
    y: Mapped[float] = mapped_column(Float)
    sell_price: Mapped[float] = mapped_column(Float)


class ForecastRow(RunScoped, Base):
    """Test-period forecast of the champion or baseline model."""

    __tablename__ = "forecasts"

    model: Mapped[str] = mapped_column(String(40), primary_key=True)
    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ds: Mapped[date] = mapped_column(Date, primary_key=True)
    cutoff: Mapped[date] = mapped_column(Date)
    yhat: Mapped[float] = mapped_column(Float)
    yhat_lo: Mapped[float] = mapped_column(Float)
    yhat_hi: Mapped[float] = mapped_column(Float)


class ValidationForecast(RunScoped, Base):
    """Validation-fold forecast with actuals (source of safety-stock error estimates)."""

    __tablename__ = "validation_forecasts"

    model: Mapped[str] = mapped_column(String(40), primary_key=True)
    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ds: Mapped[date] = mapped_column(Date, primary_key=True)
    cutoff: Mapped[date] = mapped_column(Date)
    y: Mapped[float] = mapped_column(Float)
    yhat: Mapped[float] = mapped_column(Float)


class InventoryPolicyRow(RunScoped, Base):
    """Recommended inventory parameters for a series."""

    __tablename__ = "inventory_policies"

    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mean_daily_forecast: Mapped[float] = mapped_column(Float)
    forecast_bias: Mapped[float] = mapped_column(Float)
    sigma_protection: Mapped[float] = mapped_column(Float)
    safety_stock: Mapped[float] = mapped_column(Float)
    reorder_point: Mapped[float] = mapped_column(Float)
    order_quantity: Mapped[float] = mapped_column(Float)
    service_level: Mapped[float] = mapped_column(Float)
    lead_time_days: Mapped[int] = mapped_column(Integer)


class PortfolioResult(RunScoped, Base):
    """Portfolio-level simulation KPIs per policy."""

    __tablename__ = "portfolio_results"

    policy: Mapped[str] = mapped_column(String(40), primary_key=True)
    total_cost: Mapped[float] = mapped_column(Float)
    holding_cost: Mapped[float] = mapped_column(Float)
    ordering_cost: Mapped[float] = mapped_column(Float)
    stockout_cost: Mapped[float] = mapped_column(Float)
    fill_rate: Mapped[float] = mapped_column(Float)
    stockout_day_rate: Mapped[float] = mapped_column(Float)
    avg_inventory_value: Mapped[float] = mapped_column(Float)
    n_orders: Mapped[int] = mapped_column(Integer)
    savings_vs_baseline_pct: Mapped[float] = mapped_column(Float)


class SeriesResult(RunScoped, Base):
    """Per-series simulation KPIs per policy."""

    __tablename__ = "series_results"

    policy: Mapped[str] = mapped_column(String(40), primary_key=True)
    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    holding_cost: Mapped[float] = mapped_column(Float)
    ordering_cost: Mapped[float] = mapped_column(Float)
    stockout_cost: Mapped[float] = mapped_column(Float)
    total_cost: Mapped[float] = mapped_column(Float)
    fill_rate: Mapped[float] = mapped_column(Float)
    stockout_days: Mapped[int] = mapped_column(Integer)
    avg_on_hand: Mapped[float] = mapped_column(Float)
    n_orders: Mapped[int] = mapped_column(Integer)
    units_ordered: Mapped[float] = mapped_column(Float)
    units_demanded: Mapped[float] = mapped_column(Float)
    units_lost: Mapped[float] = mapped_column(Float)


class SensitivityPoint(RunScoped, Base):
    """Cost / service trade-off at one target service level."""

    __tablename__ = "sensitivity_points"

    service_level: Mapped[float] = mapped_column(Float, primary_key=True)
    policy: Mapped[str] = mapped_column(String(40), primary_key=True)
    total_cost: Mapped[float] = mapped_column(Float)
    holding_cost: Mapped[float] = mapped_column(Float)
    ordering_cost: Mapped[float] = mapped_column(Float)
    stockout_cost: Mapped[float] = mapped_column(Float)
    fill_rate: Mapped[float] = mapped_column(Float)
    avg_inventory_value: Mapped[float] = mapped_column(Float)


class FeatureImportanceRow(RunScoped, Base):
    """Global SHAP importance of a LightGBM feature."""

    __tablename__ = "feature_importance"

    feature: Mapped[str] = mapped_column(String(64), primary_key=True)
    mean_abs_shap: Mapped[float] = mapped_column(Float)
    gain: Mapped[float | None] = mapped_column(Float)


class ShapLocalRow(RunScoped, Base):
    """Top SHAP contributions explaining a series' forecast."""

    __tablename__ = "shap_local"

    unique_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    feature: Mapped[str] = mapped_column(String(64), primary_key=True)
    shap_value: Mapped[float] = mapped_column(Float)
    feature_value: Mapped[float | None] = mapped_column(
        Float
    )  # NaN feature (e.g. lag before launch)
