"""Pydantic response / request models (the public API contract)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    """Base for schemas populated from ORM objects."""

    model_config = ConfigDict(from_attributes=True)


class Health(BaseModel):
    """Liveness / readiness information."""

    status: Literal["ok", "degraded"]
    database: bool
    latest_run: str | None
    version: str


class Headline(BaseModel):
    """Key results shown on the dashboard overview."""

    champion_test_wrmsse: float
    baseline_test_wrmsse: float
    champion_val_wrmsse: float
    savings_vs_rule_pct: float
    savings_vs_ma_pct: float
    fill_rate_forecast_driven: float
    fill_rate_rule_of_thumb: float


class InventoryParams(BaseModel):
    """Inventory economics / policy parameters of a simulation."""

    service_level: float
    lead_time_days: int
    review_period_days: int
    ordering_cost: float
    holding_cost_rate: float
    stockout_cost_multiplier: float


class RunSummary(ORMModel):
    """A published pipeline run."""

    run_id: str
    created_at: datetime
    state: str
    n_series: int
    n_items: int
    n_stores: int
    data_start: date
    data_end: date
    test_start: date
    champion: str
    baseline_model: str
    headline: Headline
    inventory: InventoryParams
    mlflow_run_id: str | None


class ModelScoreOut(ORMModel):
    """Leaderboard row."""

    split: str
    model: str
    rank: int
    wrmsse: float
    rmsse: float
    mae: float
    rmse: float
    bias: float
    coverage: float
    pinball: float
    seconds: float
    is_champion: bool


class SeriesOut(ORMModel):
    """Series catalogue entry."""

    unique_id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    avg_daily_sales: float
    avg_price: float
    demand_class: str
    champion_mae: float


class SeriesPage(BaseModel):
    """Paginated series list."""

    items: list[SeriesOut]
    total: int
    page: int
    page_size: int


class PolicyOut(ORMModel):
    """Recommended inventory parameters for a series."""

    unique_id: str
    mean_daily_forecast: float
    forecast_bias: float
    sigma_protection: float
    safety_stock: float
    reorder_point: float
    order_quantity: float
    service_level: float
    lead_time_days: int


class PolicyRow(PolicyOut):
    """Policy joined with series metadata (inventory table)."""

    item_id: str
    store_id: str
    cat_id: str
    demand_class: str
    avg_price: float


class PolicyPage(BaseModel):
    """Paginated inventory recommendations."""

    items: list[PolicyRow]
    total: int
    page: int
    page_size: int


class SeriesResultOut(ORMModel):
    """Simulation KPIs of one policy for one series."""

    policy: str
    total_cost: float
    holding_cost: float
    ordering_cost: float
    stockout_cost: float
    fill_rate: float
    stockout_days: int
    avg_on_hand: float
    n_orders: int


class SeriesDetail(BaseModel):
    """Everything about one series on one screen."""

    series: SeriesOut
    policy: PolicyOut
    results: list[SeriesResultOut]


class HistoryPoint(BaseModel):
    """Actual demand on a day."""

    ds: date
    y: float
    sell_price: float


class ForecastPoint(BaseModel):
    """Forecast with interval on a day."""

    ds: date
    yhat: float
    yhat_lo: float
    yhat_hi: float


class ForecastResponse(BaseModel):
    """History plus test-period forecasts for the champion and baseline models."""

    unique_id: str
    champion: str
    baseline_model: str
    test_start: date
    history: list[HistoryPoint]
    forecasts: dict[str, list[ForecastPoint]]


class DayState(BaseModel):
    """Inventory state at the end of a simulated day."""

    ds: date
    demand: float
    on_hand: float
    inventory_position: float
    order: float
    sold: float
    lost: float
    reorder_point: float


class Trajectory(BaseModel):
    """Simulated inventory path of one policy for one series."""

    policy: str
    days: list[DayState]
    summary: SeriesResultOut


class SeriesSimulation(BaseModel):
    """Trajectories of all policies for one series."""

    unique_id: str
    trajectories: list[Trajectory]


class PortfolioOut(ORMModel):
    """Portfolio KPIs of one policy."""

    policy: str
    total_cost: float
    holding_cost: float
    ordering_cost: float
    stockout_cost: float
    fill_rate: float
    stockout_day_rate: float
    avg_inventory_value: float
    n_orders: int
    savings_vs_baseline_pct: float


class SensitivityOut(ORMModel):
    """One point of the cost / service trade-off curve."""

    service_level: float
    policy: str
    total_cost: float
    holding_cost: float
    ordering_cost: float
    stockout_cost: float
    fill_rate: float
    avg_inventory_value: float


class WhatIfRequest(BaseModel):
    """Inventory parameters to re-simulate; omitted fields keep the run's values."""

    service_level: float | None = Field(default=None, gt=0.5, lt=1.0)
    lead_time_days: int | None = Field(default=None, ge=1, le=60)
    review_period_days: int | None = Field(default=None, ge=1, le=28)
    ordering_cost: float | None = Field(default=None, ge=0, le=10_000)
    holding_cost_rate: float | None = Field(default=None, gt=0, le=5)
    stockout_cost_multiplier: float | None = Field(default=None, ge=0, le=100)


class WhatIfResponse(BaseModel):
    """Re-simulated portfolio KPIs."""

    params: InventoryParams
    portfolio: list[PortfolioOut]
    elapsed_ms: float


class FeatureImportanceOut(ORMModel):
    """Global SHAP importance."""

    feature: str
    mean_abs_shap: float
    gain: float | None


class ShapLocalOut(ORMModel):
    """Top SHAP contribution for a series (log-demand scale)."""

    feature: str
    shap_value: float
    feature_value: float | None


class Filters(BaseModel):
    """Distinct filter values for the dashboard."""

    stores: list[str]
    categories: list[str]
    departments: list[str]
    demand_classes: list[str]
