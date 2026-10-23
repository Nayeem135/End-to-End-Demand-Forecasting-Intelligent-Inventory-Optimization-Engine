"""Run the policy comparison for one set of inventory parameters (also powers what-if)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from demand_engine.config import InventorySettings
from demand_engine.data.schema import DS, PRICE, UNIQUE_ID, Y
from demand_engine.optimization.formulas import daily_holding_cost, protection_days, safety_stock
from demand_engine.optimization.policies import (
    forecast_policy,
    order_quantity,
    rule_of_thumb_policy,
    window_mean,
)
from demand_engine.optimization.simulator import CostModel, Policy, SimulationOutput, simulate
from demand_engine.optimization.uncertainty import protection_error_stats
from demand_engine.types import FloatArray, IntArray, cell

logger = logging.getLogger(__name__)

BASELINE_POLICY = "rule_of_thumb"
POLICY_NAMES = ("rule_of_thumb", "ma_safety_stock", "forecast_driven")


@dataclass(frozen=True)
class SimulationInputs:
    """Everything the simulator needs, aligned on ``unique_ids`` x ``dates``.

    Attributes:
        unique_ids: Series keys (row order of all arrays).
        dates: Simulated days.
        demand: Actual demand in the test period ``(n, T)``.
        price: Average selling price per series ``(n,)``.
        hist_mean: Mean daily demand over the 28 days before the test period ``(n,)``
            (sets the common starting stock).
        long_run_mean: Mean daily demand over the last year before the test period ``(n,)``
            (drives the shared EOQ lot size).
        window_id: Forecast window (cut-off index) of each simulated day ``(T,)``.
        champion_forecast: Champion model test forecasts ``(n, T)``.
        baseline_forecast: Moving-average test forecasts ``(n, T)``.
        champion_errors: Champion validation forecasts (for error-based safety stock).
        baseline_errors: Baseline-model validation forecasts.
        champion: Champion model name.
    """

    unique_ids: list[str]
    dates: pd.DatetimeIndex
    demand: FloatArray
    price: FloatArray
    hist_mean: FloatArray
    long_run_mean: FloatArray
    window_id: IntArray
    champion_forecast: FloatArray
    baseline_forecast: FloatArray
    champion_errors: pd.DataFrame
    baseline_errors: pd.DataFrame
    champion: str


@dataclass(frozen=True)
class ScenarioResult:
    """Simulation outputs for all policies plus portfolio KPIs and per-series policy table."""

    params: InventorySettings
    outputs: dict[str, SimulationOutput]
    policies: dict[str, Policy]
    portfolio: pd.DataFrame
    policy_table: pd.DataFrame


def to_matrix(
    df: pd.DataFrame, value: str, unique_ids: list[str], dates: pd.DatetimeIndex
) -> FloatArray:
    """Pivot a long table to an ``(n, T)`` array in the given order; raises on gaps."""
    wide = df.pivot(index=UNIQUE_ID, columns=DS, values=value).reindex(
        index=unique_ids, columns=dates
    )
    if wide.isna().any().any():
        raise ValueError(f"'{value}' is missing for some series/dates")
    return wide.to_numpy(dtype="float64")


def build_inputs(
    history: pd.DataFrame,
    test_actuals: pd.DataFrame,
    champion_test: pd.DataFrame,
    baseline_test: pd.DataFrame,
    champion_val: pd.DataFrame,
    baseline_val: pd.DataFrame,
    champion: str,
) -> SimulationInputs:
    """Align all tables into :class:`SimulationInputs`.

    Args:
        history: Demand before the test period (starting stock and long-run EOQ rate).
        test_actuals: Long table of the test period (``y`` and ``sell_price``).
        champion_test: Champion forecasts for the test period (with ``cutoff``).
        baseline_test: Baseline-model forecasts for the test period.
        champion_val: Champion validation-fold forecasts (with ``y``).
        baseline_val: Baseline validation-fold forecasts (with ``y``).
        champion: Champion model name.
    """
    unique_ids = sorted(test_actuals[UNIQUE_ID].unique().tolist())
    dates = pd.DatetimeIndex(sorted(test_actuals[DS].unique()))
    last = history[DS].max()
    recent = history[history[DS] > last - pd.Timedelta(days=28)]
    hist_mean = recent.groupby(UNIQUE_ID)[Y].mean().reindex(unique_ids).fillna(0.0)
    year = history[history[DS] > last - pd.Timedelta(days=364)]
    long_run = year.groupby(UNIQUE_ID)[Y].mean().reindex(unique_ids).fillna(0.0)
    price = test_actuals.groupby(UNIQUE_ID)[PRICE].mean().reindex(unique_ids)
    return SimulationInputs(
        window_id=window_ids(champion_test, dates),
        unique_ids=unique_ids,
        dates=dates,
        demand=to_matrix(test_actuals, Y, unique_ids, dates),
        price=price.to_numpy(dtype="float64"),
        hist_mean=hist_mean.to_numpy(dtype="float64"),
        long_run_mean=long_run.to_numpy(dtype="float64"),
        champion_forecast=to_matrix(champion_test, "yhat", unique_ids, dates),
        baseline_forecast=to_matrix(baseline_test, "yhat", unique_ids, dates),
        champion_errors=champion_val,
        baseline_errors=baseline_val,
        champion=champion,
    )


def inputs_from_stored(
    *,
    test_start: pd.Timestamp,
    champion: str,
    baseline: str,
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    validation: pd.DataFrame,
    series: pd.DataFrame,
) -> SimulationInputs:
    """Rebuild simulator inputs from a run's *stored* tables (database or Parquet artifacts).

    Only ~16 weeks of history are stored, so the per-series demand rates the pipeline
    computed from the full history (``avg_daily_sales``, ``long_run_daily_sales``) replace
    the ones :func:`build_inputs` would derive. This keeps every consumer (API what-if,
    Streamlit app) numerically identical to the pipeline.

    Args:
        test_start: First day of the test period.
        champion: Champion model name.
        baseline: Baseline model name (moving average by default).
        history: ``unique_id, ds, y, sell_price`` (recent history + test actuals).
        forecasts: Test forecasts with ``model, unique_id, ds, cutoff, yhat``.
        validation: Validation forecasts with ``model, unique_id, cutoff, ds, y, yhat``.
        series: Catalogue with ``unique_id, avg_daily_sales, long_run_daily_sales``.
    """
    hist = history.assign(**{DS: pd.to_datetime(history[DS])})
    fc = forecasts.assign(
        **{DS: pd.to_datetime(forecasts[DS]), "cutoff": pd.to_datetime(forecasts["cutoff"])}
    )
    val = validation.assign(
        **{DS: pd.to_datetime(validation[DS]), "cutoff": pd.to_datetime(validation["cutoff"])}
    )
    inputs = build_inputs(
        history=hist[hist[DS] < test_start],
        test_actuals=hist.loc[hist[DS] >= test_start, [UNIQUE_ID, DS, Y, PRICE]],
        champion_test=fc[fc["model"] == champion],
        baseline_test=fc[fc["model"] == baseline],
        champion_val=val[val["model"] == champion],
        baseline_val=val[val["model"] == baseline],
        champion=champion,
    )
    rates = series.set_index(UNIQUE_ID).reindex(inputs.unique_ids)
    return replace(
        inputs,
        hist_mean=rates["avg_daily_sales"].fillna(0.0).to_numpy(dtype="float64"),
        long_run_mean=rates["long_run_daily_sales"].fillna(0.0).to_numpy(dtype="float64"),
    )


def window_ids(forecasts: pd.DataFrame, dates: pd.DatetimeIndex) -> IntArray:
    """Index of the forecast cut-off that produced each date (0 = first window).

    Raises:
        ValueError: If a date is covered by no or by several cut-offs.
    """
    per_day = forecasts[[DS, "cutoff"]].drop_duplicates()
    if per_day[DS].duplicated().any():
        raise ValueError("a test date is forecast from more than one cut-off")
    rank = {c: i for i, c in enumerate(sorted(per_day["cutoff"].unique()))}
    mapping = per_day.set_index(DS)["cutoff"].map(rank).reindex(dates)
    if mapping.isna().any():
        raise ValueError("some test dates have no forecast cut-off")
    return mapping.to_numpy(dtype="int64")


def error_stats(
    errors: pd.DataFrame, days: int, unique_ids: list[str]
) -> tuple[FloatArray, FloatArray]:
    """``(bias, sigma)`` arrays aligned to ``unique_ids``; unseen series get the median."""
    stats = protection_error_stats(errors, days).reindex(unique_ids)
    filled = stats.fillna(stats.median()).fillna(0.0)
    return filled["bias"].to_numpy(dtype="float64"), filled["sigma"].to_numpy(dtype="float64")


def run_scenario(inputs: SimulationInputs, params: InventorySettings) -> ScenarioResult:
    """Build all three policies for ``params`` and simulate them on actual demand."""
    p_days = protection_days(params.lead_time_days, params.review_period_days)
    annual_holding = inputs.price * params.holding_cost_rate
    costs = CostModel(
        holding_per_unit_day=daily_holding_cost(inputs.price, params.holding_cost_rate),
        stockout_per_unit=inputs.price * params.stockout_cost_multiplier,
        ordering_cost=params.ordering_cost,
    )
    bias_champion, sigma_champion = error_stats(inputs.champion_errors, p_days, inputs.unique_ids)
    bias_baseline, sigma_baseline = error_stats(inputs.baseline_errors, p_days, inputs.unique_ids)
    lot_size = order_quantity(inputs.long_run_mean, params.ordering_cost, annual_holding)

    def sq_policy(name: str, forecast: FloatArray, bias: FloatArray, sigma: FloatArray) -> Policy:
        return forecast_policy(
            name, forecast, inputs.window_id, bias, sigma, p_days, params.service_level, lot_size
        )

    policies = {
        "rule_of_thumb": rule_of_thumb_policy(
            window_mean(inputs.baseline_forecast, inputs.window_id),
            params.lead_time_days,
            params.baseline_cover_days,
        ),
        "ma_safety_stock": sq_policy(
            "ma_safety_stock", inputs.baseline_forecast, bias_baseline, sigma_baseline
        ),
        "forecast_driven": sq_policy(
            "forecast_driven", inputs.champion_forecast, bias_champion, sigma_champion
        ),
    }
    initial = np.round(inputs.hist_mean * params.initial_cover_days)
    outputs = {
        name: simulate(
            inputs.demand,
            policy,
            costs,
            params.lead_time_days,
            params.review_period_days,
            initial,
            inputs.unique_ids,
        )
        for name, policy in policies.items()
    }
    portfolio = portfolio_summary(outputs, inputs.price)
    table = policy_table(inputs, policies["forecast_driven"], bias_champion, sigma_champion, params)
    return ScenarioResult(params, outputs, policies, portfolio, table)


def portfolio_summary(outputs: dict[str, SimulationOutput], price: FloatArray) -> pd.DataFrame:
    """Aggregate KPIs per policy and savings relative to :data:`BASELINE_POLICY`."""
    rows = []
    for name, out in outputs.items():
        s = out.summary
        demanded = float(s["units_demanded"].sum())
        rows.append(
            {
                "policy": name,
                "total_cost": float(s["total_cost"].sum()),
                "holding_cost": float(s["holding_cost"].sum()),
                "ordering_cost": float(s["ordering_cost"].sum()),
                "stockout_cost": float(s["stockout_cost"].sum()),
                "fill_rate": 1.0 - float(s["units_lost"].sum()) / demanded if demanded > 0 else 1.0,
                "stockout_day_rate": float(s["stockout_days"].sum()) / out.on_hand.size,
                "avg_inventory_value": float((out.on_hand.mean(axis=1) * price).sum()),
                "n_orders": int(s["n_orders"].sum()),
            }
        )
    df = pd.DataFrame(rows).set_index("policy")
    base = cell(df, BASELINE_POLICY, "total_cost")
    df["savings_vs_baseline_pct"] = 100.0 * (base - df["total_cost"]) / base if base > 0 else 0.0
    return df.reset_index()


def policy_table(
    inputs: SimulationInputs,
    policy: Policy,
    bias: FloatArray,
    sigma: FloatArray,
    params: InventorySettings,
) -> pd.DataFrame:
    """Per-series recommendation for the first test day (what a planner would see).

    ``safety_stock`` is the full buffer above the forecast: bias correction + ``z * sigma``.
    """
    return pd.DataFrame(
        {
            UNIQUE_ID: inputs.unique_ids,
            "mean_daily_forecast": inputs.champion_forecast.mean(axis=1),
            "forecast_bias": bias,
            "sigma_protection": sigma,
            "safety_stock": bias + safety_stock(sigma, params.service_level),
            "reorder_point": policy.reorder_point[:, 0],
            "order_quantity": policy.quantity[:, 0],
            "service_level": params.service_level,
            "lead_time_days": params.lead_time_days,
        }
    )


def sensitivity(
    inputs: SimulationInputs, params: InventorySettings, service_levels: list[float]
) -> pd.DataFrame:
    """Cost / fill-rate trade-off curve across target service levels for every policy."""
    frames = []
    for level in service_levels:
        result = run_scenario(inputs, replace_params(params, service_level=level))
        frames.append(result.portfolio.assign(service_level=level))
    return pd.concat(frames, ignore_index=True)


def replace_params(params: InventorySettings, **changes: float | int) -> InventorySettings:
    """Return validated parameters with some fields changed."""
    merged = params.model_dump() | changes
    return InventorySettings.model_validate(merged)
