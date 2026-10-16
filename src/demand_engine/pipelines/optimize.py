"""Optimisation stage: inventory policies + simulation on the held-out test period."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from demand_engine.config import Settings
from demand_engine.data.schema import DS
from demand_engine.optimization.scenario import (
    ScenarioResult,
    SimulationInputs,
    build_inputs,
    run_scenario,
    sensitivity,
)
from demand_engine.pipelines.forecasting import ForecastingOutput
from demand_engine.types import cell

logger = logging.getLogger(__name__)

SERVICE_LEVEL_GRID = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]


@dataclass(frozen=True)
class OptimizationOutput:
    """Simulation results at the configured parameters plus the sensitivity sweep."""

    inputs: SimulationInputs
    result: ScenarioResult
    sensitivity: pd.DataFrame


def simulation_inputs(df: pd.DataFrame, fc: ForecastingOutput) -> SimulationInputs:
    """Assemble simulator inputs from the demand table and forecasting results."""
    test_start = fc.plan.test_cutoffs[0]
    return build_inputs(
        history=df[df[DS] <= test_start],
        test_actuals=df[df[DS] > test_start],
        champion_test=fc.test[fc.champion].predictions,
        baseline_test=fc.test[fc.baseline].predictions,
        champion_val=fc.validation[fc.champion].predictions,
        baseline_val=fc.validation[fc.baseline].predictions,
        champion=fc.champion,
    )


def run_optimization(
    df: pd.DataFrame, fc: ForecastingOutput, settings: Settings
) -> OptimizationOutput:
    """Simulate all policies and the service-level sensitivity curve."""
    inputs = simulation_inputs(df, fc)
    result = run_scenario(inputs, settings.inventory)
    port = result.portfolio.set_index("policy")
    logger.info(
        "Simulation (%d series x %d days): forecast-driven saves %.1f%% vs rule-of-thumb, "
        "fill rate %.3f",
        len(inputs.unique_ids),
        len(inputs.dates),
        cell(port, "forecast_driven", "savings_vs_baseline_pct"),
        cell(port, "forecast_driven", "fill_rate"),
    )
    sens = sensitivity(inputs, settings.inventory, SERVICE_LEVEL_GRID)
    return OptimizationOutput(inputs=inputs, result=result, sensitivity=sens)
