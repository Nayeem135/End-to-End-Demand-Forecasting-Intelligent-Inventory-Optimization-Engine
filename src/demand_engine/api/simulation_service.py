"""Live re-simulation for the API (what-if analysis and per-series trajectories).

Simulator inputs for a run are loaded from the database once and cached in
memory; a full portfolio re-simulation then takes well under a second.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import replace

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from demand_engine.api import repository as repo
from demand_engine.api.db import models as m
from demand_engine.api.schemas import DayState, SeriesResultOut, Trajectory, WhatIfRequest
from demand_engine.config import InventorySettings
from demand_engine.data.schema import DS, PRICE, UNIQUE_ID, Y
from demand_engine.optimization.scenario import (
    ScenarioResult,
    SimulationInputs,
    build_inputs,
    run_scenario,
)

logger = logging.getLogger(__name__)


def load_inputs(session: Session, run_id: str) -> tuple[SimulationInputs, InventorySettings]:
    """Rebuild :class:`SimulationInputs` and the run's inventory settings from the database."""
    run = repo.get_run(session, run_id)
    test_start = pd.Timestamp(run.test_start)
    hist = repo.read_frame(
        session,
        select(
            m.DemandHistory.unique_id,
            m.DemandHistory.ds,
            m.DemandHistory.y,
            m.DemandHistory.sell_price,
        ).where(m.DemandHistory.run_id == run_id),
    )
    hist[DS] = pd.to_datetime(hist[DS])
    fc = repo.read_frame(
        session,
        select(
            m.ForecastRow.model,
            m.ForecastRow.unique_id,
            m.ForecastRow.ds,
            m.ForecastRow.cutoff,
            m.ForecastRow.yhat,
        ).where(m.ForecastRow.run_id == run_id),
    )
    fc[DS] = pd.to_datetime(fc[DS])
    fc["cutoff"] = pd.to_datetime(fc["cutoff"])
    val = repo.read_frame(
        session,
        select(
            m.ValidationForecast.model,
            m.ValidationForecast.unique_id,
            m.ValidationForecast.cutoff,
            m.ValidationForecast.ds,
            m.ValidationForecast.y,
            m.ValidationForecast.yhat,
        ).where(m.ValidationForecast.run_id == run_id),
    )
    val[DS] = pd.to_datetime(val[DS])
    val["cutoff"] = pd.to_datetime(val["cutoff"])
    inputs = build_inputs(
        history=hist[hist[DS] < test_start],
        test_actuals=hist.loc[hist[DS] >= test_start, [UNIQUE_ID, DS, Y, PRICE]],
        champion_test=fc[fc["model"] == run.champion],
        baseline_test=fc[fc["model"] == run.baseline_model],
        champion_val=val[val["model"] == run.champion],
        baseline_val=val[val["model"] == run.baseline_model],
        champion=run.champion,
    )
    # Only ~16 weeks of history are stored, so take the per-series demand rates that the
    # pipeline computed from the full history (keeps API what-if identical to the pipeline).
    rates = (
        repo.read_frame(
            session,
            select(
                m.SeriesMeta.unique_id,
                m.SeriesMeta.avg_daily_sales,
                m.SeriesMeta.long_run_daily_sales,
            ).where(m.SeriesMeta.run_id == run_id),
        )
        .set_index(UNIQUE_ID)
        .reindex(inputs.unique_ids)
    )
    inputs = replace(
        inputs,
        hist_mean=rates["avg_daily_sales"].fillna(0.0).to_numpy(dtype="float64"),
        long_run_mean=rates["long_run_daily_sales"].fillna(0.0).to_numpy(dtype="float64"),
    )
    params = InventorySettings.model_validate(run.settings["inventory"])
    return inputs, params


class SimulationService:
    """Caches simulator inputs per run and evaluates scenarios on demand."""

    def __init__(self, max_runs: int = 2) -> None:
        """Create the service keeping inputs for at most ``max_runs`` runs in memory."""
        self._cache: OrderedDict[str, tuple[SimulationInputs, InventorySettings]] = OrderedDict()
        self._default_results: dict[str, ScenarioResult] = {}
        self._lock = threading.Lock()
        self._max_runs = max_runs

    def inputs(self, session: Session, run_id: str) -> tuple[SimulationInputs, InventorySettings]:
        """Cached inputs and default parameters of ``run_id``."""
        with self._lock:
            if run_id in self._cache:
                self._cache.move_to_end(run_id)
                return self._cache[run_id]
        loaded = load_inputs(session, run_id)
        with self._lock:
            self._cache[run_id] = loaded
            while len(self._cache) > self._max_runs:
                evicted, _ = self._cache.popitem(last=False)
                self._default_results.pop(evicted, None)
        logger.info(
            "Cached simulation inputs for run %s (%d series)", run_id, len(loaded[0].unique_ids)
        )
        return loaded

    def what_if(self, session: Session, run_id: str, request: WhatIfRequest) -> ScenarioResult:
        """Re-simulate the whole portfolio with some parameters overridden."""
        inputs, defaults = self.inputs(session, run_id)
        overrides = request.model_dump(exclude_none=True)
        params = InventorySettings.model_validate(defaults.model_dump() | overrides)
        return run_scenario(inputs, params)

    def trajectories(self, session: Session, run_id: str, unique_id: str) -> list[Trajectory]:
        """Daily inventory paths of all policies for one series at the run's parameters."""
        inputs, params = self.inputs(session, run_id)
        if unique_id not in inputs.unique_ids:
            raise repo.NotFoundError(f"series '{unique_id}' not found")
        result = self._default_results.get(run_id)
        if result is None:
            result = run_scenario(inputs, params)
            with self._lock:
                self._default_results[run_id] = result
        row = inputs.unique_ids.index(unique_id)
        trajectories = []
        for name, out in result.outputs.items():
            s = out.summary.iloc[row]
            days = [
                DayState(
                    ds=d.date(),
                    demand=float(inputs.demand[row, t]),
                    on_hand=float(out.on_hand[row, t]),
                    inventory_position=float(out.inventory_position[row, t]),
                    order=float(out.orders[row, t]),
                    sold=float(out.sold[row, t]),
                    lost=float(out.lost[row, t]),
                    reorder_point=float(result.policies[name].reorder_point[row, t]),
                )
                for t, d in enumerate(inputs.dates)
            ]
            summary = SeriesResultOut(
                policy=name,
                total_cost=float(s["total_cost"]),
                holding_cost=float(s["holding_cost"]),
                ordering_cost=float(s["ordering_cost"]),
                stockout_cost=float(s["stockout_cost"]),
                fill_rate=float(s["fill_rate"]),
                stockout_days=int(s["stockout_days"]),
                avg_on_hand=float(s["avg_on_hand"]),
                n_orders=int(s["n_orders"]),
            )
            trajectories.append(Trajectory(policy=name, days=days, summary=summary))
        return trajectories
