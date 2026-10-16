"""Rolling-origin backtesting for any :class:`~demand_engine.models.base.Forecaster`."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from demand_engine.data.schema import DS, UNIQUE_ID, Y
from demand_engine.data.splits import split_at
from demand_engine.evaluation.metrics import per_series_metrics, revenue_weights, summarize
from demand_engine.models.base import Forecaster
from demand_engine.models.registry import ForecasterFactory

FitHook = Callable[[pd.Timestamp, Forecaster], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Forecasts and accuracy of one model over a set of cut-offs."""

    model: str
    predictions: pd.DataFrame
    per_series: pd.DataFrame
    summary: dict[str, float]
    seconds: float


def rolling_forecasts(
    df: pd.DataFrame,
    factory: ForecasterFactory,
    cutoffs: Sequence[pd.Timestamp],
    horizon: int,
    on_fit: FitHook | None = None,
) -> pd.DataFrame:
    """Refit a fresh model at each cut-off and forecast the following ``horizon`` days.

    Args:
        df: Full long demand table.
        factory: Produces a fresh, unfitted model.
        cutoffs: Chronological forecast origins.
        horizon: Days forecast after each origin.
        on_fit: Optional callback receiving each fitted model (e.g. to keep it for SHAP).

    Returns:
        Concatenated forecasts with ``cutoff`` and actual ``y`` columns attached.
    """
    frames = []
    for cutoff in cutoffs:
        train, window = split_at(df, cutoff, horizon)
        if window.empty:
            raise ValueError(f"no data after cut-off {cutoff.date()}")
        model = factory()
        model.fit(train)
        if on_fit is not None:
            on_fit(cutoff, model)
        fc = model.predict(window)
        fc = fc.merge(
            window[[UNIQUE_ID, DS, Y]], on=[UNIQUE_ID, DS], how="left", validate="one_to_one"
        )
        fc["cutoff"] = cutoff
        frames.append(fc)
    return pd.concat(frames, ignore_index=True)


def backtest(
    df: pd.DataFrame,
    name: str,
    factory: ForecasterFactory,
    cutoffs: Sequence[pd.Timestamp],
    horizon: int,
    level: int,
    on_fit: FitHook | None = None,
) -> BacktestResult:
    """Run rolling forecasts and score them.

    Args:
        df: Full long demand table.
        name: Model name for reporting.
        factory: Produces a fresh, unfitted model.
        cutoffs: Chronological forecast origins.
        horizon: Days forecast after each origin.
        level: Prediction-interval level in percent.
        on_fit: Optional callback receiving each fitted model.
    """
    start = time.perf_counter()
    preds = rolling_forecasts(df, factory, cutoffs, horizon, on_fit)
    preds["model"] = name
    history = df[df[DS] <= min(cutoffs)]
    per_series = per_series_metrics(preds, history, level)
    summary = summarize(per_series, revenue_weights(history))
    seconds = time.perf_counter() - start
    logger.info(
        "Backtest %-15s WRMSSE=%.4f MAE=%.3f (%.1fs)",
        name,
        summary["wrmsse"],
        summary["mae"],
        seconds,
    )
    return BacktestResult(name, preds, per_series, summary, seconds)


def leaderboard(results: Sequence[BacktestResult]) -> pd.DataFrame:
    """Rank models by WRMSSE (lower is better)."""
    rows = [{"model": r.model, **r.summary, "seconds": r.seconds} for r in results]
    board = pd.DataFrame(rows).sort_values("wrmsse", ignore_index=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    return board
