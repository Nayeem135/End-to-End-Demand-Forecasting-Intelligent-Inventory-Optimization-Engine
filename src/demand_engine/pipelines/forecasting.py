"""Forecasting stage: backtest all candidates, select the champion, forecast the test period.

Model selection uses *validation folds only*. Test-window forecasts for every
model are produced afterwards purely for honest reporting and the simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demand_engine.config import ModelName, Settings
from demand_engine.data.schema import DS, PRICE, UNIQUE_ID, Y
from demand_engine.data.splits import SplitPlan, make_split_plan
from demand_engine.data.transform import classify_intermittency
from demand_engine.evaluation.backtest import (
    BacktestResult,
    backtest,
    leaderboard,
    leaderboard_by_class,
)
from demand_engine.evaluation.explain import Explanation, explain
from demand_engine.evaluation.metrics import summarize_by
from demand_engine.models.base import Forecaster
from demand_engine.models.lgbm import LightGBMForecaster
from demand_engine.models.registry import model_factory
from demand_engine.pipelines.tracking import Tracker

logger = logging.getLogger(__name__)


@dataclass
class ForecastingOutput:
    """Everything produced by the forecasting stage."""

    plan: SplitPlan
    validation: dict[str, BacktestResult]
    test: dict[str, BacktestResult]
    champion: str
    baseline: str
    leaderboard_validation: pd.DataFrame
    leaderboard_test: pd.DataFrame
    leaderboard_by_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    explanation: Explanation | None = None
    feature_gain: pd.DataFrame | None = None
    mlflow_run_ids: dict[str, str] = field(default_factory=dict)


def demand_classes(df: pd.DataFrame, before: pd.Timestamp) -> pd.Series:
    """Syntetos-Boylan class per series from history up to ``before`` (no evaluation data)."""
    history = df[df[DS] <= before]
    return history.groupby(UNIQUE_ID)[Y].agg(
        lambda s: classify_intermittency(s.to_numpy(dtype="float64"))
    )


def _log_model_run(
    tracker: Tracker,
    params: dict[str, object],
    validation: BacktestResult,
    test: BacktestResult,
    classes: pd.Series,
) -> None:
    """Log one model's parameters and every metric (overall, per fold, per demand class)."""
    tracker.log_params(params)
    tracker.log_metrics(validation.summary, prefix="val_")
    tracker.log_metrics(test.summary, prefix="test_")
    tracker.log_metrics({"seconds": validation.seconds + test.seconds})
    for split, result in (("val", validation), ("test", test)):
        for step, fold in enumerate(result.fold_summaries):
            tracker.log_metrics(fold, prefix=f"{split}_fold_", step=step)
        per_class = summarize_by(result.per_series, result.weights, classes, name="demand_class")
        for cls, metrics in per_class.set_index("demand_class").drop(columns="n_series").iterrows():
            tracker.log_metrics(
                {str(k): float(v) for k, v in metrics.items()}, prefix=f"{split}_{cls}_"
            )


def models_to_run(settings: Settings) -> list[ModelName]:
    """Configured candidates plus the inventory baseline model (deduplicated, ordered)."""
    names = list(dict.fromkeys([*settings.model.candidates, settings.model.baseline]))
    return names


def run_forecasting(df: pd.DataFrame, settings: Settings, tracker: Tracker) -> ForecastingOutput:
    """Execute validation backtests, champion selection, test forecasts and SHAP.

    Args:
        df: Validated long demand table.
        settings: Global settings.
        tracker: Experiment tracker (MLflow or no-op).
    """
    bt, level = settings.backtest, settings.model.interval_level
    plan = make_split_plan(df[DS].max(), bt.horizon, bt.n_folds, bt.n_test_windows)
    logger.info(
        "Validation cut-offs %s | test cut-offs %s",
        [c.date().isoformat() for c in plan.validation_cutoffs],
        [c.date().isoformat() for c in plan.test_cutoffs],
    )
    validation: dict[str, BacktestResult] = {}
    test: dict[str, BacktestResult] = {}
    run_ids: dict[str, str] = {}
    last_lgbm: list[tuple[pd.Timestamp, Forecaster]] = []

    def keep_lgbm(cutoff: pd.Timestamp, model: Forecaster) -> None:
        last_lgbm[:] = [(cutoff, model)]

    classes = demand_classes(df, plan.first_cutoff)
    for name in models_to_run(settings):
        factory = model_factory(name, settings)
        with tracker.run(name, nested=True) as run_id:
            validation[name] = backtest(
                df, name, factory, plan.validation_cutoffs, bt.horizon, level
            )
            hook = keep_lgbm if name == "lightgbm" else None
            test[name] = backtest(
                df, name, factory, plan.test_cutoffs, bt.horizon, level, on_fit=hook
            )
            _log_model_run(tracker, factory().params(), validation[name], test[name], classes)
            if run_id:
                run_ids[name] = run_id

    board_val = leaderboard([validation[n] for n in settings.model.candidates])
    board_test = leaderboard([test[n] for n in settings.model.candidates])
    by_class = pd.concat(
        [
            leaderboard_by_class(
                [validation[n] for n in settings.model.candidates], classes
            ).assign(split="validation"),
            leaderboard_by_class([test[n] for n in settings.model.candidates], classes).assign(
                split="test"
            ),
        ],
        ignore_index=True,
    )
    champion = str(board_val["model"].iloc[0])
    logger.info("Champion (validation WRMSSE): %s", champion)

    output = ForecastingOutput(
        plan=plan,
        validation=validation,
        test=test,
        champion=champion,
        baseline=settings.model.baseline,
        leaderboard_validation=board_val,
        leaderboard_test=board_test,
        leaderboard_by_class=by_class,
        mlflow_run_ids=run_ids,
    )
    if last_lgbm:
        cutoff, model = last_lgbm[0]
        assert isinstance(model, LightGBMForecaster)
        output.explanation = _explain_last_window(df, model, cutoff, bt.horizon, settings.seed)
        output.feature_gain = model.feature_importance()
    return output


def _explain_last_window(
    df: pd.DataFrame, model: LightGBMForecaster, cutoff: pd.Timestamp, horizon: int, seed: int
) -> Explanation:
    end = cutoff + pd.Timedelta(days=horizon)
    frame = df[df[DS] <= end].copy()
    frame.loc[frame[DS] > cutoff, Y] = np.nan
    frame = frame.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)
    rows = frame.loc[frame[DS] > cutoff, [UNIQUE_ID, DS]]
    return explain(model, frame, rows, seed=seed)


def series_catalogue(
    df: pd.DataFrame, plan: SplitPlan, champion_test: pd.DataFrame
) -> pd.DataFrame:
    """Per-series metadata and headline stats for the API and dashboard."""
    history = df[df[DS] <= plan.test_cutoffs[0]]
    recent = history[history[DS] > history[DS].max() - pd.Timedelta(days=28)]
    meta = df.groupby(UNIQUE_ID)[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].first()
    year = history[history[DS] > history[DS].max() - pd.Timedelta(days=364)]
    meta["avg_daily_sales"] = recent.groupby(UNIQUE_ID)[Y].mean()
    meta["long_run_daily_sales"] = year.groupby(UNIQUE_ID)[Y].mean()
    meta["avg_price"] = recent.groupby(UNIQUE_ID)[PRICE].mean()
    meta["demand_class"] = history.groupby(UNIQUE_ID)[Y].agg(
        lambda s: classify_intermittency(s.to_numpy(dtype="float64"))
    )
    err = champion_test.assign(abs_err=(champion_test[Y] - champion_test["yhat"]).abs())
    meta["champion_mae"] = err.groupby(UNIQUE_ID)["abs_err"].mean()
    return meta.reset_index()
