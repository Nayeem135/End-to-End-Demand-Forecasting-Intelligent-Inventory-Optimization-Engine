"""Optuna hyper-parameter tuning for the global LightGBM forecaster (small trial budget).

Protocol (no test-set leakage):

* the model is trained on all data up to the **earliest validation cut-off** and scored on
  the following ``horizon`` days by WRMSSE — the later validation folds and the test
  windows are never used for tuning;
* trial 0 is the current configuration, so the best trial can only match or improve it;
* the TPE sampler is seeded, so the same data and budget give the same trials;
* only the point model is trained during tuning (quantile models do not affect WRMSSE).

Every trial is logged to MLflow (parameters + score) when a tracker is supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from demand_engine.config import LightGBMSettings, Settings
from demand_engine.data.schema import DS, UNIQUE_ID, Y
from demand_engine.data.splits import make_split_plan, split_at
from demand_engine.evaluation.metrics import per_series_metrics, revenue_weights, summarize
from demand_engine.models.lgbm import LightGBMForecaster
from demand_engine.pipelines.tracking import Tracker

logger = logging.getLogger(__name__)

TUNED_FIELDS = (
    "learning_rate",
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l2",
    "tweedie_variance_power",
    "n_estimators",
)


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a tuning study."""

    baseline_wrmsse: float
    best_wrmsse: float
    best_params: dict[str, Any]
    trials: pd.DataFrame

    @property
    def improvement_pct(self) -> float:
        """Relative WRMSSE improvement of the best trial over the current configuration."""
        return 100.0 * (self.baseline_wrmsse - self.best_wrmsse) / self.baseline_wrmsse


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space (log scales where the effect is multiplicative)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.05, 1.6),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
    }


def score_config(
    df: pd.DataFrame, settings: Settings, lgbm: LightGBMSettings, cutoff: pd.Timestamp
) -> float:
    """WRMSSE of a LightGBM configuration trained up to ``cutoff`` (point model only)."""
    horizon = settings.backtest.horizon
    train, window = split_at(df, cutoff, horizon)
    model = LightGBMForecaster(lgbm, horizon=horizon, seed=settings.seed, quantile_models=False)
    preds = model.fit(train).predict(window)
    preds = preds.merge(window[[UNIQUE_ID, DS, Y]], on=[UNIQUE_ID, DS], validate="one_to_one")
    history = df[df[DS] <= cutoff]
    per_series = per_series_metrics(preds, history, settings.model.interval_level)
    return summarize(per_series, revenue_weights(history))["wrmsse"]


def tune_lightgbm(
    df: pd.DataFrame, settings: Settings, n_trials: int, tracker: Tracker | None = None
) -> TuningResult:
    """Run a seeded Optuna study and return the best LightGBM parameters.

    Args:
        df: Validated long demand table.
        settings: Global settings (horizon, folds, seed, current LightGBM config).
        n_trials: Trial budget (>= 1); trial 0 is the current configuration.
        tracker: Optional MLflow tracker; each trial is logged as a nested run.

    Raises:
        ValueError: If ``n_trials`` < 1.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    bt = settings.backtest
    plan = make_split_plan(df[DS].max(), bt.horizon, bt.n_folds, bt.n_test_windows)
    cutoff = plan.validation_cutoffs[0]
    current = settings.model.lgbm
    tracker = tracker or Tracker("", "", settings.paths.artifacts_dir)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        candidate = current.model_copy(update=params)
        with tracker.run(f"trial-{trial.number:03d}", nested=True):
            value = score_config(df, settings, candidate, cutoff)
            tracker.log_params({"trial": trial.number, **params})
            tracker.log_metrics({"val_fold1_wrmsse": value})
        logger.info("Trial %d: WRMSSE %.4f %s", trial.number, value, params)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=settings.seed)
    )
    study.enqueue_trial({f: getattr(current, f) for f in TUNED_FIELDS})
    with tracker.run("tune-lightgbm"):
        tracker.log_params({"n_trials": n_trials, "tuning_cutoff": cutoff.date().isoformat()})
        study.optimize(objective, n_trials=n_trials)
        tracker.log_metrics(
            {"best_wrmsse": study.best_value, "baseline_wrmsse": study.trials[0].value or 0.0}
        )
    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    baseline = study.trials[0].value
    assert baseline is not None
    return TuningResult(
        baseline_wrmsse=float(baseline),
        best_wrmsse=float(study.best_value),
        best_params=dict(study.best_params),
        trials=trials,
    )
