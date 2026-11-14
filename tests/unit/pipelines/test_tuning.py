import mlflow
import pandas as pd
import pytest

from demand_engine.config import Settings
from demand_engine.pipelines.tracking import Tracker
from demand_engine.pipelines.tuning import TUNED_FIELDS, tune_lightgbm


@pytest.fixture
def test_settings(test_settings: Settings) -> Settings:
    """Test settings with a LightGBM config inside the search space (trial 0 = this config)."""
    lgbm = test_settings.model.lgbm.model_copy(update={"n_estimators": 200, "min_data_in_leaf": 20})
    model = test_settings.model.model_copy(update={"lgbm": lgbm})
    return test_settings.model_copy(update={"model": model})


@pytest.fixture
def tuning_df(long_df: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(long_df["unique_id"].unique())[:8]
    return long_df[long_df["unique_id"].isin(ids)].reset_index(drop=True)


def test_trial_zero_is_current_config_and_best_never_worse(
    tuning_df: pd.DataFrame, test_settings: Settings
) -> None:
    result = tune_lightgbm(tuning_df, test_settings, n_trials=3)
    assert len(result.trials) == 3
    first = result.trials.sort_values("number").iloc[0]
    for field in TUNED_FIELDS:
        assert first[f"params_{field}"] == pytest.approx(getattr(test_settings.model.lgbm, field))
    assert result.best_wrmsse <= result.baseline_wrmsse
    assert result.improvement_pct >= 0
    assert set(result.best_params) == set(TUNED_FIELDS)


def test_tuning_is_reproducible(tuning_df: pd.DataFrame, test_settings: Settings) -> None:
    a = tune_lightgbm(tuning_df, test_settings, n_trials=2)
    b = tune_lightgbm(tuning_df, test_settings, n_trials=2)
    assert a.best_params == b.best_params
    assert a.best_wrmsse == pytest.approx(b.best_wrmsse)


def test_tuning_logs_every_trial_to_mlflow(
    tuning_df: pd.DataFrame, test_settings: Settings
) -> None:
    tracker = Tracker(
        test_settings.mlflow_tracking_uri, "tuning-test", test_settings.paths.artifacts_dir
    )
    tune_lightgbm(tuning_df, test_settings, n_trials=2, tracker=tracker)
    runs = mlflow.search_runs(experiment_names=["tuning-test"])
    trials = runs[runs["tags.mlflow.runName"].str.startswith("trial-")]
    assert len(trials) == 2
    assert trials["metrics.val_fold1_wrmsse"].notna().all()
    assert "params.learning_rate" in runs.columns


def test_invalid_budget(tuning_df: pd.DataFrame, test_settings: Settings) -> None:
    with pytest.raises(ValueError, match="n_trials"):
        tune_lightgbm(tuning_df, test_settings, n_trials=0)
