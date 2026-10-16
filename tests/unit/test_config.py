from pathlib import Path

import pytest
from pydantic import ValidationError

from demand_engine.config import (
    BacktestSettings,
    LightGBMSettings,
    ModelSettings,
    Settings,
    get_settings,
)


def test_defaults_are_laptop_sized() -> None:
    s = Settings()
    assert s.data.state == "CA"
    assert s.data.n_items == 500
    assert s.backtest.horizon == 28
    assert s.seed == 42


def test_env_override_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA__STATE", "TX")
    monkeypatch.setenv("INVENTORY__LEAD_TIME_DAYS", "3")
    s = Settings()
    assert s.data.state == "TX"
    assert s.inventory.lead_time_days == 3


def test_invalid_state_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA__STATE", "NY")
    with pytest.raises(ValidationError):
        Settings()


def test_lags_shorter_than_horizon_rejected() -> None:
    with pytest.raises(ValidationError, match="leakage"):
        Settings(
            backtest=BacktestSettings(horizon=28),
            model=ModelSettings(lgbm=LightGBMSettings(lags=[7, 28])),
        )


def test_lags_are_sorted_and_deduplicated() -> None:
    assert LightGBMSettings(lags=[56, 28, 28]).lags == [28, 56]


def test_paths_derived() -> None:
    s = Settings()
    assert s.paths.raw_dir == Path("data") / "raw"
    assert s.paths.processed_dir == Path("data") / "processed"


def test_get_settings_cached() -> None:
    assert get_settings() is get_settings()


def test_min_train_days_must_cover_conformal_windows() -> None:
    with pytest.raises(ValidationError, match="conformal"):
        Settings(backtest=BacktestSettings(horizon=28, min_train_days=100))


def test_env_example_documents_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.env.example must parse cleanly and describe exactly the built-in defaults."""
    example = Path(__file__).resolve().parents[2] / ".env.example"
    monkeypatch.chdir(tmp_path)
    assert Settings(_env_file=example) == Settings(_env_file=None)  # type: ignore[call-arg]
