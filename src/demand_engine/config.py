"""Central, typed configuration for the whole engine.

All paths, subset choices, model hyper-parameters, inventory costs and service
settings live here. Values come from (in priority order) environment variables,
a ``.env`` file in the working directory, and the defaults below. Nested fields
use ``__`` as the delimiter, e.g. ``DATA__STATE=TX`` or ``INVENTORY__LEAD_TIME_DAYS=5``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

State = Literal["CA", "TX", "WI"]
CONFORMAL_WINDOWS = 5  # calibration windows for conformal prediction intervals


class PathsSettings(BaseModel):
    """Filesystem locations (relative paths resolve against the working directory)."""

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    @property
    def raw_dir(self) -> Path:
        """Directory holding the original M5 CSV files."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory holding derived Parquet tables."""
        return self.data_dir / "processed"


class DataSettings(BaseModel):
    """Which slice of M5 to work with."""

    state: State = "CA"
    n_items: int = Field(default=500, ge=1, le=3049)
    stores: list[str] | None = Field(
        default=None, description="Optional explicit store_ids; default = all stores of the state."
    )
    history_days: int | None = Field(
        default=730,
        ge=120,
        description="Keep only the most recent N days of history (None = all ~1941 days).",
    )


class BacktestSettings(BaseModel):
    """Rolling-origin evaluation protocol."""

    horizon: int = Field(default=28, ge=1)
    n_folds: int = Field(default=3, ge=1, description="Validation folds (model selection, errors).")
    n_test_windows: int = Field(
        default=3,
        ge=1,
        description="Consecutive held-out horizons used for the final test/simulation.",
    )
    min_train_days: int = Field(
        default=180, ge=28, description="Series need this much history before the first cut-off."
    )


class LightGBMSettings(BaseModel):
    """Hyper-parameters for the global LightGBM forecaster."""

    objective: Literal["tweedie", "poisson", "regression"] = "tweedie"
    tweedie_variance_power: float = Field(default=1.1, ge=1.0, lt=2.0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=63, ge=2)
    min_data_in_leaf: int = Field(default=50, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=1, ge=0)
    lambda_l2: float = Field(default=0.1, ge=0)
    n_estimators: int = Field(default=400, ge=1)
    lags: list[int] = Field(default_factory=lambda: [28, 35, 42, 49, 56, 364])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28, 56])
    target_encoding: bool = Field(
        default=False,
        description="Add leakage-free expanding target encodings (item, store-dept, dept-weekday).",
    )

    @field_validator("lags")
    @classmethod
    def _lags_positive(cls, value: list[int]) -> list[int]:
        if not value or min(value) < 1:
            raise ValueError("lags must be a non-empty list of positive integers")
        return sorted(set(value))


ModelName = Literal[
    "naive", "seasonal_naive", "moving_average", "croston", "tsb", "ets", "lightgbm"
]
ALL_MODELS: tuple[ModelName, ...] = (
    "naive",
    "seasonal_naive",
    "moving_average",
    "croston",
    "tsb",
    "ets",
    "lightgbm",
)


class ModelSettings(BaseModel):
    """Model selection and uncertainty settings."""

    candidates: list[ModelName] = Field(default_factory=lambda: list(ALL_MODELS), min_length=1)
    baseline: ModelName = Field(
        default="moving_average", description="Forecast used by the naive inventory policy."
    )

    interval_level: int = Field(default=80, gt=0, lt=100, description="Central interval width (%).")
    season_length: int = Field(default=7, ge=1)
    moving_average_window: int = Field(default=28, ge=1)
    lgbm: LightGBMSettings = Field(default_factory=LightGBMSettings)


class InventorySettings(BaseModel):
    """Inventory economics and policy parameters (costs are per unit unless stated)."""

    lead_time_days: int = Field(default=7, ge=1, le=60)
    review_period_days: int = Field(default=1, ge=1)
    service_level: float = Field(
        default=0.95, gt=0.5, lt=1.0, description="Target cycle service level"
    )
    holding_cost_rate: float = Field(
        default=0.25, gt=0, description="Annual holding cost as a fraction of unit price."
    )
    ordering_cost: float = Field(
        default=10.0, ge=0, description="Fixed cost per order line (handling, receiving, admin)."
    )
    stockout_cost_multiplier: float = Field(
        default=1.0,
        ge=0,
        description="Lost-sale penalty as a multiple of unit price (lost margin + goodwill).",
    )
    baseline_cover_days: int = Field(
        default=14, ge=1, description="Baseline rule: order up to N days of average sales."
    )
    initial_cover_days: int = Field(
        default=14, ge=0, description="Starting on-hand stock in days of demand."
    )


class DatabaseSettings(BaseModel):
    """Relational store for published results."""

    url: str = "postgresql+psycopg://demand:demand@localhost:5432/demand"
    echo: bool = False
    connect_timeout: int = Field(
        default=5, ge=1, description="Seconds before a connection attempt fails."
    )


class ApiSettings(BaseModel):
    """HTTP service settings."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    static_dir: Path | None = Field(
        default=None, description="Built dashboard (frontend/dist) to serve from the API, if set."
    )


class Settings(BaseSettings):
    """Root settings object; obtain it via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["text", "json"] = Field(
        default="text", description="'text' for readable lines, 'json' for structured logs."
    )
    mlflow_tracking_uri: str = Field(
        default="sqlite:///artifacts/mlflow.db",
        description="MLflow backend store; empty string disables tracking.",
    )
    mlflow_experiment: str = "demand-engine"
    paths: PathsSettings = Field(default_factory=PathsSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    inventory: InventorySettings = Field(default_factory=InventorySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        """Cross-field rules that individual fields cannot express."""
        horizon = self.backtest.horizon
        if min(self.model.lgbm.lags) < horizon:
            raise ValueError(f"all LightGBM lags must be >= horizon ({horizon}) to avoid leakage")
        needed = CONFORMAL_WINDOWS * horizon + 1
        if self.backtest.min_train_days < needed:
            raise ValueError(
                f"min_train_days must be >= {needed} "
                f"({CONFORMAL_WINDOWS} conformal windows x horizon)"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
