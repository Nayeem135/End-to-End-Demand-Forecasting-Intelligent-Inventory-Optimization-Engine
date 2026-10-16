# Phased Implementation Plan

> **Status (2026-10-16): all phases implemented.** Notable deviations from the original plan,
> each made for a documented reason (see `docs/methodology.md`):
>
> * Test period = 3 rolling 28-day windows (84 days) instead of one, for a longer simulation.
> * LightGBM gained horizon-aware *origin* features (lags ≥ horizon alone ignored the latest month).
> * Safety stock uses bias + z·σ of *cumulative* lead-time errors; EOQ uses the long-run demand rate.
> * The simulation enforces a strict no-look-ahead information rule for every policy.
> * `demand-engine demo` (embedded PostgreSQL) added so the defence needs no Docker.
> * Frontend lint uses oxlint (scaffold default) instead of ESLint.

Each phase ends with ruff, mypy, and pytest all green, plus a short demoable
outcome. Estimates assume part-time FYP work.

---

## Phase 0: Foundation (≈3–4 days)

**Goal:** an empty but production-shaped repo where CI is green from day one.

- `git init`, `.gitignore` (data/, mlruns/, artifacts/, .env, node_modules/).
- `pyproject.toml` (uv, Python 3.11), with ruff and mypy (strict) config and
  pytest config (markers: `integration`, `slow`).
- `src/demand_engine/config.py`: nested `Settings` (paths, data subset, seed,
  model params, inventory costs, DB URL, MLflow URI) and `.env.example`.
- `logging.py` (consistent format, level from config).
- `tests/fixtures/synthetic.py`: generates small M5-shaped wide CSVs (trend,
  weekly seasonality, intermittent series, price changes) for all tests.
- GitHub Actions: uv install → ruff → mypy → pytest (unit), with a Postgres
  service for integration tests.
- README skeleton with setup and dataset download steps.

**Done when:** `uv run pytest` passes on config and logging tests, and CI is green.

## Phase 1: Data ingestion and validation (≈1 week)

- `data/loader.py`: read the three M5 CSVs, with explicit dtypes to save memory.
- `data/subset.py`: filter by state; stratified, seeded sampling of N items.
- `data/transform.py`: wide → long, then join calendar and prices; handle
  pre-launch zeros (drop leading zeros before the first sale).
- `data/validation.py`: schema, dtype, non-negativity, date continuity, and
  duplicate checks, raising `DataValidationError`.
- `data/splits.py`: rolling-origin fold generator plus a held-out test window.
- Pipeline step `ingest` writes `data/processed/sales.parquet`.
- `notebooks/01_eda.ipynb`: demand patterns, intermittency (ADI/CV²
  classification), seasonality, and price effects. These make good defence
  slides.

**Done when:** ingestion runs on the real subset in under 1 minute, and every
function has unit tests on synthetic data.

## Phase 2: Features and baseline forecasting (≈1 week)

- `features/calendar.py` (dow, month, SNAP, events), `features/price.py`
  (relative price, price change), and `features/lags.py` (lags ≥ 28, rolling
  mean/std on shifted series).
- **Leakage test:** a property-style test asserts that no feature at *t* depends
  on target values after *t − h*.
- `models/base.py`: `Forecaster` protocol / ABC.
- `models/baselines.py`: Naive, SeasonalNaive(7), MovingAverage.
- `models/statistical.py`: StatsForecast AutoETS, CrostonOptimized, TSB.
- `evaluation/metrics.py`: MAE, RMSE, bias, RMSSE, WRMSSE, pinball, coverage
  (unit-tested against hand-computed values).
- `evaluation/backtest.py`: runs any `Forecaster` across folds and returns a
  tidy metrics table.

**Done when:** a backtest leaderboard of the baseline and statistical models
exists.

## Phase 3: LightGBM, uncertainty, MLflow, and SHAP (≈1.5 weeks)

- `models/lgbm.py`: a global LightGBM model (Tweedie), seeded, with
  hyperparameters from config. Quantile models (P10/P50/P90) or
  residual-quantile intervals.
- MLflow logging in the train pipeline: params, per-fold metrics, model
  artifact, feature importance.
- `evaluation/explain.py`: SHAP global summary and per-series local
  explanations, saved as artifacts and as data for the frontend.
- Optional: light Optuna tuning with a fixed seed and a small budget.
- `notebooks/02_model_comparison.ipynb`: leaderboard, error by category and
  intermittency class.

**Done when:** LightGBM beats SeasonalNaive on WRMSSE in the backtest, all
runs are visible in the MLflow UI, and the SHAP plots are generated.

## Phase 4: Inventory optimization and simulation (≈1.5 weeks)

This is the core contribution, so it gets the most rigorous tests.

- `optimization/policy.py`: pure functions for `safety_stock`, `reorder_point`,
  `eoq`, and `newsvendor_quantity`. Tests cover known textbook examples and
  edge cases (zero demand, σ = 0).
- `optimization/uncertainty.py`: σ over the lead time from backtest residuals.
- `optimization/simulator.py`: a deterministic daily lost-sales simulator
  (on-hand, pipeline orders, lead time, review period), with a clear
  per-day event order. Invariant tests check inventory conservation and that
  costs are never negative.
- `optimization/compare.py`: baseline vs forecast-driven policy, reporting
  cost breakdown, fill rate, stockout days, and average inventory.
- Sensitivity analysis: sweep the service level and lead time to get a
  cost-vs-service trade-off curve, which is a strong defence chart.

**Done when:** the simulation on the test window gives a reproducible % cost
saving at the target fill rate.

## Phase 5: Persistence and API (≈1 week)

- SQLAlchemy 2 models: `Series`, `ModelRun`, `Forecast`, `InventoryPolicy`,
  `SimulationRun`, `SimulationResult`, plus the Alembic initial migration.
- Repository layer (no SQL in routers). A pipeline step `publish` loads
  results into Postgres.
- FastAPI routers:
  - `GET /health`
  - `GET /series`, with filters and pagination
  - `GET /series/{id}/forecast`: history, forecast, and intervals
  - `GET /series/{id}/policy`: SS, ROP, Q, with an explanation
  - `GET /series/{id}/explanation`: SHAP
  - `GET /models/leaderboard`
  - `GET /simulations/latest`, `GET /simulations/{id}`
  - `POST /simulations/what-if`: recomputes the policy and simulation for
    user-chosen service level, lead time, and costs (the live demo highlight)
- Pydantic response schemas and OpenAPI docs. Integration tests use a real
  Postgres (CI service / Compose).

**Done when:** every endpoint has tests and Swagger UI works.

## Phase 6: Frontend dashboard (≈1.5 weeks)

- Vite + React + TS + Tailwind, with a typed API client (generated from
  OpenAPI or hand-typed).
- Pages:
  1. **Overview:** KPI cards (cost saved %, fill rate, forecast accuracy) and
     a cost-breakdown chart.
  2. **Forecast explorer:** select a SKU-store and see history, forecast, and
     the P10–P90 band.
  3. **Inventory:** SS / ROP / Q table and a simulated inventory-level chart
     with reorder events.
  4. **What-if simulator:** sliders for service level, lead time, and costs
     that update results live.
  5. **Model insights:** leaderboard and SHAP feature importance.
- Loading, empty, and error states, plus Vitest tests for the components and
  formatting utilities.

**Done when:** the full demo flow works against the local API.

## Phase 7: Integration, hardening, and defence prep (≈1 week)

- `pipelines/run_all.py`: one command runs ingest → train → backtest →
  optimize → simulate → publish.
- Docker Compose: `db`, `mlflow`, `api` (runs migrations on start), and
  `frontend`, with health checks.
- An end-to-end integration test on synthetic data (whole pipeline plus API).
- Coverage target ≥ 85% on `src/demand_engine` (excluding the API entrypoint).
- Docs: `architecture.md` (diagram), `methodology.md` (models, metrics,
  inventory math, assumptions and limitations), and `demo_script.md`
  (5-minute walkthrough plus a fallback with cached results if anything fails
  live).
- Rehearse a cold start: from `git clone` to a running dashboard on the laptop.

**Done when:** a fresh clone plus `docker compose up` gives a working demo,
and CI is green.

---

## Risk register

| Risk | Mitigation |
|---|---|
| M5 too large for laptop RAM | Subset early, explicit dtypes, Parquet, configurable N |
| Data leakage inflates accuracy | Lags ≥ horizon, dedicated leakage tests, untouched test window |
| Live demo failure | Precomputed results in DB, cached fallback, Compose health checks |
| Intermittent demand hurts ML | Croston/TSB baselines, Tweedie loss, per-intermittency-class reporting |
| Savings look cherry-picked | Same test window and costs for both policies, sensitivity sweep, honest limitations section |
| Scope creep | Phases 0–5 are the core; 6–7 polish. Optional extras only after Phase 7 |
