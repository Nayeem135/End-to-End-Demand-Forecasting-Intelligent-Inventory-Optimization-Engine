# End-to-End Demand Forecasting & Intelligent Inventory Optimization Engine

University final-year project. It must demonstrate **both** data science and
software engineering, be bug-free, and run reliably in a live defence demo on a
laptop.

**Constraints:** 7-day build. Keep the end-to-end system working at all times;
**add no features the user did not ask for** — propose them instead.

## Project goal

1. **Forecast** daily unit demand for every SKU-store pair, with uncertainty
   (point forecast plus quantiles).
2. **Optimize** inventory by turning forecasts into decisions per SKU-store:
   safety stock, reorder point, and order quantity.
3. **Prove value** by simulating day-by-day replenishment on held-out actual
   demand and comparing total cost and service level against a naive baseline
   policy.

The headline result for the defence is: *"Forecast-driven policy reduces total
inventory cost by X% while keeping fill rate ≥ target, compared with the
baseline."* Every number must be reproducible from one command.

## Dataset: M5 Forecasting (Walmart)

- Files: `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`
  (Kaggle "M5 Forecasting - Accuracy"). The user downloads them manually into
  `data/raw/`. Never commit data.
- **Configurable subset**: by default, one state (`CA`, all 4 stores) and 500 items,
  sampled reproducibly with the global seed and stratified by category (`DATA__STATE`,
  `DATA__STORES`, `DATA__N_ITEMS`, `SEED`). A single store with ~150 items is one config
  change away (`DATA__STORES=["CA_1"]`, `DATA__N_ITEMS=150`); the user chose to keep the
  larger default because the reported results are computed on it.
- Raw wide CSVs are converted once to a long-format Parquet table
  (`item_id, store_id, date, sales, sell_price, calendar features`) in
  `data/processed/`. The pipelines read Parquet. PostgreSQL stores **results**
  (series metadata, forecasts, policies, simulation runs), not the raw sales.
- Tests never need the real dataset: `tests/fixtures` provides a synthetic
  generator for M5-shaped data.

## Tech stack

| Area | Tools |
|---|---|
| Language / deps | Python 3.11, `uv` (`pyproject.toml` + `uv.lock`) |
| Data / features | pandas, pyarrow, numpy, pandera (demand-table schema in `data/validation.py`) |
| Forecasting | `statsforecast` (baselines, ETS, Croston/TSB/ADIDA), LightGBM |
| Explainability | SHAP (TreeExplainer on LightGBM) |
| Tracking | MLflow (SQLite store `artifacts/mlflow.db`; UI via Compose profile `mlflow`) |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2 (typed ORM), Alembic, PostgreSQL 16 |
| Config | `pydantic-settings` (`.env` plus env vars, nested via `__`) |
| Quality | pytest (+ pytest-cov), ruff (lint + format), mypy (strict) |
| Frontend | React 19, TypeScript (strict), Vite, Tailwind CSS 4, Recharts, Vitest, oxlint |
| Streamlit app | `app/streamlit_app.py` — reads Parquet artifacts directly (no DB/API needed) |
| Dev tooling | GNU Make (`Makefile`), pre-commit (`.pre-commit-config.yaml`: ruff, mypy, hygiene hooks) |
| Infra | Docker Compose (db, api, frontend; `pipeline`/`mlflow` profiles), GitHub Actions CI, embedded PostgreSQL (`pgserver`) for Docker-free demos/tests |

## Repository structure (src layout)

```text
.
├── CLAUDE.md
├── README.md
├── pyproject.toml / uv.lock
├── alembic.ini                   # URL comes from settings, never stored here
├── Makefile                      # setup, data, train, test, lint, api, app
├── .pre-commit-config.yaml / .streamlit/config.toml
├── Dockerfile / docker-compose.yml / .dockerignore
├── .env.example                  # every setting documented; real .env is git-ignored
├── .github/workflows/ci.yml
├── src/demand_engine/            # the single importable Python package
│   ├── config.py                 # pydantic-settings Settings (the only source of paths/params)
│   ├── logging_config.py         # logging setup (not logging.py: avoid shadowing stdlib)
│   ├── types.py                  # FloatArray/IntArray aliases, typed `cell()` accessor
│   ├── data/                     # loader, subset, transform, validation, splits, synthetic
│   ├── features/                 # calendar, price, lags (>= horizon), origin (horizon-aware), builder
│   ├── models/                   # base (Forecaster contract), statistical, lgbm, registry
│   ├── evaluation/               # metrics, backtest (rolling origin), explain (SHAP)
│   ├── optimization/             # formulas, uncertainty, policies, simulator, scenario
│   ├── api/                      # main (app factory), routes, schemas, repository, simulation_service
│   │   └── db/                   # models, session, bulk (COPY), migrate, migrations/ (Alembic)
│   └── pipelines/                # ingest, forecasting, optimize, run_all, artifacts, tracking, publish, cli
├── app/streamlit_app.py          # Streamlit dashboard over the latest run's artifacts
├── frontend/                     # React + TS + Vite dashboard (own package.json, Dockerfile, nginx.conf)
├── notebooks/                    # 01_eda, 02_results; import from demand_engine, no logic of their own
├── tests/
│   ├── conftest.py               # synthetic M5 fixtures (tests never need the real dataset)
│   ├── unit/                     # mirrors src/demand_engine/ package by package
│   └── integration/              # real PostgreSQL: migrations, publish, every API endpoint, e2e pipeline
├── docs/                         # architecture, methodology, demo script, implementation plan
└── data/  artifacts/             # git-ignored, created at runtime
```

## Core technical decisions (do not change silently)

- **Forecaster interface**: `fit(history) -> self`, `predict(future) -> DataFrame[unique_id, ds,
  yhat, yhat_lo, yhat_hi]`. The base class drops `y` from `future` and enforces the output
  contract, so models are interchangeable in backtests, the optimiser and the API.
- **Horizon / protocol**: 28 days. Rolling origin: 3 validation folds (champion selection,
  error statistics) then 3 test windows (84 days; reporting + simulation only).
- **Leakage safety**: fixed lags/rolling windows are `>= horizon`; *origin* features
  (horizon `h`, demand stats at `t - h`) restore recent information. Tests perturb future
  targets and assert window features are unchanged — keep these tests passing.
- **Models**: naive, seasonal naive, moving average, Croston, TSB, ETS(A,N,A) via
  statsforecast (conformal intervals); global LightGBM (Tweedie + P10/P90 quantile models).
- **Metrics**: MAE, RMSE, bias, RMSSE, revenue-weighted WRMSSE (headline), pinball, coverage.
- **Inventory**: protection period `P = L + R - 1`. From validation folds, the P-day
  cumulative error gives `bias` and `sigma`; reorder point `s_t = forecast_P + bias + z*sigma`.
  Order size = EOQ of the long-run (last-year) demand rate, shared by the forecast policies.
- **No look-ahead in simulation**: on day t a policy uses only the forecast issued at the
  latest cut-off <= t; beyond its window a forecast repeats its final week.
- **Fair comparison**: `ma_safety_stock` and `forecast_driven` are the identical policy fed
  with different forecasts; `rule_of_thumb` is the min/max baseline. Same demand, costs, start stock.
- **API what-if** must reproduce the pipeline's stored results exactly at default
  parameters (integration test `test_what_if_default_reproduces_pipeline_exactly`).
- **Seeds**: one global `SEED`, passed explicitly (subset sampling, LightGBM, training horizons).

## Coding rules

- Type hints everywhere; `mypy --strict` must pass on `src/`.
- Docstrings (Google style) on every module, public class, and public function.
  The docstring states units and assumptions, e.g. "units/day" or "cost per unit
  per day".
- **No hardcoded paths or magic parameters**: read them from
  `demand_engine.config.get_settings()`, and pass settings or values into
  functions instead of reading globals deep inside logic.
- Use `logging` via `logging.getLogger(__name__)`; **never `print`** in `src/`.
- Every module in `src/demand_engine/` has a matching unit-test file.
- Keep functions small and pure where possible: I/O stays at the edges
  (pipelines, API, repositories), and the math stays in pure, easily tested
  functions.
- Validate inputs at boundaries (Pydantic schemas, DataFrame column/dtype
  checks) and raise explicit, descriptive exceptions. Never swallow errors or
  return fake defaults.
- Fixed random seeds; the same config must produce identical outputs.
- Frontend: strict TypeScript, no `any`, a typed API client, and components
  kept small; numbers are formatted consistently.

## Commands

```bash
make setup | data | train | test | lint | api | app   # day-to-day entry points (see `make`)
uv run pre-commit run --all-files                     # same checks as the git hook
uv sync                                     # install deps (Python 3.11 venv)
uv run ruff check . && uv run ruff format --check .
uv run mypy                                 # strict, configured in pyproject
uv run pytest                               # unit + integration (embedded Postgres via pgserver)
uv run pytest -m "not integration"          # fast unit tests only
uv run demand-engine synthetic              # offline synthetic M5-format data (if no Kaggle data)
uv run demand-engine run-all --no-publish   # ingest + forecast + optimise + simulate (~9 min default)
uv run demand-engine demo                   # embedded Postgres + migrate + publish + API (+ dashboard)
uv run demand-engine serve --dashboard frontend/dist   # API against DATABASE__URL
docker compose up --build                   # db + api + dashboard (http://localhost:8080)
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```

## Required workflow

After **every** task:

1. Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
   and `uv run pytest` (plus the frontend lint/typecheck/test when the frontend
   changed).
2. Fix **all** failures. Do not disable rules, add blanket `# type: ignore`, or
   skip tests to get green.
3. Only then report the task done, quoting the actual check results. If a check
   cannot run (for example, Postgres is unavailable), say so explicitly rather
   than claiming success.

## Implementation plan

The phased plan, with deliverables and definition of done, is in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md). Work phase by
phase; do not start a phase until the previous one passes all checks.
