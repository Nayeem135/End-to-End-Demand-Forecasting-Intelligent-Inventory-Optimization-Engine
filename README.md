# End-to-End Demand Forecasting & Intelligent Inventory Optimization Engine

Forecasts daily demand for every SKU-store pair of the **M5 (Walmart)** dataset, turns the
forecasts into **safety stock, reorder points and order quantities**, and proves the value
by **replaying 12 weeks of unseen demand** through a day-by-day inventory simulation —
served through a FastAPI backend and a React dashboard with a live what-if simulator.

![CI](https://img.shields.io/badge/CI-ruff%20%C2%B7%20mypy%20strict%20%C2%B7%20pytest-brightgreen)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)

## Results (California, 498 items × 4 stores = 1,961 series, test period 29 Feb – 22 May 2016)

| | Rule of thumb | Moving avg + safety stock | **Forecast-driven (LightGBM)** |
|---|---:|---:|---:|
| Total simulated cost | $279,774 | $65,361 | **$65,126** |
| Fill rate | 84.2 % | 98.6 % | **98.6 %** |
| Orders placed | 14,674 | 3,083 | **3,073** |

* **Inventory optimisation saves 76.7 %** of total cost versus a typical min/max rule while
  lifting the fill rate from 84.2 % to 98.6 %.
* **Better forecasts make the same policy cheaper at every target service level** — by
  ≈ 5 % at an 80 % target, shrinking to 0.1–0.4 % at 95 % and above (0.4 % at the default).
  Measured fairly: both policies are identical except for the forecast. At high targets the
  gain is within the run-to-run sensitivity of a single LightGBM fit, so the robust claim is
  the direction and the trend, not the exact decimal.
* **LightGBM is the most accurate of seven models**: test WRMSSE 0.835 vs 0.865 (ETS) and
  0.870 (moving average); 80 % prediction intervals cover 86.8 % of actuals.

The full reasoning — including two evaluation pitfalls found and fixed along the way —
is in [docs/methodology.md](docs/methodology.md).

## What is inside

| Layer | Highlights |
|---|---|
| **Data** | chunked M5 loader, stratified seeded subset, wide→long, pandera schema validation, synthetic M5-format generator (tests & offline demo) |
| **Features** | calendar, SNAP, events, relative prices; lags ≥ horizon plus horizon-aware *origin* features; leakage proven by tests |
| **Models** | naive, seasonal naive, moving average, Croston, TSB, ETS (statsforecast, conformal intervals); global LightGBM (Tweedie + quantiles) |
| **Evaluation** | rolling-origin backtesting (3 validation + 3 test windows), WRMSSE, RMSSE, MAE, bias, pinball, coverage; SHAP; MLflow tracking |
| **Optimisation** | bias-corrected safety stock from measured lead-time errors, reorder points that follow the daily forecast, EOQ, vectorised lost-sales simulator, service-level sensitivity |
| **Serving** | PostgreSQL + SQLAlchemy 2 + Alembic, bulk `COPY` publishing, FastAPI with live what-if (< 1 s for ~2,000 series) |
| **Dashboards** | React 19 + TypeScript + Tailwind + Recharts (overview, forecast explorer, inventory, what-if, models) and a Streamlit app over the same results |
| **Engineering** | `mypy --strict`, ruff, 200+ tests incl. real-PostgreSQL integration, 97 % coverage, Docker Compose, GitHub Actions |

## Quick start

Prerequisites: Python 3.11 via [uv](https://docs.astral.sh/uv/), Node 22+, GNU Make (Windows: `winget install ezwinports.make`). Docker is optional.

```bash
uv sync                                  # Python environment
cd frontend && npm ci && npm run build && cd ..
```

**Data.** Put the three M5 files (`sales_train_evaluation.csv`, `calendar.csv`,
`sell_prices.csv`) from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
into `data/raw/` — or run `uv run demand-engine synthetic` for a fully offline synthetic dataset
in the same format.

### Option A — no Docker (recommended for the laptop demo)

```bash
make setup        # uv sync, frontend build, git hooks
make data         # data/raw M5 CSVs -> data/processed/sales.parquet
make train        # 7 models + backtests + inventory simulation (~9 min)
make api          # embedded PostgreSQL + publish + API & React dashboard on :8000
make app          # alternative Streamlit dashboard on :8501 (no database needed)
```

Without `make`, the same commands are `uv run demand-engine ingest | run | demo` and
`uv run streamlit run app/streamlit_app.py`. API docs: http://127.0.0.1:8000/docs

### Option B — Docker Compose

```bash
docker compose --profile pipeline run --rm pipeline   # run the pipeline + publish
docker compose up --build                              # http://localhost:8080
docker compose --profile mlflow up mlflow              # MLflow UI on http://localhost:5000
```

### Development

```bash
make lint && make test                     # ruff, mypy --strict, pytest
uv run pre-commit run --all-files          # also runs on every commit after `make setup`
cd frontend && npm run lint && npm run typecheck && npm test
uv run demand-engine serve & (cd frontend && npm run dev)   # hot-reload UI on :5173
```

All settings (subset, horizon, models, costs, lead time, database…) are environment
variables documented in [.env.example](.env.example).

## Repository map

```text
src/demand_engine/   data · features · models · evaluation · optimization · pipelines · api
frontend/            React dashboard
tests/               unit (mirrors src) + integration (PostgreSQL, API, end-to-end)
notebooks/           01_eda (data insights) · 02_results (accuracy, SHAP, inventory value)
docs/                architecture · methodology · demo script · implementation plan
```

See [docs/architecture.md](docs/architecture.md) for the design and
[docs/demo_script.md](docs/demo_script.md) for the defence walkthrough.

## Data licence

The M5 data belongs to its organisers and Walmart and is used here for academic,
non-commercial purposes under the competition terms. It is not redistributed in this
repository (`data/` is git-ignored).
