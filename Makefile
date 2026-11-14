# Developer entry points. Every target is a thin wrapper around uv / npm, so the same
# commands work in Windows cmd, Git Bash and Linux CI.
.DEFAULT_GOAL := help
.PHONY: help setup data train test lint run docker-up api app

help:  ## List targets
	@echo "setup  install Python + frontend deps and git hooks"
	@echo "data   build data/processed/sales.parquet from data/raw (M5 CSVs)"
	@echo "train  forecast, backtest, optimise and simulate -> artifacts/runs/<id>"
	@echo "test   run the full test suite (unit + integration)"
	@echo "lint   ruff lint, ruff format check, mypy --strict"
	@echo "run    start the API + dashboard on :8000 (embedded PostgreSQL, no Docker)"
	@echo "docker-up  Postgres + API + dashboard (:8080) + MLflow (:5000) via Docker Compose"
	@echo "api    same as run"
	@echo "app    Streamlit dashboard on :8501 (reads artifacts, no database)"

setup:
	uv sync
	npm --prefix frontend ci
	npm --prefix frontend run build
	uv run pre-commit install

data:
	uv run demand-engine ingest

train:
	uv run demand-engine run

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

run:
	uv run demand-engine demo

docker-up:
	docker compose --profile mlflow up --build

api: run

app:
	uv run streamlit run app/streamlit_app.py
