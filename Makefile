# Developer entry points. Every target is a thin wrapper around uv / npm, so the same
# commands work in Windows cmd, Git Bash and Linux CI.
.DEFAULT_GOAL := help
.PHONY: help setup data train test lint api app

help:  ## List targets
	@echo "setup  install Python + frontend deps and git hooks"
	@echo "data   build data/processed/sales.parquet from data/raw (M5 CSVs)"
	@echo "train  forecast, backtest, optimise and simulate -> artifacts/runs/<id>"
	@echo "test   run the full test suite (unit + integration)"
	@echo "lint   ruff lint, ruff format check, mypy --strict"
	@echo "api    embedded PostgreSQL + publish latest run + API/dashboard on :8000"
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

api:
	uv run demand-engine demo

app:
	uv run streamlit run app/streamlit_app.py
