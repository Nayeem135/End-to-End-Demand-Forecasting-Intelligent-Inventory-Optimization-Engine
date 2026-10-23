"""Streamlit dashboard for the demand forecasting & inventory optimisation engine.

Reads the latest completed pipeline run straight from its Parquet artifacts, so it
needs neither the database nor the API:

    uv run --group app streamlit run app/streamlit_app.py      (or: make app)

The what-if tab re-runs the same simulation code as the pipeline and the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from demand_engine.config import InventorySettings, get_settings
from demand_engine.optimization.scenario import (
    SimulationInputs,
    inputs_from_stored,
    replace_params,
    run_scenario,
)
from demand_engine.pipelines.artifacts import ArtifactNotFoundError, RunArtifacts, latest_run

POLICY_LABELS = {
    "rule_of_thumb": "Rule of thumb",
    "ma_safety_stock": "Moving avg + safety stock",
    "forecast_driven": "Forecast-driven (ML)",
}


@dataclass(frozen=True)
class RunData:
    """Everything the app shows, loaded once per run."""

    manifest: dict[str, Any]
    leaderboard: pd.DataFrame
    portfolio: pd.DataFrame
    sensitivity: pd.DataFrame
    series: pd.DataFrame
    history: pd.DataFrame
    forecasts: pd.DataFrame
    policies: pd.DataFrame
    shap_global: pd.DataFrame | None
    inputs: SimulationInputs
    params: InventorySettings


@st.cache_resource(show_spinner="Loading pipeline results…")
def load_run(root: str) -> RunData:
    """Load a run's artifacts and rebuild its simulator inputs (cached per run directory)."""
    run = RunArtifacts(Path(root))
    manifest = run.load_json("manifest")
    history = run.load_frame("history")
    forecasts = run.load_frame("forecasts")
    series = run.load_frame("series")
    inputs = inputs_from_stored(
        test_start=pd.Timestamp(manifest["test_start"]),
        champion=manifest["champion"],
        baseline=manifest["baseline_model"],
        history=history,
        forecasts=forecasts,
        validation=run.load_frame("val_forecasts"),
        series=series,
    )
    try:
        shap_global: pd.DataFrame | None = run.load_frame("shap_global")
    except ArtifactNotFoundError:
        shap_global = None
    return RunData(
        manifest=manifest,
        leaderboard=run.load_frame("leaderboard"),
        portfolio=run.load_frame("portfolio"),
        sensitivity=run.load_frame("sensitivity"),
        series=series,
        history=history,
        forecasts=forecasts,
        policies=run.load_frame("policies"),
        shap_global=shap_global,
        inputs=inputs,
        params=InventorySettings.model_validate(manifest["settings"]["inventory"]),
    )


def labelled(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Portfolio indexed by human-readable policy names."""
    return portfolio.assign(policy=portfolio["policy"].map(POLICY_LABELS)).set_index("policy")


def overview_tab(data: RunData) -> None:
    """Headline KPIs, cost breakdown and service-level sensitivity."""
    h = data.manifest["headline"]
    accuracy_gain = 100 * (1 - h["champion_test_wrmsse"] / h["baseline_test_wrmsse"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inventory cost saving", f"{h['savings_vs_rule_pct']:.1f}%", help="vs rule of thumb")
    c2.metric("Extra saving from ML", f"{h['savings_vs_ma_pct']:.1f}%", help="vs moving average")
    c3.metric(
        "Fill rate",
        f"{100 * h['fill_rate_forecast_driven']:.1f}%",
        delta=f"{100 * (h['fill_rate_forecast_driven'] - h['fill_rate_rule_of_thumb']):.1f} pts",
    )
    c4.metric("Forecast accuracy gain", f"{accuracy_gain:.1f}%", help="WRMSSE vs moving average")

    port = labelled(data.portfolio)
    st.subheader("Where the money goes")
    st.bar_chart(port[["holding_cost", "ordering_cost", "stockout_cost"]], horizontal=True)
    st.dataframe(
        port[["total_cost", "fill_rate", "n_orders", "savings_vs_baseline_pct"]].style.format(
            {"total_cost": "${:,.0f}", "fill_rate": "{:.1%}", "savings_vs_baseline_pct": "{:.1f}%"}
        )
    )
    st.subheader("Cost vs target service level (optimised policies)")
    sens = data.sensitivity[data.sensitivity["policy"] != "rule_of_thumb"]
    wide = sens.pivot(index="service_level", columns="policy", values="total_cost")
    st.line_chart(wide.rename(columns=POLICY_LABELS))


def forecasts_tab(data: RunData) -> None:
    """Actuals vs champion and baseline forecasts for one series."""
    ids = data.series.sort_values("avg_daily_sales", ascending=False)["unique_id"].tolist()
    uid = st.selectbox("SKU-store", ids, key="forecast_series")
    hist = data.history[data.history["unique_id"] == uid].set_index("ds")["y"].rename("actual")
    fc = data.forecasts[data.forecasts["unique_id"] == uid]
    lines = [hist] + [
        g.set_index("ds")["yhat"].rename(f"{model} forecast") for model, g in fc.groupby("model")
    ]
    st.line_chart(pd.concat(lines, axis=1))
    meta = data.series.set_index("unique_id").loc[uid]
    st.caption(
        f"{meta['cat_id']} · {meta['demand_class']} demand · "
        f"{meta['avg_daily_sales']:.1f} units/day · champion MAE on test {meta['champion_mae']:.2f}"
    )


def inventory_tab(data: RunData) -> None:
    """Per-series recommendations (safety stock, reorder point, order quantity)."""
    table = data.policies.merge(data.series[["unique_id", "cat_id", "store_id"]], on="unique_id")
    cats = ["All", *sorted(table["cat_id"].unique())]
    cat = st.selectbox("Category", cats, key="inventory_category")
    if cat != "All":
        table = table[table["cat_id"] == cat]
    cols = ["unique_id", "mean_daily_forecast", "forecast_bias", "sigma_protection",
            "safety_stock", "reorder_point", "order_quantity"]  # fmt: skip
    st.dataframe(table[cols].round(2), hide_index=True)


def what_if_tab(data: RunData) -> None:
    """Re-simulate the whole portfolio with different assumptions."""
    p = data.params
    c1, c2 = st.columns(2)
    service = c1.slider("Target service level", 0.80, 0.99, p.service_level, 0.01)
    lead = c1.slider("Lead time (days)", 1, 21, p.lead_time_days)
    order_cost = c2.slider("Cost per order ($)", 0.0, 50.0, p.ordering_cost, 1.0)
    holding = c2.slider("Holding cost (per year)", 0.05, 1.0, p.holding_cost_rate, 0.05)
    params = replace_params(
        p, service_level=service, lead_time_days=lead, ordering_cost=order_cost,
        holding_cost_rate=holding,
    )  # fmt: skip
    result = run_scenario(data.inputs, params)
    port = labelled(result.portfolio)
    fd, ma = port.loc[POLICY_LABELS["forecast_driven"]], port.loc[POLICY_LABELS["ma_safety_stock"]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Saving vs rule of thumb", f"{fd['savings_vs_baseline_pct']:.1f}%")
    m2.metric(
        "Extra saving from ML",
        f"{100 * (ma['total_cost'] - fd['total_cost']) / ma['total_cost']:.2f}%",
    )
    m3.metric("Forecast-driven fill rate", f"{100 * fd['fill_rate']:.1f}%")
    st.bar_chart(port[["holding_cost", "ordering_cost", "stockout_cost"]], horizontal=True)


def models_tab(data: RunData) -> None:
    """Leaderboard and global SHAP importance."""
    split = st.radio("Split", ["test", "validation"], horizontal=True, key="models_split")
    board = data.leaderboard[data.leaderboard["split"] == split]
    st.dataframe(
        board[["rank", "model", "wrmsse", "mae", "bias", "coverage"]].round(4), hide_index=True
    )
    if data.shap_global is not None:
        st.subheader("Global SHAP importance")
        st.bar_chart(data.shap_global.head(15).set_index("feature")["mean_abs_shap"])


def main() -> None:
    """Render the app."""
    st.set_page_config(page_title="Demand & Inventory Engine", layout="wide")
    st.title("Demand Forecasting & Inventory Optimisation")
    artifacts_dir = get_settings().paths.artifacts_dir
    try:
        run = latest_run(artifacts_dir)
    except ArtifactNotFoundError as exc:
        st.warning(f"{exc}. Run `make train` first.")
        return
    data = load_run(str(run.root))
    m = data.manifest
    st.caption(
        f"Run {m['run_id']} · M5 {m['state']} · {m['n_series']:,} series · "
        f"test period from {m['test_start']} · champion: {m['champion']}"
    )
    tabs = st.tabs(["Overview", "Forecasts", "Inventory", "What-if", "Models"])
    for tab, render in zip(
        tabs, (overview_tab, forecasts_tab, inventory_tab, what_if_tab, models_tab), strict=True
    ):
        with tab:
            render(data)


main()
