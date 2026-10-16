import numpy as np
import pandas as pd
import pytest

from demand_engine.config import InventorySettings
from demand_engine.optimization.scenario import (
    POLICY_NAMES,
    SimulationInputs,
    build_inputs,
    replace_params,
    run_scenario,
    sensitivity,
    to_matrix,
    window_ids,
)

IDS = ["a", "b"]
DATES = pd.date_range("2020-03-01", periods=28)


def _long(values: dict[str, float], col: str = "y") -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({"unique_id": uid, "ds": DATES, col: v, "sell_price": 2.0})
            for uid, v in values.items()
        ]
    )


def _val(bias: float) -> pd.DataFrame:
    ds = pd.date_range("2020-02-01", periods=28)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "cutoff": pd.Timestamp("2020-01-31"),
                    "ds": ds,
                    "y": 3.0,
                    "yhat": 3.0 + bias,
                }
            )
            for uid in IDS
        ]
    )


@pytest.fixture
def inputs() -> SimulationInputs:
    history = pd.concat(
        [
            pd.DataFrame(
                {"unique_id": uid, "ds": pd.date_range("2020-01-01", periods=60), "y": 3.0}
            )
            for uid in IDS
        ]
    )
    actuals = _long({"a": 3.0, "b": 3.0})
    fc = _long({"a": 3.0, "b": 3.0}, col="yhat").assign(cutoff=pd.Timestamp("2020-02-29"))
    return build_inputs(history, actuals, fc, fc, _val(0.5), _val(1.0), "lightgbm")


def test_to_matrix_order_and_gaps() -> None:
    df = _long({"b": 2.0, "a": 1.0})
    m = to_matrix(df, "y", IDS, DATES)
    assert m[:, 0].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="missing"):
        to_matrix(df.iloc[1:], "y", IDS, DATES)


def test_build_inputs_alignment(inputs: SimulationInputs) -> None:
    assert inputs.unique_ids == IDS
    assert inputs.demand.shape == (2, 28)
    assert inputs.hist_mean.tolist() == [3.0, 3.0]
    assert inputs.long_run_mean.tolist() == [3.0, 3.0]
    assert inputs.price.tolist() == [2.0, 2.0]
    assert inputs.window_id.tolist() == [0] * 28


def test_run_scenario_outputs(inputs: SimulationInputs) -> None:
    result = run_scenario(inputs, InventorySettings())
    assert set(result.outputs) == set(POLICY_NAMES)
    port = result.portfolio.set_index("policy")
    assert port.loc["rule_of_thumb", "savings_vs_baseline_pct"] == 0.0
    assert (port["fill_rate"].between(0, 1)).all()
    assert len(result.policy_table) == 2
    table = result.policy_table.set_index("unique_id")
    # champion over-forecasts by 0.5/day -> 7-day bias of -3.5 and no spread
    assert table.loc["a", "forecast_bias"] == pytest.approx(-3.5)
    assert table.loc["a", "sigma_protection"] == pytest.approx(0.0)
    assert table.loc["a", "safety_stock"] == pytest.approx(-3.5)
    assert table.loc["a", "reorder_point"] == pytest.approx(7 * 3.0 - 3.5)


def test_higher_service_level_never_reduces_safety_stock(inputs: SimulationInputs) -> None:
    low = run_scenario(inputs, replace_params(InventorySettings(), service_level=0.8))
    high = run_scenario(inputs, replace_params(InventorySettings(), service_level=0.99))
    assert (high.policy_table["safety_stock"] >= low.policy_table["safety_stock"]).all()


def test_sensitivity_grid(inputs: SimulationInputs) -> None:
    sens = sensitivity(inputs, InventorySettings(), [0.9, 0.95])
    assert len(sens) == 2 * len(POLICY_NAMES)
    assert set(sens["service_level"]) == {0.9, 0.95}


def test_replace_params_validates() -> None:
    assert replace_params(InventorySettings(), lead_time_days=3).lead_time_days == 3
    with pytest.raises(ValueError, match="service_level"):
        replace_params(InventorySettings(), service_level=1.5)


def test_deterministic(inputs: SimulationInputs) -> None:
    a = run_scenario(inputs, InventorySettings()).portfolio
    b = run_scenario(inputs, InventorySettings()).portfolio
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a["total_cost"]).all()


def test_window_ids_map_dates_to_cutoffs() -> None:
    fc = pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=4),
            "cutoff": pd.to_datetime(["2019-12-31"] * 2 + ["2020-01-02"] * 2),
        }
    )
    assert window_ids(fc, pd.DatetimeIndex(fc["ds"])).tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="no forecast"):
        window_ids(fc, pd.date_range("2020-01-01", periods=5))
    dup = pd.concat([fc, fc.assign(cutoff=pd.Timestamp("2019-12-01"))])
    with pytest.raises(ValueError, match="more than one"):
        window_ids(dup, pd.DatetimeIndex(fc["ds"]))
