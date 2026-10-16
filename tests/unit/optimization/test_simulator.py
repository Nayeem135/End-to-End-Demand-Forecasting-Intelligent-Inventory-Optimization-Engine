import numpy as np
import pytest

from demand_engine.optimization.simulator import CostModel, Policy, simulate

COSTS = CostModel(
    holding_per_unit_day=np.array([1.0]), stockout_per_unit=np.array([10.0]), ordering_cost=5.0
)


def _const_policy(kind: str, s: float, q: float, t: int) -> Policy:
    return Policy("p", kind, np.full((1, t), s), np.full((1, t), q))  # type: ignore[arg-type]


def test_hand_traced_sq_run() -> None:
    """L=2, s=3, Q=5, start 5 units, demand 2/day - verified by hand."""
    demand = np.full((1, 5), 2.0)
    out = simulate(
        demand,
        _const_policy("sQ", 3, 5, 5),
        COSTS,
        lead_time=2,
        review_period=1,
        initial_on_hand=np.array([5.0]),
    )
    # day0: oh 5->3, IP 3<=3 -> order 5 (arrives day2); day1: 3->1, IP 6; day2: +5 -> 6 -> 4, IP 4
    # day3: 4->2, IP 2<=3 -> order 5 (arrives day5); day4: 2->0, IP 5
    assert out.on_hand[0].tolist() == [3, 1, 4, 2, 0]
    assert out.orders[0].tolist() == [5, 0, 0, 5, 0]
    assert out.lost.sum() == 0
    row = out.summary.iloc[0]
    assert row["holding_cost"] == 10.0
    assert row["ordering_cost"] == 10.0
    assert row["fill_rate"] == 1.0


def test_lost_sales_when_stock_runs_out() -> None:
    demand = np.array([[4.0, 4.0, 4.0]])
    out = simulate(demand, _const_policy("sQ", -1, 1, 3), COSTS, 1, 1, np.array([6.0]))
    assert out.sold[0].tolist() == [4, 2, 0]
    assert out.lost[0].tolist() == [0, 2, 4]
    assert out.summary.iloc[0]["stockout_cost"] == 60.0
    assert out.summary.iloc[0]["fill_rate"] == pytest.approx(0.5)
    assert out.summary.iloc[0]["stockout_days"] == 2


def test_order_up_to_policy() -> None:
    demand = np.full((1, 3), 3.0)
    out = simulate(demand, _const_policy("sS", 5, 12, 3), COSTS, 1, 1, np.array([7.0]))
    # day0: 7->4, IP 4<=5 -> order 12-4=8
    assert out.orders[0, 0] == 8


def test_multiple_q_lifts_position_above_s() -> None:
    demand = np.zeros((1, 1))
    out = simulate(demand, _const_policy("sQ", 10, 4, 1), COSTS, 1, 1, np.array([0.0]))
    assert out.orders[0, 0] == 12  # 3 x Q so that 0 + 12 > 10


def test_review_period_restricts_ordering_days() -> None:
    demand = np.full((1, 6), 1.0)
    out = simulate(demand, _const_policy("sQ", 100, 1, 6), COSTS, 1, 3, np.array([0.0]))
    assert np.flatnonzero(out.orders[0]).tolist() == [0, 3]


@pytest.mark.parametrize("seed", range(5))
def test_inventory_conservation_invariant(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, t = 20, 40
    demand = rng.poisson(3.0, size=(n, t)).astype(float)
    policy = Policy("p", "sQ", rng.uniform(0, 20, (n, t)), rng.uniform(1, 30, (n, t)))
    costs = CostModel(np.ones(n), np.ones(n), 2.0)
    init = rng.integers(0, 30, n).astype(float)
    out = simulate(demand, policy, costs, lead_time=4, review_period=1, initial_on_hand=init)
    arrived = out.orders[:, : t - 4].sum(axis=1)
    assert out.on_hand[:, -1] == pytest.approx(init + arrived - out.sold.sum(axis=1))
    assert (out.on_hand >= 0).all()
    assert out.sold + out.lost == pytest.approx(demand)
    assert (out.summary[["holding_cost", "ordering_cost", "stockout_cost"]] >= 0).all().all()
    assert out.summary["fill_rate"].between(0, 1).all()


def test_zero_demand_fill_rate_is_one() -> None:
    out = simulate(np.zeros((1, 3)), _const_policy("sQ", 0, 1, 3), COSTS, 1, 1, np.array([0.0]))
    assert out.summary.iloc[0]["fill_rate"] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"lead_time": 0}, ">= 1"),
        ({"initial_on_hand": np.array([1.0, 2.0])}, "initial_on_hand"),
        ({"demand": np.array([1.0, 2.0])}, "2-D"),
        ({"demand": -np.ones((1, 3))}, "non-negative"),
    ],
)
def test_validation(kwargs: dict[str, object], msg: str) -> None:
    args: dict[str, object] = {
        "demand": np.ones((1, 3)),
        "policy": _const_policy("sQ", 1, 1, 3),
        "costs": COSTS,
        "lead_time": 1,
        "review_period": 1,
        "initial_on_hand": np.array([1.0]),
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        simulate(**args)  # type: ignore[arg-type]


def test_policy_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        simulate(np.ones((1, 4)), _const_policy("sQ", 1, 1, 3), COSTS, 1, 1, np.array([1.0]))
