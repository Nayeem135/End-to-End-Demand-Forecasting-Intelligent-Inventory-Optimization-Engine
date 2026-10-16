import numpy as np
import pytest

from demand_engine.optimization.policies import (
    forecast_policy,
    forward_sum,
    order_quantity,
    rule_of_thumb_policy,
    window_mean,
)

ONE_WINDOW = np.zeros(4, dtype="int64")


def test_forward_sum_within_window_and_seasonal_padding() -> None:
    fc = np.array([[1.0, 2.0, 3.0, 4.0]])
    out = forward_sum(fc, 2, ONE_WINDOW)
    # padding cycles the window's last (up to 7) days: 1,2,3,4,1,2,...
    assert out[0].tolist() == pytest.approx([5.0, 7.0, 5.0, 3.0])


def test_forward_sum_never_reads_next_window() -> None:
    fc = np.array([[1.0, 1.0, 100.0, 100.0]])
    window_id = np.array([0, 0, 1, 1])
    out = forward_sum(fc, 2, window_id)
    # day 1 belongs to window 0: it must not see window 1's forecast of 100
    assert out[0, 1] == pytest.approx(2.0)
    assert out[0, 2] == pytest.approx(200.0)


def test_forward_sum_validation() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        forward_sum(np.ones((1, 2)), 0, np.zeros(2, dtype="int64"))
    with pytest.raises(ValueError, match="non-decreasing"):
        forward_sum(np.ones((1, 2)), 1, np.array([1, 0]))


def test_window_mean() -> None:
    fc = np.array([[1.0, 3.0, 10.0, 20.0]])
    out = window_mean(fc, np.array([0, 0, 1, 1]))
    assert out[0].tolist() == [2.0, 2.0, 15.0, 15.0]


def test_rule_of_thumb_levels() -> None:
    rate = np.array([[2.0, 2.0], [0.0, 0.0]])
    p = rule_of_thumb_policy(rate, lead_time=7, cover_days=14)
    assert p.kind == "sS"
    assert p.reorder_point[:, 0].tolist() == [14.0, 0.0]
    assert p.quantity[:, 0].tolist() == [28.0, 1.0]  # S > s enforced for zero-demand item


def test_order_quantity_at_least_one() -> None:
    q = order_quantity(np.array([0.0, 10.0]), 10.0, np.array([1.0, 1.0]))
    assert q[0] == 1.0
    assert q[1] == pytest.approx(np.sqrt(2 * 3650 * 10))


def test_forecast_policy_reorder_point_and_constant_lot_size() -> None:
    fc = np.array([[1.0] * 5 + [10.0] * 5])
    window_id = np.array([0] * 5 + [1] * 5)
    p = forecast_policy(
        "x",
        fc,
        window_id,
        bias=np.array([1.0]),
        sigma=np.array([0.0]),
        protection=2,
        service_level=0.95,
        lot_size=np.array([7.0]),
    )
    assert p.kind == "sQ"
    assert p.reorder_point[0, 0] == pytest.approx(2.0 + 1.0)  # forecast + bias
    assert p.reorder_point[0, 5] == pytest.approx(20.0 + 1.0)
    assert np.unique(p.quantity).tolist() == [7.0]


def test_forecast_policy_safety_stock_scales_with_sigma() -> None:
    fc = np.ones((1, 4))
    kwargs = {"window_id": ONE_WINDOW, "bias": np.zeros(1), "protection": 1, "lot_size": np.ones(1)}
    low = forecast_policy("x", fc, sigma=np.array([1.0]), service_level=0.9, **kwargs)
    high = forecast_policy("x", fc, sigma=np.array([1.0]), service_level=0.99, **kwargs)
    assert (high.reorder_point > low.reorder_point).all()
