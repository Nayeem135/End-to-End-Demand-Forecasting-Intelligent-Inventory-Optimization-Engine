import numpy as np
import pytest

from demand_engine.optimization import formulas as f


def test_protection_days() -> None:
    assert f.protection_days(7, 1) == 7
    assert f.protection_days(5, 7) == 11
    with pytest.raises(ValueError, match=">= 1"):
        f.protection_days(0, 1)


def test_z_score_textbook_values() -> None:
    assert f.z_score(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert f.z_score(0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="service_level"):
        f.z_score(1.0)


def test_safety_stock_and_rop() -> None:
    ss = f.safety_stock(np.array([10.0, 0.0]), 0.95)
    assert ss == pytest.approx([16.449, 0.0], abs=1e-3)
    assert f.reorder_point(np.array([70.0, 5.0]), ss) == pytest.approx([86.449, 5.0], abs=1e-3)
    with pytest.raises(ValueError, match="non-negative"):
        f.safety_stock(np.array([-1.0]), 0.9)


def test_eoq_textbook_example() -> None:
    # D=1000/yr, K=$10, H=$0.5/unit/yr -> sqrt(2*1000*10/0.5) = 200
    q = f.eoq(np.array([1000.0, 0.0]), 10.0, np.array([0.5, 0.5]))
    assert q == pytest.approx([200.0, 0.0])
    with pytest.raises(ValueError, match="positive"):
        f.eoq(np.array([1.0]), 1.0, np.array([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        f.eoq(np.array([-1.0]), 1.0, np.array([1.0]))


def test_daily_holding_cost() -> None:
    assert f.daily_holding_cost(np.array([365.0]), 0.2) == pytest.approx([0.2])


def test_critical_ratio() -> None:
    assert f.critical_ratio(9.0, 1.0) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="costs"):
        f.critical_ratio(0.0, 0.0)
