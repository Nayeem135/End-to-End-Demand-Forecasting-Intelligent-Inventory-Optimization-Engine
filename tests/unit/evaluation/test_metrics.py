import numpy as np
import pandas as pd
import pytest

from demand_engine.evaluation import metrics as m

Y = np.array([1.0, 2.0, 3.0, 4.0])
YHAT = np.array([2.0, 2.0, 2.0, 2.0])


def test_point_metrics_hand_computed() -> None:
    assert m.mae(Y, YHAT) == pytest.approx(1.0)
    assert m.rmse(Y, YHAT) == pytest.approx(np.sqrt(1.5))
    assert m.bias(Y, YHAT) == pytest.approx(-0.5)


def test_rmsse_scale() -> None:
    train = np.array([0.0, 1.0, 0.0, 1.0])  # naive diffs^2 = 1
    assert m.naive_scale(train) == pytest.approx(1.0)
    assert m.rmsse(Y, YHAT, train) == pytest.approx(np.sqrt(1.5))
    assert np.isnan(m.rmsse(Y, YHAT, np.array([5.0, 5.0, 5.0])))
    assert np.isnan(m.naive_scale(np.array([1.0])))


def test_pinball_and_coverage() -> None:
    # tau=0.9, q=2: errors y-q = [-1,0,1,2] -> losses [0.1,0,0.9,1.8]
    assert m.pinball(Y, YHAT, 0.9) == pytest.approx((0.1 + 0 + 0.9 + 1.8) / 4)
    assert m.coverage(Y, Y - 0.5, Y + 0.5) == 1.0
    assert m.coverage(Y, YHAT, YHAT) == 0.25


def _history() -> pd.DataFrame:
    ds = pd.date_range("2020-01-01", periods=30)
    return pd.concat(
        [
            pd.DataFrame(
                {"unique_id": "a", "ds": ds, "y": np.tile([0.0, 2.0], 15), "sell_price": 1.0}
            ),
            pd.DataFrame(
                {"unique_id": "b", "ds": ds, "y": np.tile([0.0, 2.0], 15), "sell_price": 3.0}
            ),
        ]
    )


def test_revenue_weights() -> None:
    w = m.revenue_weights(_history(), days=28)
    assert w.sum() == pytest.approx(1.0)
    assert w["b"] == pytest.approx(0.75)


def test_revenue_weights_zero_sales_uniform() -> None:
    hist = _history().assign(y=0.0)
    assert m.revenue_weights(hist).tolist() == [0.5, 0.5]


def test_per_series_and_summary() -> None:
    hist = _history()
    preds = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": pd.to_datetime(["2020-02-01", "2020-02-02"] * 2),
            "y": [1.0, 1.0, 2.0, 2.0],
            "yhat": [1.0, 1.0, 1.0, 1.0],
            "yhat_lo": [0.0, 0.0, 0.0, 0.0],
            "yhat_hi": [2.0, 2.0, 1.5, 1.5],
        }
    )
    per = m.per_series_metrics(preds, hist, level=80)
    a, b = per.set_index("unique_id").loc["a"], per.set_index("unique_id").loc["b"]
    assert a["mae"] == 0.0
    assert b["mae"] == 1.0
    assert b["coverage"] == 0.0
    assert b["rmsse"] == pytest.approx(1.0 / 2.0)  # naive scale = 4
    summary = m.summarize(per, m.revenue_weights(hist))
    assert summary["wrmsse"] == pytest.approx(0.75 * 0.5)
    assert set(summary) == {"wrmsse", "rmsse", "mae", "rmse", "bias", "coverage", "pinball"}
