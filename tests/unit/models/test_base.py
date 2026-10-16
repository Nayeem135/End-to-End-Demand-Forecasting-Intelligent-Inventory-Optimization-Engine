import numpy as np
import pandas as pd
import pytest

from demand_engine.models.base import ForecastContractError, Forecaster, window_length


class _Echo(Forecaster):
    """Test double: returns a configurable forecast and records what it was given."""

    name = "echo"

    def __init__(self, value: float = 1.0, drop_rows: int = 0) -> None:
        super().__init__()
        self.value = value
        self.drop_rows = drop_rows
        self.seen_columns: list[str] = []

    def _fit(self, history: pd.DataFrame) -> None:
        pass

    def _predict(self, future: pd.DataFrame) -> pd.DataFrame:
        self.seen_columns = list(future.columns)
        out = future[["unique_id", "ds"]].copy()
        out["yhat"] = self.value
        out["yhat_lo"] = self.value + 1  # deliberately inverted
        out["yhat_hi"] = self.value - 1
        return out.iloc[self.drop_rows :]


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2020-01-01", periods=5), "y": 1.0})
    fut = pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2020-01-06", periods=3), "y": 99.0})
    return hist, fut


def test_predict_never_sees_target() -> None:
    hist, fut = _frames()
    model = _Echo().fit(hist)
    model.predict(fut)
    assert "y" not in model.seen_columns


def test_contract_clips_and_orders_interval() -> None:
    hist, fut = _frames()
    out = _Echo(value=-5.0).fit(hist).predict(fut)
    assert (out[["yhat", "yhat_lo", "yhat_hi"]] >= 0).all().all()
    assert (out["yhat_lo"] <= out["yhat"]).all()
    assert (out["yhat"] <= out["yhat_hi"]).all()


def test_missing_rows_raise() -> None:
    hist, fut = _frames()
    with pytest.raises(ForecastContractError, match="missing"):
        _Echo(drop_rows=1).fit(hist).predict(fut)


def test_predict_before_fit_raises() -> None:
    _, fut = _frames()
    with pytest.raises(RuntimeError, match="fitted"):
        _Echo().predict(fut)


def test_fit_rejects_empty_and_nan() -> None:
    hist, _ = _frames()
    with pytest.raises(ValueError, match="empty"):
        _Echo().fit(hist.iloc[0:0])
    bad = hist.copy()
    bad.loc[0, "y"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        _Echo().fit(bad)


def test_window_length_requires_equal_lengths() -> None:
    fut = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        }
    )
    with pytest.raises(ValueError, match="same number"):
        window_length(fut)
    assert window_length(fut.iloc[:2]) == 2
