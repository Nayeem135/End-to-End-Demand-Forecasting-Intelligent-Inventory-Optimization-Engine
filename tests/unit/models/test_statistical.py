import pandas as pd
import pytest

from demand_engine.data.splits import split_at
from demand_engine.models import statistical as st
from demand_engine.models.base import Forecaster

H = 14


@pytest.fixture
def split(small_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = small_long["ds"].max() - pd.Timedelta(days=H)
    return split_at(small_long, cutoff, H)


@pytest.mark.parametrize(
    "factory",
    [
        st.make_naive,
        st.make_seasonal_naive,
        st.make_moving_average,
        st.make_croston,
        st.make_tsb,
        st.make_auto_ets,
    ],
)
def test_models_satisfy_contract(factory: object, split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, window = split
    model = factory()  # type: ignore[operator]
    assert isinstance(model, Forecaster)
    out = model.fit(train).predict(window)
    assert len(out) == len(window)
    assert out[["yhat", "yhat_lo", "yhat_hi"]].notna().all().all()
    assert (out["yhat_lo"] <= out["yhat_hi"]).all()


def test_seasonal_naive_repeats_last_week(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, window = split
    out = st.make_seasonal_naive(7).fit(train).predict(window)
    uid = train["unique_id"].iloc[0]
    last_week = train[train["unique_id"] == uid]["y"].tail(7).to_numpy()
    first_week = out[out["unique_id"] == uid]["yhat"].head(7).to_numpy()
    assert list(first_week) == pytest.approx(list(last_week))


def test_moving_average_is_flat_mean(split: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, window = split
    out = st.make_moving_average(14).fit(train).predict(window)
    uid = train["unique_id"].iloc[0]
    expected = train[train["unique_id"] == uid]["y"].tail(14).mean()
    assert out[out["unique_id"] == uid]["yhat"].to_numpy() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: st.make_seasonal_naive(7), {"season_length": 7}),
        (lambda: st.make_moving_average(28), {"window": 28}),
        (lambda: st.make_auto_ets(7), {"ets_model": "ANA", "intervals": "native"}),
        (st.make_tsb, {"alpha_d": 0.1, "alpha_p": 0.1}),
    ],
)
def test_params_include_hyperparameters(factory: object, expected: dict[str, object]) -> None:
    params = factory().params()  # type: ignore[operator]
    assert expected.items() <= params.items()
    assert "interval_level" in params
