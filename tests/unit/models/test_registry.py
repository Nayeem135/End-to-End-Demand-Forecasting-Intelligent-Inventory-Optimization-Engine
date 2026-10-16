import pytest

from demand_engine.config import Settings
from demand_engine.models.base import Forecaster
from demand_engine.models.registry import model_factory


@pytest.mark.parametrize(
    "name", ["naive", "seasonal_naive", "moving_average", "croston", "tsb", "ets", "lightgbm"]
)
def test_every_registered_model_builds(name: str) -> None:
    model = model_factory(name, Settings())()  # type: ignore[arg-type]
    assert isinstance(model, Forecaster)
    assert model.name == name


def test_factory_returns_fresh_instances() -> None:
    factory = model_factory("naive", Settings())
    assert factory() is not factory()


def test_unknown_model() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        model_factory("prophet", Settings())  # type: ignore[arg-type]
