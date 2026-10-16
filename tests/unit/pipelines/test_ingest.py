import pytest

from demand_engine.config import Settings
from demand_engine.data.synthetic import write_synthetic_m5
from demand_engine.data.validation import validate_long
from demand_engine.pipelines.ingest import build_demand_table, load_demand_table, run_ingest


def test_run_ingest_writes_validated_subset(test_settings: Settings) -> None:
    write_synthetic_m5(test_settings.paths.raw_dir, n_items=12, n_days=420, seed=4)
    path = run_ingest(test_settings)
    assert path.is_file()
    df = load_demand_table(test_settings)
    validate_long(df)
    assert df["item_id"].nunique() <= test_settings.data.n_items
    assert set(df["state_id"]) == {"CA"}


def test_build_is_deterministic(test_settings: Settings) -> None:
    write_synthetic_m5(test_settings.paths.raw_dir, n_items=12, n_days=420, seed=4)
    a = build_demand_table(test_settings)
    b = build_demand_table(test_settings)
    assert a.equals(b)


def test_load_without_ingest_explains(test_settings: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="ingest"):
        load_demand_table(test_settings)


def test_all_series_too_short(test_settings: Settings) -> None:
    write_synthetic_m5(test_settings.paths.raw_dir, n_items=3, n_days=100, seed=4)
    with pytest.raises(ValueError, match="no series left"):
        build_demand_table(test_settings)
