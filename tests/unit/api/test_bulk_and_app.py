from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select

from demand_engine.api.db.bulk import bulk_insert, to_records
from demand_engine.api.db.models import Base, DemandHistory
from demand_engine.api.main import create_app
from demand_engine.config import ApiSettings, DatabaseSettings, Settings
from demand_engine.types import cell

TABLE = Base.metadata.tables["demand_history"]


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": "r1",
            "unique_id": ["a", "a"],
            "ds": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "y": [1.0, np.nan],
            "sell_price": np.array([2.5, 2.5], dtype="float32"),
            "extra": [0, 0],
        }
    )


def test_to_records_converts_types_and_drops_extra_columns() -> None:
    records = to_records(_history(), TABLE)
    assert set(records[0]) == {c.name for c in TABLE.columns}
    assert records[0]["ds"].isoformat() == "2020-01-01"
    assert records[1]["y"] is None
    assert isinstance(records[0]["sell_price"], float)


def test_to_records_missing_columns() -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        to_records(_history().drop(columns="y"), TABLE)


def test_bulk_insert_generic_dialect(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    Base.metadata.create_all(engine, tables=[Base.metadata.tables["pipeline_runs"], TABLE])
    with engine.begin() as conn:
        assert bulk_insert(conn, TABLE, _history().iloc[:1]) == 1
        assert bulk_insert(conn, TABLE, _history().iloc[0:0]) == 0
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(DemandHistory)).scalar_one() == 1


def test_health_reports_degraded_without_database() -> None:
    settings = Settings(
        database=DatabaseSettings(
            url="postgresql+psycopg://x:y@127.0.0.1:1/none", connect_timeout=1
        )
    )
    client = TestClient(create_app(settings))
    body = client.get("/api/health").json()
    assert body == {
        "status": "degraded",
        "database": False,
        "latest_run": None,
        "version": body["version"],
    }


def test_dashboard_is_served_with_spa_fallback(tmp_path: Path) -> None:
    site = tmp_path / "dist"
    (site / "assets").mkdir(parents=True)
    (site / "index.html").write_text("<h1>dashboard</h1>", encoding="utf-8")
    (site / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    settings = Settings(
        database=DatabaseSettings(url="sqlite://"), api=ApiSettings(static_dir=site)
    )
    client = TestClient(create_app(settings))
    assert "dashboard" in client.get("/").text
    assert "dashboard" in client.get("/what-if").text  # client-side route survives refresh
    assert "console.log" in client.get("/assets/app.js").text
    assert "secret" not in client.get("/..%2Fsecret.txt").text  # no path traversal
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/docs").status_code == 200


def test_cell_helper() -> None:
    df = pd.DataFrame({"v": [1.5], "s": ["x"]}, index=["r"])
    assert cell(df, "r", "v") == 1.5
    with pytest.raises(TypeError, match="not numeric"):
        cell(df, "r", "s")
    with pytest.raises(KeyError):
        cell(df, "missing", "v")
