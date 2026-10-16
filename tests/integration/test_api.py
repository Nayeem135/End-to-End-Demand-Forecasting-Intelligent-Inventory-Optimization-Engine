import pytest
from fastapi.testclient import TestClient

from demand_engine.pipelines.artifacts import RunArtifacts

pytestmark = pytest.mark.integration


def _first_series(client: TestClient) -> str:
    page = client.get("/api/series", params={"page_size": 1}).json()
    return str(page["items"][0]["unique_id"])


def test_health(client: TestClient, published: RunArtifacts) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["latest_run"] == published.run_id


def test_overview_matches_manifest(client: TestClient, published: RunArtifacts) -> None:
    body = client.get("/api/overview").json()
    manifest = published.load_json("manifest")
    assert body["run_id"] == published.run_id
    assert body["champion"] == manifest["champion"]
    assert body["headline"]["savings_vs_rule_pct"] == pytest.approx(
        manifest["headline"]["savings_vs_rule_pct"]
    )
    assert (
        body["inventory"]["lead_time_days"] == manifest["settings"]["inventory"]["lead_time_days"]
    )
    assert client.get("/api/runs").json()[0]["run_id"] == published.run_id


def test_unknown_run_is_404(client: TestClient) -> None:
    response = client.get("/api/overview", params={"run_id": "nope"})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.parametrize("split", ["validation", "test"])
def test_leaderboard(client: TestClient, split: str) -> None:
    rows = client.get("/api/models/leaderboard", params={"split": split}).json()
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert sum(r["is_champion"] for r in rows) == 1
    assert {r["model"] for r in rows} == {"naive", "moving_average", "lightgbm"}


def test_leaderboard_rejects_bad_split(client: TestClient) -> None:
    assert client.get("/api/models/leaderboard", params={"split": "train"}).status_code == 422


def test_series_paging_filtering_sorting(client: TestClient) -> None:
    page1 = client.get("/api/series", params={"page_size": 5}).json()
    page2 = client.get("/api/series", params={"page_size": 5, "page": 2}).json()
    assert page1["total"] == page2["total"] > 5
    assert {s["unique_id"] for s in page1["items"]}.isdisjoint(
        {s["unique_id"] for s in page2["items"]}
    )
    sales = [s["avg_daily_sales"] for s in page1["items"]]
    assert sales == sorted(sales, reverse=True)

    store = client.get("/api/filters").json()["stores"][0]
    filtered = client.get("/api/series", params={"store_id": store, "page_size": 200}).json()
    assert filtered["items"]
    assert {s["store_id"] for s in filtered["items"]} == {store}

    uid = page1["items"][0]["unique_id"]
    found = client.get("/api/series", params={"search": uid[:9]}).json()
    assert any(s["unique_id"] == uid for s in found["items"])


def test_series_page_size_limit(client: TestClient) -> None:
    assert client.get("/api/series", params={"page_size": 1000}).status_code == 422


def test_series_detail_forecast_simulation_explanation(client: TestClient) -> None:
    uid = _first_series(client)
    detail = client.get(f"/api/series/{uid}").json()
    assert detail["series"]["unique_id"] == uid
    assert detail["policy"]["order_quantity"] >= 1
    assert {r["policy"] for r in detail["results"]} == {
        "rule_of_thumb",
        "ma_safety_stock",
        "forecast_driven",
    }

    fc = client.get(f"/api/series/{uid}/forecast").json()
    assert fc["history"]
    assert set(fc["forecasts"]) == {fc["champion"], fc["baseline_model"]}
    first = fc["forecasts"][fc["champion"]][0]
    assert first["yhat_lo"] <= first["yhat"] <= first["yhat_hi"]
    assert first["ds"] >= fc["test_start"]

    sim = client.get(f"/api/series/{uid}/simulation").json()
    assert len(sim["trajectories"]) == 3
    days = sim["trajectories"][0]["days"]
    assert len(days) == 28  # 2 test windows x 14-day horizon
    assert all(d["on_hand"] >= 0 for d in days)
    # trajectories must agree with the per-series results stored by the pipeline
    stored = {r["policy"]: r["total_cost"] for r in detail["results"]}
    for t in sim["trajectories"]:
        assert t["summary"]["total_cost"] == pytest.approx(stored[t["policy"]], rel=1e-9)

    shap = client.get(f"/api/series/{uid}/explanation").json()
    assert shap
    assert abs(shap[0]["shap_value"]) >= abs(shap[-1]["shap_value"])


def test_unknown_series_is_404(client: TestClient) -> None:
    for path in ("", "/forecast", "/simulation", "/explanation"):
        assert client.get(f"/api/series/NOT_A_SERIES{path}").status_code == 404


def test_series_id_validation(client: TestClient) -> None:
    assert client.get("/api/series/bad%20id").status_code == 422


def test_global_explanation_and_policies(client: TestClient) -> None:
    imp = client.get("/api/explainability/global", params={"limit": 5}).json()
    assert len(imp) == 5
    assert imp[0]["mean_abs_shap"] >= imp[-1]["mean_abs_shap"]
    pol = client.get("/api/inventory/policies", params={"page_size": 3}).json()
    assert len(pol["items"]) == 3
    assert {"safety_stock", "reorder_point", "item_id", "store_id"} <= set(pol["items"][0])


def test_portfolio_and_sensitivity(client: TestClient) -> None:
    port = client.get("/api/simulation/portfolio").json()
    assert {p["policy"] for p in port} == {"rule_of_thumb", "ma_safety_stock", "forecast_driven"}
    sens = client.get("/api/simulation/sensitivity").json()
    assert len({s["service_level"] for s in sens}) == 6


def test_what_if_default_reproduces_pipeline_exactly(client: TestClient) -> None:
    """The API's live re-simulation must match what the pipeline computed and stored."""
    stored = {p["policy"]: p for p in client.get("/api/simulation/portfolio").json()}
    live = client.post("/api/simulation/what-if", json={}).json()
    for p in live["portfolio"]:
        assert p["total_cost"] == pytest.approx(stored[p["policy"]]["total_cost"], rel=1e-9)
        assert p["fill_rate"] == pytest.approx(stored[p["policy"]]["fill_rate"], rel=1e-9)


def test_what_if_changes_parameters(client: TestClient) -> None:
    base = client.post("/api/simulation/what-if", json={}).json()
    longer = client.post("/api/simulation/what-if", json={"lead_time_days": 14}).json()
    assert longer["params"]["lead_time_days"] == 14
    assert longer["params"]["service_level"] == base["params"]["service_level"]
    cost = {p["policy"]: p["total_cost"] for p in base["portfolio"]}
    cost_long = {p["policy"]: p["total_cost"] for p in longer["portfolio"]}
    assert cost != cost_long


@pytest.mark.parametrize(
    "body",
    [{"service_level": 1.2}, {"lead_time_days": 0}, {"ordering_cost": -1}, {"lead_time_days": "x"}],
)
def test_what_if_validation(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/api/simulation/what-if", json=body).status_code == 422


def test_openapi_documents_all_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/simulation/what-if" in paths
    assert "/api/series/{unique_id}/forecast" in paths
