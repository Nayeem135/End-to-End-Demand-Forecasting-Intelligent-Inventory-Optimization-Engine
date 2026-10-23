"""Run the Streamlit app headlessly against a real pipeline run (streamlit.testing)."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from demand_engine.config import get_settings
from demand_engine.pipelines.artifacts import RunArtifacts

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


@pytest.fixture
def artifacts_env(pipeline_run: RunArtifacts, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("PATHS__ARTIFACTS_DIR", str(pipeline_run.root.parent.parent))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.usefixtures("artifacts_env")
def test_app_renders_every_tab_without_errors(pipeline_run: RunArtifacts) -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert pipeline_run.run_id in at.caption[0].value
    labels = [m.label for m in at.metric]
    assert "Inventory cost saving" in labels
    assert "Saving vs rule of thumb" in labels  # what-if tab rendered
    headline = pipeline_run.load_json("manifest")["headline"]
    saving = next(m for m in at.metric if m.label == "Inventory cost saving")
    assert saving.value == f"{headline['savings_vs_rule_pct']:.1f}%"


@pytest.mark.usefixtures("artifacts_env")
def test_what_if_slider_resimulates() -> None:
    at = AppTest.from_file(APP, default_timeout=120).run()
    before = next(m for m in at.metric if m.label == "Forecast-driven fill rate").value
    at.slider[1].set_value(14).run()  # lead time
    assert not at.exception
    after = next(m for m in at.metric if m.label == "Forecast-driven fill rate").value
    assert after != before


def test_app_explains_missing_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATHS__ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
    finally:
        get_settings.cache_clear()
    assert not at.exception
    assert "make train" in at.warning[0].value
