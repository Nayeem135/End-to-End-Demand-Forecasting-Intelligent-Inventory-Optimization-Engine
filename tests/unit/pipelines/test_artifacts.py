from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from demand_engine.pipelines.artifacts import ArtifactNotFoundError, latest_run, new_run


def test_round_trip_frames_and_json(tmp_path: Path) -> None:
    run = new_run(tmp_path, now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert run.run_id == "20260102T030405"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    run.save_frame("t", df)
    pd.testing.assert_frame_equal(run.load_frame("t"), df)
    run.save_json("meta", {"k": 1, "when": datetime(2026, 1, 1, tzinfo=UTC)})
    assert run.load_json("meta")["k"] == 1


def test_missing_artifacts_raise(tmp_path: Path) -> None:
    run = new_run(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        run.load_frame("nope")
    with pytest.raises(ArtifactNotFoundError):
        run.load_json("nope")


def test_run_ids_are_unique_within_a_second(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert new_run(tmp_path, now).run_id != new_run(tmp_path, now).run_id


def test_latest_run_ignores_incomplete_runs(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFoundError, match="no completed"):
        latest_run(tmp_path)
    done = new_run(tmp_path, datetime(2026, 1, 1, tzinfo=UTC))
    done.save_json("manifest", {"run_id": done.run_id})
    new_run(tmp_path, datetime(2026, 1, 2, tzinfo=UTC))  # newer but has no manifest yet
    assert latest_run(tmp_path).run_id == done.run_id
