from pathlib import Path

import mlflow

from demand_engine.pipelines.tracking import Tracker, flatten


def test_flatten_nested() -> None:
    assert flatten("", {"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_disabled_tracker_is_noop(tmp_path: Path) -> None:
    tracker = Tracker("", "exp", tmp_path)
    with tracker.run("x") as run_id:
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0})
        tracker.set_tags({"t": "v"})
        tracker.log_artifact(tmp_path / "missing.txt")
    assert run_id is None
    assert not tracker.enabled


def test_enabled_tracker_logs_to_sqlite(tmp_path: Path) -> None:
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    tracker = Tracker(uri, "unit-test", tmp_path / "artifacts")
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    with tracker.run("parent") as parent:
        tracker.log_params({"model": {"lr": 0.1}})
        tracker.log_metrics({"wrmsse": 0.8, "bad": float("nan")}, prefix="val_")
        tracker.log_artifact(tmp_path / "note.txt")
        with tracker.run("child", nested=True) as child:
            tracker.set_tags({"kind": "child"})
    assert parent
    assert child
    run = mlflow.get_run(parent)
    assert run.data.params["model.lr"] == "0.1"
    assert run.data.metrics == {"val_wrmsse": 0.8}  # NaN skipped
