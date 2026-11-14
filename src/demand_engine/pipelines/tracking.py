"""Thin MLflow wrapper so experiment tracking is optional and isolated from pipeline logic."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested dicts into ``a.b.c`` keys (MLflow params are flat)."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            out |= flatten(f"{prefix}.{key}" if prefix else str(key), inner)
        return out
    return {prefix: value}


class Tracker:
    """Logs parameters, metrics and artifacts to MLflow, or does nothing when disabled."""

    def __init__(self, tracking_uri: str, experiment: str, artifact_root: Path) -> None:
        """Create the tracker.

        Args:
            tracking_uri: MLflow backend store URI; empty string disables tracking.
            experiment: Experiment name.
            artifact_root: Where MLflow should store artifacts for a new experiment.
        """
        self.enabled = bool(tracking_uri)
        self._uri = tracking_uri
        self._experiment = experiment
        self._artifact_root = artifact_root

    @contextmanager
    def run(self, name: str, nested: bool = False) -> Iterator[str | None]:
        """Context manager for a (possibly nested) MLflow run; yields the run id."""
        if not self.enabled:
            yield None
            return
        import mlflow

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        mlflow.set_tracking_uri(self._uri)
        if mlflow.get_experiment_by_name(self._experiment) is None:
            self._artifact_root.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(
                self._experiment, artifact_location=self._artifact_root.resolve().as_uri()
            )
        mlflow.set_experiment(self._experiment)
        with mlflow.start_run(run_name=name, nested=nested) as active:
            yield str(active.info.run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log flattened parameters (values stringified, truncated to MLflow limits)."""
        if not self.enabled:
            return
        import mlflow

        mlflow.log_params({k: str(v)[:500] for k, v in flatten("", params).items()})

    def log_metrics(
        self, metrics: Mapping[str, float], prefix: str = "", step: int | None = None
    ) -> None:
        """Log numeric metrics (optionally at a step, e.g. a fold), skipping non-finite values."""
        if not self.enabled:
            return
        import math

        import mlflow

        clean = {f"{prefix}{k}": float(v) for k, v in metrics.items() if math.isfinite(float(v))}
        mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path) -> None:
        """Attach a local file to the active run."""
        if not self.enabled:
            return
        import mlflow

        mlflow.log_artifact(str(path))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Tag the active run."""
        if not self.enabled:
            return
        import mlflow

        mlflow.set_tags(dict(tags))
