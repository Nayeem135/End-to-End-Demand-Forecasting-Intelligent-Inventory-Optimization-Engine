"""File-based artifact store: one directory of Parquet/JSON files per pipeline run."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RUNS_DIR = "runs"


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when a requested artifact or run does not exist."""


@dataclass(frozen=True)
class RunArtifacts:
    """Read/write access to the artifacts of a single pipeline run."""

    root: Path

    @property
    def run_id(self) -> str:
        """Run identifier (the directory name)."""
        return self.root.name

    def save_frame(self, name: str, df: pd.DataFrame) -> Path:
        """Persist ``df`` as ``<name>.parquet``."""
        path = self.root / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def load_frame(self, name: str) -> pd.DataFrame:
        """Load ``<name>.parquet``."""
        path = self.root / f"{name}.parquet"
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact '{name}' not found in {self.root}")
        return pd.read_parquet(path)

    def save_json(self, name: str, payload: dict[str, Any]) -> Path:
        """Persist ``payload`` as pretty-printed ``<name>.json``."""
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def load_json(self, name: str) -> dict[str, Any]:
        """Load ``<name>.json``."""
        path = self.root / f"{name}.json"
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact '{name}' not found in {self.root}")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data


def new_run(artifacts_dir: Path, now: datetime | None = None) -> RunArtifacts:
    """Create a fresh, uniquely named run directory."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    base = artifacts_dir / RUNS_DIR
    root = base / stamp
    suffix = 1
    while root.exists():
        suffix += 1
        root = base / f"{stamp}-{suffix}"
    root.mkdir(parents=True)
    logger.info("Artifacts for this run: %s", root)
    return RunArtifacts(root)


def latest_run(artifacts_dir: Path) -> RunArtifacts:
    """Return the most recent *completed* run (one containing ``manifest.json``)."""
    base = artifacts_dir / RUNS_DIR
    runs = (
        sorted(p for p in base.glob("*") if (p / "manifest.json").is_file())
        if base.is_dir()
        else []
    )
    if not runs:
        raise ArtifactNotFoundError(
            f"no completed pipeline runs in {base}; run 'demand-engine run-all'"
        )
    return RunArtifacts(runs[-1])
