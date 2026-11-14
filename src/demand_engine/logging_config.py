"""Logging setup shared by pipelines, the API and notebooks.

Two output formats, chosen with the ``LOG_FORMAT`` setting:

* ``text`` (default) - human-readable ``time | level | logger | message`` lines;
* ``json`` - one JSON object per line (structured logs for log collectors).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Literal

LogFormat = Literal["text", "json"]

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_NOISY_LOGGERS = ("lightgbm", "mlflow", "urllib3", "numba", "matplotlib", "alembic.runtime")


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object: time, level, logger, message (+ exception)."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise ``record`` to a single-line JSON string."""
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: LogFormat = "text") -> None:
    """Configure root logging once with a consistent format.

    Args:
        level: Root log level name such as ``"INFO"`` or ``"DEBUG"``.
        fmt: ``"text"`` for human-readable lines, ``"json"`` for structured output.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
