"""Logging setup shared by pipelines, the API and notebooks."""

from __future__ import annotations

import logging
import os

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_NOISY_LOGGERS = ("lightgbm", "mlflow", "urllib3", "numba", "matplotlib", "alembic.runtime")


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a consistent format.

    Args:
        level: Root log level name such as ``"INFO"`` or ``"DEBUG"``.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
