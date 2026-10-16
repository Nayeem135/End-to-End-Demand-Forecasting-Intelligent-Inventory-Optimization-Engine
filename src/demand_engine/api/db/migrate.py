"""Programmatic Alembic helpers (no dependency on the repository-level ``alembic.ini``)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def alembic_config(url: str) -> Config:
    """Alembic configuration pointing at the packaged migrations and ``url``."""
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(url: str, revision: str = "head") -> None:
    """Apply migrations up to ``revision``."""
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str, revision: str = "base") -> None:
    """Revert migrations down to ``revision``."""
    command.downgrade(alembic_config(url), revision)
