"""Engine / session factories."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@lru_cache(maxsize=4)
def get_engine(url: str, echo: bool = False, connect_timeout: int = 5) -> Engine:
    """Return a pooled engine for ``url`` (cached per URL and options).

    A connect timeout keeps health checks responsive when the database is down.
    """
    connect_args = {"connect_timeout": connect_timeout} if url.startswith("postgresql") else {}
    return create_engine(url, echo=echo, pool_pre_ping=True, connect_args=connect_args)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to ``engine``."""
    return sessionmaker(bind=engine, expire_on_commit=False)
