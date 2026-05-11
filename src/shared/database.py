"""
shared/database.py — Database engine and session factories.

Provides:
    - AsyncSessionFactory  (for FastAPI)
    - SyncSessionFactory   (for Celery / batch scripts)
    - Base                 (declarative base for SQLAlchemy ORM models)
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()


# =====================================================================
# Declarative Base (shared between FastAPI & Django raw models)
# =====================================================================
class Base(DeclarativeBase):
    pass


# =====================================================================
# Async engine + session (FastAPI)
# =====================================================================
_async_engine = create_async_engine(
    settings.DATABASE_URL_ASYNC,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionFactory = async_sessionmaker(
    bind=_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncSession:
    """FastAPI dependency — yields a transactional async session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session  # type: ignore[misc]
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_async_db() -> None:
    """Create all tables (dev convenience)."""
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(
        module="database",
        function="init_async_db",
        message="Async DB tables created / verified.",
    )


# =====================================================================
# Sync engine + session (Celery / batch)
# =====================================================================
_sync_engine = create_engine(
    settings.DATABASE_URL_SYNC,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SyncSessionFactory = sessionmaker(bind=_sync_engine, class_=Session)


def get_sync_session() -> Session:
    """Context-manager-like helper for sync code."""
    return SyncSessionFactory()


def init_sync_db() -> None:
    """Create all tables synchronously."""
    Base.metadata.create_all(bind=_sync_engine)
    logger.info(
        module="database",
        function="init_sync_db",
        message="Sync DB tables created / verified.",
    )
