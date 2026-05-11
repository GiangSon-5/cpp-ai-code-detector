"""
core/dependencies.py — FastAPI dependency injection.

Provides:
    - get_db_session:  async session for request lifecycle
    - get_gold_repo:   GoldRepository instance
    - get_logger:      AppLogger instance
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.database import AsyncSessionFactory
from src.shared.logger import AppLogger


async def get_db_session() -> AsyncSession:  # type: ignore[misc]
    """Yield an async database session scoped to the request."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_logger() -> AppLogger:
    """Return the singleton logger instance."""
    return AppLogger()
