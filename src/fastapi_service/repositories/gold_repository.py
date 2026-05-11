"""
repositories/gold_repository.py — Async CRUD for Gold layer.

Implements Repository Pattern with full error handling and deep logging.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.fastapi_service.models.gold_prediction import GoldPrediction
from src.fastapi_service.models.gold_model_performance import GoldModelPerformance
from src.shared.data_contracts import GoldPredictionRecord
from src.shared.logger import AppLogger

logger = AppLogger()


class GoldRepository:
    """Async repository for Gold-layer database operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    @AppLogger.log_function(module="gold_repository")
    async def save_prediction(self, data: GoldPredictionRecord) -> GoldPrediction:
        """Insert a new prediction into gold_predictions."""
        row = GoldPrediction(
            code_hash=data.code_hash,
            user_id=data.user_id,
            model_used=data.model_used,
            classification=data.classification,
            prediction=data.prediction,
            confidence=data.confidence,
            perplexity=data.perplexity,
            max_ppl=data.max_ppl,
            burstiness=data.burstiness,
            is_ambiguous=data.is_ambiguous,
            retry_count=data.retry_count,
            total_tokens=data.total_tokens,
            total_chunks=data.total_chunks,
            global_critique=data.global_critique,
            top_ai_signals=data.top_ai_signals,
            top_hu_signals=data.top_hu_signals,
            chunk_details=[c.model_dump() for c in data.chunk_details],
            inference_ms=data.inference_ms,
            timestamp=data.timestamp,
        )
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row)
        return row

    @AppLogger.log_function(module="gold_repository")
    async def get_by_hash(self, code_hash: str) -> Optional[GoldPrediction]:
        """Lookup prediction by code_hash (cache layer)."""
        stmt = select(GoldPrediction).where(GoldPrediction.code_hash == code_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    @AppLogger.log_function(module="gold_repository")
    async def get_user_history(self, user_id: int, limit: int = 50) -> list[GoldPrediction]:
        """Fetch recent predictions for a specific user."""
        stmt = (
            select(GoldPrediction)
            .where(GoldPrediction.user_id == user_id)
            .order_by(GoldPrediction.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    @AppLogger.log_function(module="gold_repository")
    async def list_recent(self, limit: int = 50) -> list[GoldPrediction]:
        """Fetch the N most recent predictions."""
        stmt = (
            select(GoldPrediction)
            .order_by(GoldPrediction.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    @AppLogger.log_function(module="gold_repository")
    async def get_user_stats(self, user_id: int) -> dict[str, Any]:
        """Aggregate stats for a user's submissions."""
        total_stmt = (
            select(func.count(GoldPrediction.id))
            .where(GoldPrediction.user_id == user_id)
        )
        ai_stmt = (
            select(func.count(GoldPrediction.id))
            .where(GoldPrediction.user_id == user_id)
            .where(GoldPrediction.prediction == "AI GENERATED")
        )
        avg_conf_stmt = (
            select(func.avg(GoldPrediction.confidence))
            .where(GoldPrediction.user_id == user_id)
        )
        total = (await self.session.execute(total_stmt)).scalar() or 0
        ai_count = (await self.session.execute(ai_stmt)).scalar() or 0
        avg_conf = (await self.session.execute(avg_conf_stmt)).scalar() or 0.0

        return {
            "user_id": user_id,
            "total_submissions": total,
            "ai_generated_count": ai_count,
            "human_written_count": total - ai_count,
            "avg_confidence": round(float(avg_conf), 4),
        }

    @AppLogger.log_function(module="gold_repository")
    async def save_model_performance(self, data: dict[str, Any]) -> GoldModelPerformance:
        """Save a model evaluation record."""
        row = GoldModelPerformance(**data)
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row)
        return row

    @AppLogger.log_function(module="gold_repository")
    async def get_latest_performance(self, model_name: str) -> Optional[GoldModelPerformance]:
        """Get the most recent performance record for a model."""
        stmt = (
            select(GoldModelPerformance)
            .where(GoldModelPerformance.model_name == model_name)
            .order_by(GoldModelPerformance.eval_date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
