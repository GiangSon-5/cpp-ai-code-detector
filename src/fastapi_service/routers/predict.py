"""
routers/predict.py — AI prediction endpoints.

Endpoints:
    POST /api/analyze         — sync analysis, returns full JSON
    POST /api/analyze_stream  — SSE streaming analysis
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.fastapi_service.core.dependencies import get_db_session, get_logger
from src.fastapi_service.repositories.gold_repository import GoldRepository
from src.fastapi_service.schemas.prediction_schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    SSEProgressEvent,
)
from src.fastapi_service.services.agent_service import AgentService
from src.shared.data_contracts import CodeSubmittedEvent, PredictionCompletedEvent
from src.shared.hashing import compute_code_hash
from src.shared.logger import AppLogger

router = APIRouter(prefix="/api", tags=["prediction"])
logger = AppLogger()

# Singleton agent service (models loaded once on startup)
_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


# Simple in-memory cache (bounded to 100 entries)
_RESULT_CACHE: dict[str, AnalyzeResponse] = {}
_CACHE_MAX = 100


# ------------------------------------------------------------------
# POST /api/analyze — Synchronous full analysis
# ------------------------------------------------------------------
@router.post("/analyze", response_model=AnalyzeResponse)
@AppLogger.log_function(module="router.predict")
async def analyze_code(
    request: AnalyzeRequest,
    session: AsyncSession = Depends(get_db_session),
) -> AnalyzeResponse:
    """Analyze C++ code for AI detection.

    Accepts Base64-encoded code, returns full analysis JSON.
    """
    t0 = time.perf_counter()

    # Decode Base64
    raw_code = _decode_base64(request.code_base64)

    # Check for empty code
    stripped = raw_code.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Empty code")

    # Cache check
    code_hash = compute_code_hash(raw_code)
    if code_hash in _RESULT_CACHE:
        logger.info(
            module="router.predict",
            function="analyze_code",
            message=f"Cache hit for {code_hash[:12]}",
        )
        return _RESULT_CACHE[code_hash]

    # Run full pipeline
    agent = get_agent_service()
    response, gold_record = await agent.analyze_code(raw_code, user_id=None)
    inference_ms = (time.perf_counter() - t0) * 1000

    # Record latency for admin dashboard
    try:
        from src.fastapi_service.routers.health import record_latency
        record_latency(inference_ms)
    except Exception:
        pass

    # Save to Gold DB
    try:
        repo = GoldRepository(session)
        await repo.save_prediction(gold_record)
    except Exception as exc:
        logger.error(
            module="router.predict",
            function="analyze_code",
            error=f"Failed to save Gold record: {exc}",
        )

    # Update cache (FIFO eviction)
    if len(_RESULT_CACHE) >= _CACHE_MAX:
        oldest_key = next(iter(_RESULT_CACHE))
        del _RESULT_CACHE[oldest_key]
    _RESULT_CACHE[code_hash] = response

    return response


# ------------------------------------------------------------------
# POST /api/analyze_stream — SSE streaming analysis
# ------------------------------------------------------------------
@router.post("/analyze_stream")
@AppLogger.log_function(module="router.predict")
async def analyze_code_stream(
    request: AnalyzeRequest,
    session: AsyncSession = Depends(get_db_session),
) -> StreamingResponse:
    """Analyze C++ code with SSE streaming progress events.

    Returns text/event-stream with JSON SSEProgressEvent payloads.
    """
    raw_code = _decode_base64(request.code_base64)

    stripped = raw_code.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Empty code")

    agent = get_agent_service()

    async def event_generator():
        t_stream_start = time.perf_counter()
        final_data = None
        async for event in agent.analyze_code_stream(raw_code, user_id=None):
            payload = event.model_dump_json()
            yield f"data: {payload}\n\n"

            if event.step == "complete" and event.data:
                final_data = event.data
                # Record streaming latency
                try:
                    from src.fastapi_service.routers.health import record_latency
                    record_latency((time.perf_counter() - t_stream_start) * 1000)
                except Exception:
                    pass

        # Save to Gold DB (best effort, after streaming completes)
        if final_data:
            try:
                code_hash = compute_code_hash(raw_code)
                # Re-run would be wasteful — construct record from final_data
                from src.shared.data_contracts import GoldPredictionRecord, ChunkResult

                chunk_details = [
                    ChunkResult(**c) for c in final_data.get("chunks", [])
                ]
                gold_record = GoldPredictionRecord(
                    code_hash=code_hash,
                    user_id=None,
                    model_used=final_data.get("model_used", ""),
                    classification=final_data.get("model_used", "").replace("C++ ", "").replace(" Model", ""),
                    prediction=final_data.get("final_pred", ""),
                    confidence=final_data.get("final_score", 0.5),
                    perplexity=final_data.get("perplexity", 0.0),
                    max_ppl=final_data.get("max_ppl", 0.0),
                    burstiness=final_data.get("burstiness", 0.0),
                    is_ambiguous=final_data.get("is_ambiguous", False),
                    retry_count=0,
                    total_tokens=final_data.get("total_tokens", 0),
                    total_chunks=final_data.get("total_chunks", 0),
                    global_critique=final_data.get("global_critique", ""),
                    top_ai_signals=[],
                    top_hu_signals=[],
                    chunk_details=chunk_details,
                    inference_ms=0,
                )
                repo = GoldRepository(session)
                await repo.save_prediction(gold_record)
            except Exception as exc:
                logger.error(
                    module="router.predict",
                    function="analyze_code_stream",
                    error=f"Failed to save Gold record after stream: {exc}",
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def _decode_base64(encoded: str) -> str:
    """Decode Base64-encoded string to UTF-8 text."""
    try:
        decoded_bytes = base64.b64decode(encoded, validate=True)
        return decoded_bytes.decode("utf-8")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Base64 encoding: {exc}",
        )
