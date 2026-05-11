"""
routers/health.py — Health check and runtime stats endpoint.

Exposes real metrics consumed by Django admin dashboard via polling.
"""

from __future__ import annotations

import time
from collections import deque

from fastapi import APIRouter

from src.fastapi_service.schemas.prediction_schema import HealthResponse
from src.shared.logger import AppLogger

router = APIRouter(tags=["health"])
logger = AppLogger()

# ── Runtime stats (module-level, updated by middleware in main.py) ──
_STATS: dict = {
    "request_count": 0,
    "start_time": time.time(),
    "latency_history": deque(maxlen=200),   # last 200 inference latencies (ms)
    "last_latency_ms": None,
}


def record_latency(latency_ms: float) -> None:
    """Called by predict.py after each analysis completes."""
    _STATS["request_count"] += 1
    _STATS["last_latency_ms"] = round(latency_ms, 2)
    _STATS["latency_history"].append(latency_ms)


def _compute_p95(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * 0.95) - 1)
    return round(sorted_v[idx], 2)


def _get_gpu_stats() -> dict:
    """Try to get real GPU VRAM from torch. Returns zeros if no GPU.

    Uses memory_reserved (actual VRAM footprint held by PyTorch CUDA allocator)
    instead of memory_allocated (only tensors, ignores fragmentation/overhead).
    Total is capped at 4 GB — the configured limit for this local GPU server.
    """
    # Max VRAM budget for this local GPU server (4 GB mode)
    VRAM_BUDGET_GB = 4.0
    try:
        import torch
        if torch.cuda.is_available():
            # memory_reserved = pages actually pinned in VRAM by the CUDA allocator
            used  = torch.cuda.memory_reserved() / 1024**3
            # Cap display total to configured budget, not full card capacity
            total = VRAM_BUDGET_GB
            # Clamp used to budget (can slightly exceed due to torch internals)
            used  = min(round(used, 2), total)
            pct   = used / total * 100 if total > 0 else 0
            return {
                "gpu_vram_used_gb": used,
                "gpu_vram_total_gb": total,
                "gpu_vram_pct": round(pct, 1),
            }
    except Exception:
        pass
    return {"gpu_vram_used_gb": None, "gpu_vram_total_gb": None, "gpu_vram_pct": None}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return server health status, model availability, and runtime metrics."""
    # Model status
    models_loaded = False
    gpu_available = False
    try:
        from src.fastapi_service.routers.predict import get_agent_service
        agent = get_agent_service()
        models_loaded = agent._model_manager.models_loaded
        gpu_available = agent._model_manager.gpu_available
    except Exception:
        pass

    # Cache size
    cache_size = 0
    try:
        from src.fastapi_service.routers.predict import _RESULT_CACHE
        cache_size = len(_RESULT_CACHE)
    except Exception:
        pass

    # Latency stats
    history = list(_STATS["latency_history"])
    p95 = _compute_p95(history)
    uptime = round(time.time() - _STATS["start_time"], 1)

    # GPU
    gpu = _get_gpu_stats()

    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        gpu_available=gpu_available,
        version="1.0.0",
        request_count=_STATS["request_count"],
        cache_size=cache_size,
        last_latency_ms=_STATS["last_latency_ms"],
        p95_latency_ms=p95,
        uptime_seconds=uptime,
        **gpu,
    )


@router.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "C++ AI Code Detector — FastAPI AI Service",
        "version": "1.0.0",
        "docs": "/docs",
    }
