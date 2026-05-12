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
    """Lấy VRAM thực tế từ Local GPU Server (port 8002).

    Orchestrator KHÔNG có GPU — model thực sự chạy ở Local GPU Server.
    Hàm này query GET /gpu-stats của Local GPU Server để lấy số thật.

    Fallback (nếu GPU server offline): dùng torch trực tiếp nếu có.
    """
    import os
    gpu_server_url = os.getenv("FASTAPI_AI_URL", "http://localhost:8002").rstrip("/")

    # ── Thử lấy từ Local GPU Server trước ───────────────────────────
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{gpu_server_url}/gpu-stats")
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "gpu_vram_used_gb":  data.get("gpu_vram_used_gb"),
                    "gpu_vram_total_gb": data.get("gpu_vram_total_gb", 4.0),
                    "gpu_vram_pct":      data.get("gpu_vram_pct"),
                }
    except Exception:
        pass

    # ── Fallback: đọc từ process hiện tại (nếu có torch + GPU) ─────
    try:
        import torch
        if torch.cuda.is_available():
            reserved     = torch.cuda.memory_reserved(0) / 1024**3
            total_display = 4.0
            pct           = reserved / total_display * 100 if total_display > 0 else 0
            return {
                "gpu_vram_used_gb":  round(reserved, 2),
                "gpu_vram_total_gb": total_display,
                "gpu_vram_pct":      round(pct, 1),
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
