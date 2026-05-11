"""
fastapi_service/main.py — FastAPI application entry point.

Startup sequence:
    1. Rotate logger (2-session)
    2. Initialize database tables
    3. Load AI models
    4. Mount routers

Run:
    cd project_root
    uvicorn src.fastapi_service.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.shared.logger import AppLogger
from src.shared.database import init_async_db

logger = AppLogger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    # ── STARTUP ──────────────────────────────────────────────────
    logger.rotate_on_startup()
    logger.info(
        module="main",
        function="lifespan",
        message="FastAPI AI Service starting up...",
    )

    # Init DB tables
    try:
        await init_async_db()
    except Exception as exc:
        logger.error(
            module="main",
            function="lifespan",
            error=f"Database init failed (non-fatal): {exc}",
        )

    # Load AI models
    try:
        from src.fastapi_service.routers.predict import get_agent_service
        agent = get_agent_service()
        load_result = agent.ensure_models_loaded()
        logger.info(
            module="main",
            function="lifespan",
            message="Model loading complete",
            output_data=load_result,
        )
    except Exception as exc:
        logger.warning(
            module="main",
            function="lifespan",
            message=f"Model loading failed (service will run without models): {exc}",
        )

    logger.info(
        module="main",
        function="lifespan",
        message="FastAPI AI Service is READY.",
    )

    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────
    logger.info(
        module="main",
        function="lifespan",
        message="FastAPI AI Service shutting down.",
    )
    try:
        from src.fastapi_service.engine.llm_handler import LLMHandler
        await LLMHandler().close()
    except Exception:
        pass


# ── APP FACTORY ──────────────────────────────────────────────────
app = FastAPI(
    title="C++ AI Code Detector — AI Service",
    description="Enterprise AI service for detecting AI-generated C++ code",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = None
    error_msg = None

    try:
        response = await call_next(request)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency = (time.perf_counter() - t0) * 1000
        status = response.status_code if response else 500
        logger.info(
            module="middleware",
            function="log_requests",
            message=f"{request.method} {request.url.path} → {status}",
            input_data={
                "method": request.method,
                "path": str(request.url.path),
                "query": str(request.query_params),
            },
            output_data={"status_code": status},
            error=error_msg,
            latency_ms=latency,
        )

    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        module="main",
        function="global_exception_handler",
        error=f"{type(exc).__name__}: {exc}",
        input_data={"path": str(request.url.path)},
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── MOUNT ROUTERS ────────────────────────────────────────────────
from src.fastapi_service.routers.health import router as health_router
from src.fastapi_service.routers.predict import router as predict_router

app.include_router(health_router)
app.include_router(predict_router)
