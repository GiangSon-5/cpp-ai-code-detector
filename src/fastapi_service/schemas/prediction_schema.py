"""
schemas/prediction_schema.py — Pydantic v2 request/response schemas.

100% matches the SPEC field names and types.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.shared.data_contracts import ChunkResult, FingerprintResult

class AnalyzeRequest(BaseModel):
    """POST body for /api/analyze_stream."""
    code_base64: str = Field(..., min_length=4, description="Base64-encoded C++ source code")


class AnalyzeResponse(BaseModel):
    """Final JSON returned to the client."""
    final_pred: str = Field(..., description="AI GENERATED | HUMAN WRITTEN")
    final_score: float = Field(..., ge=0.0, le=1.0)
    model_used: str = Field(..., description="C++ OOP Model | C++ Normal Model")
    # Individual model confidence scores (for UI display)
    dl_score: float = Field(0.0, ge=0.0, le=1.0, description="DL model (RoBERTa Ensemble) AI probability")
    ml_score: float = Field(0.0, ge=0.0, le=1.0, description="ML model (LightGBM) AI probability")
    hybrid_score: float = Field(0.0, ge=0.0, le=1.0, description="Weighted hybrid (0.6*DL + 0.4*ML)")
    perplexity: float = 0.0
    max_ppl: float = 0.0
    burstiness: float = 0.0
    is_ambiguous: bool = False
    total_tokens: int = 0
    total_chunks: int = 0
    global_critique: str = ""
    global_html: str = ""
    chunks: list[ChunkResult] = Field(default_factory=list)
    fingerprint: FingerprintResult | None = None  # LightGBM + SHAP XAI



class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
    models_loaded: bool = False
    gpu_available: bool = False
    version: str = "1.0.0"
    # Runtime stats (for admin dashboard)
    request_count: int = 0
    cache_size: int = 0
    last_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    gpu_vram_used_gb: float | None = None
    gpu_vram_total_gb: float | None = None
    gpu_vram_pct: float | None = None
    uptime_seconds: float | None = None


class SSEProgressEvent(BaseModel):
    """Shape of each SSE event streamed during analysis."""
    step: str
    progress: int = Field(..., ge=0, le=100)
    message: str = ""
    data: dict | None = None
