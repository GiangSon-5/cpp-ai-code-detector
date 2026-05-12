"""
server.py — FastAPI Endpoints cho Local GPU Server (thay thế Colab Worker)

API Contract giống hệt Colab server.py để Local Orchestrator (src/fastapi_service)
không cần thay đổi gì, chỉ cần đổi FASTAPI_AI_URL trong .env.

Endpoints:
  GET  /                        — Health check
  POST /api/predict/roberta     — RoBERTa 1-fold + LIG
  POST /api/predict/lightgbm    — LightGBM prediction (nếu có model)
  POST /api/proxy/vllm          — DISABLED (trả 501 với thông báo rõ ràng)

Bỏ qua:
  - ngrok (không cần — chạy local)
  - vLLM / Qwen (không đủ VRAM 4GB)
  - Google Drive (model đọc từ local path)
"""

import os
import gc
import time
import base64
import hashlib

import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from .config import (
    LOCAL_GPU_API_KEY, MAX_CACHE_SIZE, LOCAL_GPU_PORT, gpu_log,
    PATH_SAVED_MODELS,
)
from .engine import LocalModelManager, run_roberta_engine_local


# ==================================================================================
# AUTH
# ==================================================================================
API_KEY_NAME    = "X-API-Key"
api_key_header  = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def verify_api_key(key: str = Security(api_key_header)):
    if key == LOCAL_GPU_API_KEY:
        return key
    gpu_log("warning", "server", "verify_api_key", "Invalid API key attempt")
    raise HTTPException(status_code=403, detail="Could not validate credentials")


# ==================================================================================
# CACHE (in-memory FIFO)
# ==================================================================================
LOCAL_GPU_CACHE: dict = {}


def _cache_put(key: str, value: dict):
    """FIFO eviction khi vượt MAX_CACHE_SIZE."""
    global LOCAL_GPU_CACHE
    if len(LOCAL_GPU_CACHE) >= MAX_CACHE_SIZE:
        oldest = next(iter(LOCAL_GPU_CACHE))
        del LOCAL_GPU_CACHE[oldest]
        gpu_log("info", "server", "_cache_put",
                f"Evicted oldest cache entry: {oldest[:10]}...")
    LOCAL_GPU_CACHE[key] = value


# ==================================================================================
# SCHEMAS
# ==================================================================================

class CodeRequest(BaseModel):
    code_base64: str
    model_type: str = "NORMAL"   # "NORMAL" | "OOP"


class VllmRequest(BaseModel):
    payload: dict


# ==================================================================================
# APP FACTORY
# ==================================================================================

def create_app(manager: LocalModelManager) -> FastAPI:
    """Tạo FastAPI app với manager đã được inject."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        print("\n" + "=" * 60)
        print("🚀 [LOCAL GPU SERVER] Starting up...")
        gpu_log("info", "server", "lifespan", "Server starting up")

        # 1. Load RoBERTa (1 fold each)
        load_result = manager.load_resources()
        print(f"   RoBERTa: {load_result}")

        # 2. Load LightGBM (optional)
        from .hybrid_evaluator import local_hybrid_evaluator
        local_hybrid_evaluator.load_resources()

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"✅ [LOCAL GPU SERVER] READY in {elapsed:.0f}ms")
        print("=" * 60 + "\n")
        gpu_log("info", "server", "lifespan",
                "Startup complete", latency_ms=round(elapsed, 1))

        yield

        # Shutdown
        print("\n" + "=" * 60)
        print("🛑 [LOCAL GPU SERVER] Shutting down...")
        manager.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ [LOCAL GPU SERVER] VRAM cleared.")
        print("=" * 60 + "\n")
        gpu_log("info", "server", "lifespan", "Shutdown complete")

    app = FastAPI(
        lifespan=lifespan,
        title="C++ AI Code Detector — Local GPU Worker",
        version="1.0.0",
        description=(
            "Thay thế Colab Worker cho môi trường thử nghiệm local. "
            "Chỉ chạy 1 fold RoBERTa, không có vLLM/Qwen."
        ),
    )

    # ------------------------------------------------------------------
    # GET /
    # ------------------------------------------------------------------
    @app.get("/")
    async def health():
        from .hybrid_evaluator import local_hybrid_evaluator
        return {
            "status":           "Online",
            "mode":             "Local GPU Worker (1-fold, no vLLM)",
            "lightgbm_ready":   local_hybrid_evaluator.is_ready,
            "oop_folds_loaded": len(manager.oop_models),
            "normal_folds_loaded": len(manager.normal_models),
            "gpu":              (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "CPU"),
            "cache_size":       len(LOCAL_GPU_CACHE),
        }

    # ------------------------------------------------------------------
    # GET /gpu-stats  — VRAM metrics cho Orchestrator polling
    # ------------------------------------------------------------------
    @app.get("/gpu-stats")
    async def gpu_stats():
        """Trả về thống kê VRAM thực tế từ Local GPU Server process.

        Dùng memory_reserved() thay vì memory_allocated() vì:
          - memory_allocated() → chỉ tính tensors đang giữ tích cực (rất thấp sau inference)
          - memory_reserved() → tổng VRAM PyTorch đã allocate từ driver (bao gồm cache pool)
            → phản ánh VRAM thực sự bị chiếm bởi process này
        """
        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "gpu_vram_used_gb": 0.0,
                "gpu_vram_total_gb": 4.0,
                "gpu_vram_pct": 0.0,
                "gpu_name": "N/A",
            }

        try:
            # memory_reserved() = allocated + cached pool — phản ánh VRAM thực tế bị chiếm
            reserved_bytes   = torch.cuda.memory_reserved(0)
            allocated_bytes  = torch.cuda.memory_allocated(0)
            total_bytes      = torch.cuda.get_device_properties(0).total_memory

            used_gb     = reserved_bytes  / 1024**3
            alloc_gb    = allocated_bytes / 1024**3
            total_gb    = total_bytes     / 1024**3

            # Hiển thị giới hạn 4GB
            display_total = min(total_gb, 4.0)
            display_used  = min(used_gb, display_total)
            pct           = (display_used / display_total * 100) if display_total > 0 else 0.0

            return {
                "gpu_available":    True,
                "gpu_name":         torch.cuda.get_device_name(0),
                "gpu_vram_used_gb": round(display_used, 2),
                "gpu_vram_alloc_gb": round(alloc_gb, 2),   # tensors only (thường thấp hơn)
                "gpu_vram_total_gb": round(display_total, 2),
                "gpu_vram_pct":     round(pct, 1),
                "models_in_ram": {
                    "oop":    len(manager.oop_models),
                    "normal": len(manager.normal_models),
                },
            }
        except Exception as exc:
            gpu_log("error", "server", "gpu_stats", f"GPU stats error: {exc}")
            return {
                "gpu_available": False,
                "gpu_vram_used_gb": 0.0,
                "gpu_vram_total_gb": 4.0,
                "gpu_vram_pct": 0.0,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # POST /api/predict/roberta
    # ------------------------------------------------------------------
    @app.post("/api/predict/roberta")
    async def predict_roberta(
        request: CodeRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """RoBERTa 1-fold + LIG attribution."""
        t0 = time.perf_counter()

        if not request.code_base64.strip():
            raise HTTPException(status_code=400, detail="Empty code_base64")

        try:
            decoded = (
                base64.b64decode(request.code_base64)
                .decode("utf-8")
                .replace("\r\n", "\n")
                .strip()
            )

            # Cache check
            cache_key = hashlib.md5(
                f"bert_{request.model_type}_{decoded}".encode()
            ).hexdigest()

            if cache_key in LOCAL_GPU_CACHE:
                latency = (time.perf_counter() - t0) * 1000
                print(f"\n⚡ [CACHE HIT] RoBERTa ({request.model_type}) — {cache_key[:10]}")
                gpu_log("info", "server", "predict_roberta",
                        "CACHE HIT", model_type=request.model_type,
                        cache_hit=True, latency_ms=round(latency, 1))
                return LOCAL_GPU_CACHE[cache_key]

            # Đảm bảo model đã load
            if not manager._loaded:
                manager.load_resources()

            print(f"\n🚀 [LOCAL GPU] RoBERTa ({request.model_type}) — {cache_key[:10]}...")
            gpu_log("info", "server", "predict_roberta",
                    "CACHE MISS — running inference",
                    model_type=request.model_type, code_len=len(decoded))

            model_name = (
                "C++ OOP Model" if request.model_type == "OOP"
                else "C++ Normal Model"
            )
            result_data = run_roberta_engine_local(decoded, model_name, manager)

            if "error" in result_data:
                raise HTTPException(status_code=500, detail=result_data["error"])

            final_res = {
                "success": True,
                "score":   result_data.get("final_score", 0.5),
                **result_data,
            }

            _cache_put(cache_key, final_res)

            latency = (time.perf_counter() - t0) * 1000
            gpu_log("info", "server", "predict_roberta",
                    f"Inference complete: {result_data.get('final_pred', '?')}",
                    final_score=result_data.get("final_score"),
                    prediction=result_data.get("final_pred"),
                    latency_ms=round(latency, 1))
            return final_res

        except HTTPException:
            raise
        except Exception as exc:
            gpu_log("error", "server", "predict_roberta", f"Error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # POST /api/predict/lightgbm
    # ------------------------------------------------------------------
    @app.post("/api/predict/lightgbm")
    async def predict_lightgbm(
        request: CodeRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """LightGBM prediction (chạy trên CPU — không cần GPU)."""
        t0 = time.perf_counter()

        if not request.code_base64.strip():
            raise HTTPException(status_code=400, detail="Empty code_base64")

        try:
            decoded = (
                base64.b64decode(request.code_base64)
                .decode("utf-8")
                .replace("\r\n", "\n")
                .strip()
            )

            cache_key = hashlib.md5(f"lgbm_{decoded}".encode()).hexdigest()
            if cache_key in LOCAL_GPU_CACHE:
                latency = (time.perf_counter() - t0) * 1000
                print(f"\n⚡ [CACHE HIT] LightGBM — {cache_key[:10]}")
                gpu_log("info", "server", "predict_lightgbm",
                        "CACHE HIT", cache_hit=True, latency_ms=round(latency, 1))
                return LOCAL_GPU_CACHE[cache_key]

            from .hybrid_evaluator import local_hybrid_evaluator
            lgbm_score = local_hybrid_evaluator.predict_lgbm(decoded)

            if lgbm_score is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "LightGBM model không sẵn sàng. "
                        "Kiểm tra LOCAL_SAVED_MODELS_PATH trong .env."
                    ),
                )

            final_res = {"success": True, "score": lgbm_score}
            _cache_put(cache_key, final_res)

            latency = (time.perf_counter() - t0) * 1000
            gpu_log("info", "server", "predict_lightgbm",
                    f"Prediction: {lgbm_score:.4f}",
                    score=lgbm_score, latency_ms=round(latency, 1))
            return final_res

        except HTTPException:
            raise
        except Exception as exc:
            gpu_log("error", "server", "predict_lightgbm", f"Error: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # POST /api/proxy/vllm   — DISABLED
    # ------------------------------------------------------------------
    @app.post("/api/proxy/vllm")
    async def proxy_vllm_disabled(
        request: VllmRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        vLLM proxy bị vô hiệu hóa trong Local GPU Server.
        Lý do: 4GB VRAM không đủ để load Qwen 2.5-7B.

        Orchestrator sẽ nhận 501 và tự fallback về Gemini hoặc bỏ qua perplexity.
        """
        gpu_log("info", "server", "proxy_vllm_disabled",
                "vLLM proxy called but disabled (4GB VRAM mode)")
        raise HTTPException(
            status_code=501,
            detail=(
                "vLLM/Qwen bị vô hiệu hóa trong Local GPU Server "
                "(4GB VRAM không đủ để load Qwen 2.5-7B). "
                "Sử dụng Gemini API cho critique hoặc bỏ qua perplexity."
            ),
        )

    return app


# ==================================================================================
# STANDALONE RUNNER
# ==================================================================================

def run_local_gpu_server():
    """Khởi động server độc lập (không cần ngrok)."""
    print("\n" + "=" * 60)
    print("🖥️  LOCAL GPU SERVER")
    print(f"   Port  : {LOCAL_GPU_PORT}")
    print(f"   API   : http://0.0.0.0:{LOCAL_GPU_PORT}")
    print(f"   Key   : {LOCAL_GPU_API_KEY}")
    print("=" * 60)

    manager = LocalModelManager()
    app = create_app(manager)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=LOCAL_GPU_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    run_local_gpu_server()
