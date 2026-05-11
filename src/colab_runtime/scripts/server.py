"""
server.py — FastAPI Endpoints (Dumb Worker Mode)

Endpoints:
    GET  /                    — Health check
    POST /api/predict/roberta — RoBERTa Ensemble + LIG
    POST /api/predict/lightgbm — LightGBM prediction
    POST /api/proxy/vllm      — Proxy to internal vLLM server (port 8001)

Features:
    - API Key authentication
    - In-memory COLAB_CACHE with size limit
    - Deep logging 2-session
"""

import os
import gc
import json
import time
import base64
import hashlib
import threading

import torch
import uvicorn
import nest_asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from .config import NGROK_TOKEN, MAX_CACHE_SIZE, colab_log
from .engine import run_roberta_engine

API_KEY = os.getenv("API_KEY", "colab-secret-key-123")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Cache kết quả ngay trên Colab để tiết kiệm GPU
COLAB_CACHE = {}


async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    colab_log("warning", "server", "get_api_key", "Invalid API key attempt")
    raise HTTPException(status_code=403, detail="Could not validate credentials")


def _manage_cache(key: str, value: dict):
    """Add to cache with FIFO eviction when exceeding MAX_CACHE_SIZE."""
    global COLAB_CACHE
    if len(COLAB_CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(COLAB_CACHE))
        del COLAB_CACHE[oldest_key]
        colab_log("info", "server", "_manage_cache",
                  f"Evicted oldest cache entry: {oldest_key[:10]}...")
    COLAB_CACHE[key] = value


# ==================================================================================
# FACTORY: Tạo FastAPI App (Dumb Worker Mode)
# ==================================================================================

def create_app(manager):
    """
    Tạo FastAPI application với endpoints chỉ gọi thẳng model, không chứa logic Agent.
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        print("\n" + "=" * 50)
        print("🚀 [STARTUP] ĐANG KHỞI ĐỘNG HỆ THỐNG MODEL WORKER...")
        colab_log("info", "server", "lifespan", "Server starting up")

        # 1. Load RoBERTa
        manager.load_resources()

        # 2. Load LightGBM
        from .hybrid_evaluator import hybrid_evaluator
        hybrid_evaluator.load_resources()

        latency = (time.perf_counter() - t0) * 1000
        print("✅ [STARTUP] HOÀN TẤT! SERVER ĐÃ SẴN SÀNG.")
        print("=" * 50 + "\n")
        colab_log("info", "server", "lifespan",
                  f"Startup complete in {latency:.0f}ms",
                  latency_ms=round(latency, 1))
        yield
        print("\n" + "=" * 50)
        print("🛑 [SHUTDOWN] ĐANG DỌN DẸP VRAM...")
        colab_log("info", "server", "lifespan", "Server shutting down")
        manager.cleanup()
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ [SHUTDOWN] ĐÃ XẢ SẠCH VRAM.")
        print("=" * 50 + "\n")

    app = FastAPI(lifespan=lifespan, title="C++ AI Code Detector - GPU Worker", version="3.0")

    class CodeRequest(BaseModel):
        code_base64: str
        model_type: str = "NORMAL"  # "NORMAL" hoặc "OOP" cho RoBERTa

    class VllmRequest(BaseModel):
        payload: dict

    # --- Health Check ---
    @app.get("/")
    async def home():
        from .hybrid_evaluator import hybrid_evaluator
        return {
            "status": "Online",
            "mode": "Microservices GPU Worker",
            "lightgbm_ready": hybrid_evaluator.is_ready,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "cache_size": len(COLAB_CACHE),
        }

    # --- Proxy Endpoint for vLLM (Port 8001) ---
    @app.post("/api/proxy/vllm")
    async def proxy_vllm(request: VllmRequest, api_key: str = Depends(get_api_key)):
        """Proxy request to internal vLLM server running on port 8001"""
        t0 = time.perf_counter()
        colab_log("info", "server", "proxy_vllm", "vLLM proxy request received")

        import httpx
        url = "http://localhost:8001/v1/chat/completions"
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=request.payload)
                response.raise_for_status()
                latency = (time.perf_counter() - t0) * 1000
                colab_log("info", "server", "proxy_vllm",
                          f"vLLM proxy complete",
                          latency_ms=round(latency, 1))
                return response.json()
            except Exception as e:
                colab_log("error", "server", "proxy_vllm", f"vLLM proxy error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    # --- Endpoint 1: RoBERTa Prediction ---
    @app.post("/api/predict/roberta")
    async def predict_roberta(request: CodeRequest, api_key: str = Depends(get_api_key)):
        """Run RoBERTa Ensemble + LIG attribution on C++ code."""
        t0 = time.perf_counter()

        if not request.code_base64.strip():
            raise HTTPException(status_code=400, detail="Empty code")

        try:
            decoded_str = base64.b64decode(request.code_base64).decode('utf-8').replace('\r\n', '\n').strip()

            # 1. Kiểm tra Cache dựa trên nội dung code đã xử lý (đồng bộ với Local)
            code_hash = hashlib.md5(f"bert_{request.model_type}_{decoded_str}".encode()).hexdigest()
            if code_hash in COLAB_CACHE:
                latency = (time.perf_counter() - t0) * 1000
                print(f"\n⚡ [GPU CACHE HIT] RoBERTa ({request.model_type}) — {code_hash[:10]}")
                colab_log("info", "server", "predict_roberta",
                          f"CACHE HIT: {code_hash[:10]}",
                          model_type=request.model_type,
                          cache_hit=True, latency_ms=round(latency, 1))
                return COLAB_CACHE[code_hash]

            print(f"\n🚀 [GPU WORKER] RoBERTa ({request.model_type}) — {code_hash[:10]}...")
            colab_log("info", "server", "predict_roberta",
                      f"CACHE MISS — running inference: {code_hash[:10]}",
                      model_type=request.model_type,
                      code_len=len(decoded_str))

            model_name = "C++ OOP Model" if request.model_type == "OOP" else "C++ Normal Model"
            result_data = run_roberta_engine(decoded_str, model_name, manager)

            # Trả về kết quả PHẲNG để khớp với logic Local Brain
            final_res = {
                "success": True,
                "score": result_data.get("final_score", 0.5),
                **result_data
            }

            # Lưu cache (với FIFO eviction)
            _manage_cache(code_hash, final_res)

            latency = (time.perf_counter() - t0) * 1000
            colab_log("info", "server", "predict_roberta",
                      f"Inference complete: {result_data.get('final_pred', '?')}",
                      final_score=result_data.get("final_score"),
                      prediction=result_data.get("final_pred"),
                      latency_ms=round(latency, 1))
            return final_res

        except Exception as e:
            colab_log("error", "server", "predict_roberta", f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # --- Endpoint 2: LightGBM Prediction ---
    @app.post("/api/predict/lightgbm")
    async def predict_lightgbm(request: CodeRequest, api_key: str = Depends(get_api_key)):
        """Run LightGBM prediction on C++ code features."""
        t0 = time.perf_counter()

        if not request.code_base64.strip():
            raise HTTPException(status_code=400, detail="Empty code")

        try:
            decoded_str = base64.b64decode(request.code_base64).decode('utf-8').replace('\r\n', '\n').strip()

            # 1. Kiểm tra Cache (đồng bộ hash với Local)
            code_hash = hashlib.md5(f"lgbm_{decoded_str}".encode()).hexdigest()
            if code_hash in COLAB_CACHE:
                latency = (time.perf_counter() - t0) * 1000
                print(f"\n⚡ [GPU CACHE HIT] LightGBM — {code_hash[:10]}")
                colab_log("info", "server", "predict_lightgbm",
                          f"CACHE HIT: {code_hash[:10]}",
                          cache_hit=True, latency_ms=round(latency, 1))
                return COLAB_CACHE[code_hash]

            print(f"\n🚀 [GPU WORKER] LightGBM — {code_hash[:10]}...")
            colab_log("info", "server", "predict_lightgbm",
                      f"CACHE MISS — running prediction: {code_hash[:10]}")

            from .hybrid_evaluator import hybrid_evaluator
            lgbm_score = hybrid_evaluator.predict_lgbm(decoded_str)
            if lgbm_score is None:
                colab_log("error", "server", "predict_lightgbm",
                          "LightGBM prediction failed or model not loaded")
                raise HTTPException(status_code=500, detail="LightGBM prediction failed or model not loaded")

            final_res = {"success": True, "score": lgbm_score}
            _manage_cache(code_hash, final_res)

            latency = (time.perf_counter() - t0) * 1000
            colab_log("info", "server", "predict_lightgbm",
                      f"Prediction complete: {lgbm_score:.4f}",
                      score=lgbm_score, latency_ms=round(latency, 1))
            return final_res

        except Exception as e:
            colab_log("error", "server", "predict_lightgbm", f"Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ==================================================================================
# NGROK TUNNEL + SERVER LAUNCHER
# ==================================================================================

def start_ngrok(port=8000):
    """Khởi tạo ngrok tunnel và trả về public URL"""
    from pyngrok import ngrok

    print("🔄 Đang chuẩn bị Ngrok cho GPU Worker...")
    colab_log("info", "server", "start_ngrok", f"Starting ngrok tunnel on port {port}")
    time.sleep(1)

    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)
    else:
        print("⚠️ Ngrok Token chưa cấu hình! Tunnel có thể không ổn định.")
        colab_log("warning", "server", "start_ngrok", "No NGROK_TOKEN configured")

    public_url = ngrok.connect(port).public_url
    print(f"\n🚀 ═══════════════════════════════════════════════════")
    print(f"   TÊN MIỀN NGROK CHO GPU WORKER:")
    print(f"   {public_url}")
    print(f"   (API Key: {API_KEY})")
    print(f"═══════════════════════════════════════════════════════\n")
    colab_log("info", "server", "start_ngrok",
              f"ngrok tunnel active: {public_url}",
              public_url=public_url)
    return public_url


def run_server(app, port=8000):
    """Chạy uvicorn server trong background thread"""
    nest_asyncio.apply()

    def _start():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

    thread = threading.Thread(target=_start, daemon=True)
    thread.start()
    print(f"🌐 GPU Worker Server đang chạy tại port {port}")
    colab_log("info", "server", "run_server", f"Uvicorn started on port {port}")
    return thread
