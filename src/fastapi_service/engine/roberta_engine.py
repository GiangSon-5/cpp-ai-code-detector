"""
engine/roberta_engine.py — Remote Colab Inference Client.

Sends the code to the Colab ngrok URL instead of running inference locally,
preventing PyTorch/CUDA dependencies on the Local Brain.
"""

from __future__ import annotations

import time
import base64
from typing import Any
import httpx

from src.fastapi_service.engine.html_renderer import render_html_heatmap
from src.shared.data_contracts import ChunkResult
from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()

class RoBERTaEngine:
    """HTTP Client to call the Colab GPU Worker."""

    def __init__(self, model_manager: Any = None) -> None:
        self._colab_url = settings.FASTAPI_AI_URL
        # Support both Local GPU Server (LOCAL_GPU_API_KEY) and Colab (COLAB_API_KEY)
        import os as _os
        self._api_key = (
            _os.getenv("LOCAL_GPU_API_KEY")
            or _os.getenv("COLAB_API_KEY")
            or "colab-secret-key-123"
        )

    @AppLogger.log_function(module="roberta_engine")
    def analyze(
        self,
        code: str,
        model_type: str = "NORMAL",
        enable_lig: bool = True,
    ) -> dict[str, Any]:
        """Send code to Colab worker and parse the result."""
        t0 = time.perf_counter()

        if not self._colab_url:
            raise RuntimeError(
                "FASTAPI_AI_URL is not set in .env! "
                "Đặt http://localhost:8002 để dùng Local GPU Server "
                "hoặc URL ngrok để dùng Colab."
            )

        # Encode code as base64
        code_b64 = base64.b64encode(code.encode('utf-8')).decode('utf-8')
        
        payload = {
            "code_base64": code_b64,
            "model_type": model_type
        }
        
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json"
        }

        url = f"{self._colab_url.rstrip('/')}/api/predict/roberta"

        try:
            logger.info(
                module="roberta_engine", 
                function="analyze", 
                message=f"Calling GPU worker at {url}"
            )
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
            # Parse response
            # Expected from Colab: final_pred, final_score, total_tokens, total_chunks, chunks (list)
            # chunks list items: index, score, label, tokens, attrs, snippet
            
            chunk_results: list[ChunkResult] = []
            chunk_htmls: list[str] = []
            
            for c in data.get("chunks", []):
                attrs = c.get("attrs", [])
                tokens = c.get("tokens", [])
                
                chunk_html = ""
                if attrs and tokens:
                    chunk_html = render_html_heatmap(
                        tokens, attrs, title=f"Chunk {c.get('index')} Attribution"
                    )
                
                chunk_results.append(ChunkResult(
                    index=c.get("index", 0),
                    score=round(c.get("score", 0.5), 4),
                    label=c.get("label", "UNKNOWN"),
                    top_ai=[],  # We can parse this from tokens/attrs if needed, but expert explainer is heavy
                    top_hu=[],
                    snippet=c.get("snippet", ""),
                    html=chunk_html,
                    critique=""
                ))
                if chunk_html:
                    chunk_htmls.append(chunk_html)
                    
            global_html = "\n<hr>\n".join(chunk_htmls) if chunk_htmls else ""
            
            latency = (time.perf_counter() - t0) * 1000
            
            return {
                "chunks": chunk_results,
                "fold_scores": [],  # Optional
                "mean_score": round(data.get("final_score", 0.5), 4),
                "prediction": data.get("final_pred", "HUMAN WRITTEN"),
                "total_tokens": data.get("total_tokens", 0),
                "total_chunks": data.get("total_chunks", 0),
                "global_html": global_html,
                "model_type": model_type,
                "latency_ms": round(latency, 1),
            }
            
        except Exception as exc:
            logger.error(
                module="roberta_engine", 
                function="analyze", 
                error=f"GPU worker call failed: {exc}"
            )
            raise RuntimeError(f"GPU worker error: {exc}")

import os
