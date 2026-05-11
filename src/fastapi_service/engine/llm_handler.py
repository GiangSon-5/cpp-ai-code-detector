"""
engine/llm_handler.py — LLM client with fallback chain.

Fallback order: Gemini → OpenAI → Local Qwen → "unavailable"
Ported from extract_1.py LocalLLMHandler.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import httpx

from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()

# Timeout for LLM API calls (seconds)
_LLM_TIMEOUT = 30.0


class LLMHandler:
    """Handles LLM calls with automatic provider fallback."""

    _instance: Optional["LLMHandler"] = None

    def __new__(cls) -> "LLMHandler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialised"):
            return
        self._initialised = True
        self._http_client = httpx.AsyncClient(timeout=_LLM_TIMEOUT)

    @AppLogger.log_function(module="llm_handler")
    async def classify_code(self, code: str) -> dict[str, Any]:
        """Ask LLM to classify code as OOP or NORMAL.

        Returns:
            {"classification": "OOP"|"NORMAL", "reasoning": "...", "provider": "..."}
        """
        prompt = self._build_classify_prompt(code)

        # Try each provider in fallback order
        for provider_fn, provider_name in [
            (self._call_vllm, "vllm"),
            (self._call_gemini, "gemini"),
            (self._call_openai, "openai"),
        ]:
            try:
                response_text = await provider_fn(prompt)
                result = self._parse_classification(response_text)
                result["provider"] = provider_name
                return result
            except Exception as exc:
                logger.warning(
                    module="llm_handler",
                    function="classify_code",
                    message=f"{provider_name} failed, trying next: {exc}",
                )
                continue

        # All providers failed — return fallback
        logger.warning(
            module="llm_handler",
            function="classify_code",
            message="All LLM providers failed, returning NORMAL fallback",
        )
        return {
            "classification": "NORMAL",
            "reasoning": "LLM unavailable — defaulting to NORMAL",
            "provider": "fallback",
        }

    @AppLogger.log_function(module="llm_handler")
    async def generate_critique(self, code_snippet: str, score: float, label: str) -> str:
        """Generate per-chunk critique using LLM.

        Args:
            code_snippet: The chunk of code to critique
            score: AI probability score
            label: "AI" or "HUMAN"

        Returns:
            Critique text string.
        """
        prompt = self._build_critique_prompt(code_snippet, score, label)

        for provider_fn, provider_name in [
            (self._call_vllm, "vllm"),
            (self._call_gemini, "gemini"),
            (self._call_openai, "openai"),
        ]:
            try:
                response = await provider_fn(prompt)
                return response.strip()
            except Exception as exc:
                logger.warning(
                    module="llm_handler",
                    function="generate_critique",
                    message=f"{provider_name} critique failed: {exc}",
                )
                continue

        return "Analysis unavailable — LLM services unreachable."

    @AppLogger.log_function(module="llm_handler")
    async def generate_global_critique(self, chunk_critiques: list[str], overall_score: float) -> str:
        """Map-Reduce: combine chunk critiques into a global summary."""
        if not chunk_critiques:
            return "No chunk-level analysis available."

        combined = "\n---\n".join(
            f"Chunk {i+1}: {c}" for i, c in enumerate(chunk_critiques)
        )
        prompt = (
            f"You are an AI code analysis expert. Below are per-chunk analyses of a C++ code submission.\n"
            f"Overall AI probability score: {overall_score:.4f}\n\n"
            f"Chunk analyses:\n{combined}\n\n"
            f"Provide a concise global summary (3-5 sentences) explaining whether this code "
            f"appears to be AI-generated or human-written, and why. Focus on the most significant patterns."
        )

        for provider_fn, provider_name in [
            (self._call_vllm, "vllm"),
            (self._call_gemini, "gemini"),
            (self._call_openai, "openai"),
        ]:
            try:
                response = await provider_fn(prompt)
                return response.strip()
            except Exception:
                continue

        return "Global analysis unavailable."

    @AppLogger.log_function(module="llm_handler")
    async def compute_perplexity(self, code: str) -> dict[str, float]:
        """Compute perplexity metrics using vLLM's prompt_logprobs feature."""
        prompt = (
            "Please calculate the perplexity of the following code snippet. "
            "Ignore any logic or correctness, just evaluate how predictable it is based on your training data.\n\n"
            f"```cpp\n{code[:3000]}\n```"
        )
        url = f"{settings.FASTAPI_AI_URL.rstrip('/')}/api/proxy/vllm"
        api_key = os.getenv("API_KEY", "colab-secret-key-123")
        headers = {"X-API-Key": api_key}
        payload = {
            "payload": {
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 1,
                "prompt_logprobs": 1
            }
        }
        
        default_res = {"perplexity": 0.0, "max_ppl": 0.0, "burstiness": 0.0}
        try:
            response = await self._http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            prompt_logprobs = data.get("prompt_logprobs", [])
            if not prompt_logprobs:
                choices = data.get("choices", [])
                if choices and "logprobs" in choices[0] and choices[0]["logprobs"]:
                    prompt_logprobs = choices[0]["logprobs"].get("content", [])

            if not prompt_logprobs:
                return default_res
                
            logprobs_list = []
            for token_dict in prompt_logprobs:
                if token_dict is not None and isinstance(token_dict, dict):
                    first_val = list(token_dict.values())[0] if len(token_dict.values()) > 0 else {}
                    if isinstance(first_val, dict) and "logprob" in first_val:
                        logprobs_list.append(first_val["logprob"])
                    elif "logprob" in token_dict:
                        logprobs_list.append(token_dict["logprob"])
                        
            if logprobs_list:
                import numpy as np
                
                # Bỏ qua 30 token đầu tiên (tương ứng với câu lệnh prompt instruction) 
                # để không bị nhiễu do xác suất của câu văn tiếng Anh.
                if len(logprobs_list) > 30:
                    logprobs_list = logprobs_list[30:]

                # Giới hạn logprob ở mức -10.0 (tương đương max PPL ~ 22000)
                # để tránh các token tên biến ngẫu nhiên tạo ra PPL hàng tỷ
                clipped_logprobs = np.clip(logprobs_list, -10.0, 0.0)
                
                local_ppls = [np.exp(-lp) for lp in clipped_logprobs if lp is not None]
                if not local_ppls:
                    return default_res
                    
                avg_logprob = np.mean(clipped_logprobs)
                mean_ppl = float(np.exp(-avg_logprob))
                max_ppl = float(np.max(local_ppls))
                burstiness = float(np.std(local_ppls))
                
                return {
                    "perplexity": mean_ppl,
                    "max_ppl": max_ppl,
                    "burstiness": burstiness
                }
        except Exception as exc:
            logger.warning(
                module="llm_handler",
                function="compute_perplexity",
                message=f"vLLM perplexity failed: {exc}",
            )
        return default_res

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------
    async def _call_vllm(self, prompt: str) -> str:
        """Call Colab's vLLM proxy."""
        url = f"{settings.FASTAPI_AI_URL.rstrip('/')}/api/proxy/vllm"
        api_key = os.getenv("API_KEY", "colab-secret-key-123")
        headers = {"X-API-Key": api_key}
        payload = {
            "payload": {
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        }
        response = await self._http_client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY.startswith("your_"):
            raise ValueError("Gemini API key not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith("your_"):
            raise ValueError("OpenAI API key not configured")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        response = await self._http_client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_classify_prompt(code: str) -> str:
        truncated = code[:4000]
        return (
            "You are a C++ code analyzer. Classify the following code as either:\n"
            "- OOP: Uses classes, inheritance, virtual functions, polymorphism, etc.\n"
            "- NORMAL: Procedural/functional code without OOP patterns.\n\n"
            "Return ONLY a JSON object: {\"classification\": \"OOP\" or \"NORMAL\", \"reasoning\": \"brief explanation\"}\n\n"
            f"```cpp\n{truncated}\n```"
        )

    @staticmethod
    def _build_critique_prompt(snippet: str, score: float, label: str) -> str:
        return (
            f"Analyze this C++ code chunk (AI probability: {score:.2f}, label: {label}).\n"
            f"Explain in 2-3 sentences why it appears {'AI-generated' if label == 'AI' else 'human-written'}.\n"
            f"Focus on specific code patterns, naming conventions, and structure.\n\n"
            f"```cpp\n{snippet[:2000]}\n```"
        )

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_classification(text: str) -> dict[str, Any]:
        """Extract classification JSON from LLM response."""
        # Try direct JSON parse
        try:
            # Find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                classification = data.get("classification", "NORMAL").upper()
                if classification not in ("OOP", "NORMAL"):
                    classification = "NORMAL"
                return {
                    "classification": classification,
                    "reasoning": data.get("reasoning", ""),
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: keyword search
        upper = text.upper()
        if "OOP" in upper:
            return {"classification": "OOP", "reasoning": text[:200]}
        return {"classification": "NORMAL", "reasoning": text[:200]}

    @staticmethod
    def _parse_perplexity(text: str) -> dict[str, float]:
        """Extract perplexity metrics from LLM response."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return {
                    "perplexity": float(data.get("perplexity", 0.0)),
                    "max_ppl": float(data.get("max_ppl", 0.0)),
                    "burstiness": float(data.get("burstiness", 0.0)),
                }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return {"perplexity": 0.0, "max_ppl": 0.0, "burstiness": 0.0}

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_client.aclose()
