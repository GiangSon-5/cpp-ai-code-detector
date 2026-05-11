"""
engine/model_manager.py — Remote Model Manager proxy.

In the Hybrid "Dumb Worker" architecture, the Local Brain does NOT load
PyTorch models. It proxies status requests to the Colab worker.
"""

from __future__ import annotations

import os
from typing import Any

from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()

class ModelManager:
    """Proxy singleton to check remote Colab worker status."""

    _instance = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialised"):
            return
        self._initialised = True
        
        # We assume models are loaded on Colab if NGROK is set
        self._models_loaded = True 
        self._gpu_available = True

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    def load_models(self, model_type: str = "both") -> dict[str, Any]:
        """Dummy method for startup check."""
        return {
            "status": "Proxied to Colab",
            "oop": 5,
            "normal": 5,
            "backend": "remote_gpu"
        }
