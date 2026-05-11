"""
shared/config.py — Centralised Configuration (loaded once, used everywhere).

Reads from .env at the project root.  Every module imports `settings` from here
instead of reading os.environ directly so we have one single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Locate the project root (the folder that contains .env)
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/shared/../../ → project root
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH, override=False)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    """Immutable application-wide settings."""

    # --- Project paths ---
    PROJECT_ROOT: Path = _PROJECT_ROOT
    LOG_DIR: Path = field(default_factory=lambda: _PROJECT_ROOT / "logs")

    # --- Django ---
    DEBUG: bool = _get("DEBUG", "True").lower() in ("true", "1", "yes")
    SECRET_KEY: str = _get("SECRET_KEY", "django-insecure-change-me")
    ALLOWED_HOSTS: list[str] = field(
        default_factory=lambda: [h.strip() for h in _get("ALLOWED_HOSTS", "localhost").split(",")]
    )

    # --- FastAPI AI URL (Colab ngrok) ---
    FASTAPI_AI_URL: str = _get("FASTAPI_AI_URL", "http://localhost:8000")

    # --- PostgreSQL ---
    DB_NAME: str = _get("DB_NAME", "cpp_detector")
    DB_USER: str = _get("DB_USER", "postgres")
    DB_PASSWORD: str = _get("DB_PASSWORD", "postgres")
    DB_HOST: str = _get("DB_HOST", "localhost")
    DB_PORT: int = int(_get("DB_PORT", "5432"))

    @property
    def DATABASE_URL_ASYNC(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # --- Message Broker ---
    KAFKA_BOOTSTRAP_SERVERS: str = _get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    REDPANDA_API_URL: str = _get("REDPANDA_API_URL", "http://localhost:18081")

    # --- Celery ---
    CELERY_BROKER_URL: str = _get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = _get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    # --- DagsHub / S3 Data Lake ---
    DAGSHUB_REPO_OWNER: str = _get("DAGSHUB_REPO_OWNER", "")
    DAGSHUB_REPO_NAME: str = _get("DAGSHUB_REPO_NAME", "")
    DAGSHUB_TOKEN: str = _get("DAGSHUB_TOKEN", "")
    S3_ENDPOINT_URL: str = _get("S3_ENDPOINT_URL", "")
    AWS_ACCESS_KEY_ID: str = _get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = _get("AWS_SECRET_ACCESS_KEY", "")

    # --- AI API Keys ---
    GEMINI_API_KEY: str = _get("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = _get("OPENAI_API_KEY", "")
    NGROK_TOKEN: str = _get("NGROK_TOKEN", "")

    # --- Monitoring ---
    PROMETHEUS_PORT: int = int(_get("PROMETHEUS_PORT", "9090"))
    GRAFANA_PORT: int = int(_get("GRAFANA_PORT", "3000"))
    LOKI_URL: str = _get("LOKI_URL", "http://localhost:3100")

    # --- AI Thresholds ---
    DEFAULT_THRESHOLD: float = 0.5
    AMBIGUOUS_LOW: float = 0.40
    AMBIGUOUS_HIGH: float = 0.60
    FUSION_ALPHA: float = 0.48  # α·BERT + (1-α)·LightGBM


# Singleton — import `settings` from anywhere
settings = Settings()
