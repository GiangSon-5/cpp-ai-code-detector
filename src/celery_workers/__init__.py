"""
Celery Workers — Background task processing.

celery_app.py — Celery application factory.
"""

import os
import sys
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celery import Celery

# Broker URL — Redpanda (Kafka-compatible)
BROKER_URL = os.environ.get(
    "CELERY_BROKER_URL",
    "kafka://localhost:9092",
)

# Result backend — PostgreSQL
RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND",
    "db+postgresql://postgres:postgres@localhost:5432/cpp_detector",
)

# Create Celery app
celery_app = Celery(
    "cpp_detector",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        "src.celery_workers.tasks.bronze_tasks",
        "src.celery_workers.tasks.silver_tasks",
        "src.celery_workers.tasks.gold_tasks",
        "src.celery_workers.tasks.retraining_tasks",
    ],
)

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    # Retry policy
    task_default_retry_delay=10,
    task_max_retries=3,
    # Beat schedule for periodic tasks
    beat_schedule={
        "check-retrain-every-24h": {
            "task": "check_retrain_trigger",
            "schedule": 86400.0,  # 24 hours
        },
    },
)
