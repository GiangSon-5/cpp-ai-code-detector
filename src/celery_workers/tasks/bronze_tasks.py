"""
tasks/bronze_tasks.py — Push raw code to DagsHub S3 Bronze layer.

Deep logging on every operation.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime

from src.celery_workers import celery_app
from src.shared.logger import AppLogger
from src.shared.s3_client import DagsHubS3Client

logger = AppLogger()


@celery_app.task(
    name="push_bronze_to_s3",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def push_bronze_to_s3(self, code_hash: str, raw_code: str, metadata: dict):
    """Push raw code + metadata lên DagsHub S3 Bronze bucket.

    Args:
        code_hash: SHA-256 hash of the code
        raw_code: raw C++ source code
        metadata: dict with user_id, source, timestamp, etc.

    S3 Key: bronze/{YYYY}/{MM}/{DD}/{code_hash}.json.gz
    """
    t0 = time.perf_counter()

    logger.info(
        module="bronze_tasks",
        function="push_bronze_to_s3",
        message=f"Starting Bronze push: {code_hash[:12]}",
        input_data={"code_hash": code_hash, "size": len(raw_code)},
    )

    try:
        # Build payload
        now = datetime.utcnow()
        payload = {
            "code_hash": code_hash,
            "raw_code": raw_code,
            "language": metadata.get("language", "cpp"),
            "source": metadata.get("source", "web_upload"),
            "user_id": metadata.get("user_id"),
            "file_size_bytes": len(raw_code.encode("utf-8")),
            "timestamp": now.isoformat() + "Z",
            "schema_version": "1.0",
        }

        # Format as JSON string (single line for JSONL style even if it's one object)
        json_str = json.dumps(payload, ensure_ascii=False)
        json_bytes = json_str.encode("utf-8")

        # Build S3 key with .jsonl extension
        key = f"bronze/{now.year}/{now.month:02d}/{now.day:02d}/{code_hash}.jsonl"

        # Upload
        s3 = DagsHubS3Client()
        s3.upload_bytes(key=key, data=json_bytes, content_type="application/jsonl")

        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            module="bronze_tasks",
            function="push_bronze_to_s3",
            message=f"Bronze push complete: {key}",
            output_data={
                "s3_key": key,
                "file_size": len(json_bytes),
            },
            latency_ms=latency,
        )

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        logger.error(
            module="bronze_tasks",
            function="push_bronze_to_s3",
            error=f"Bronze push failed: {exc}",
            input_data={"code_hash": code_hash},
            latency_ms=latency,
        )
        raise self.retry(exc=exc)
