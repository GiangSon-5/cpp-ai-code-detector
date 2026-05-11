"""
tasks/silver_tasks.py — Extract features and push Silver layer to S3.

Extracts 32 ML features + 512 DL token IDs, pushes as Parquet to S3.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from io import BytesIO

from src.celery_workers import celery_app
from src.shared.logger import AppLogger

logger = AppLogger()


@celery_app.task(
    name="extract_and_push_silver",
    bind=True,
    max_retries=3,
    default_retry_delay=15,
    acks_late=True,
)
def extract_and_push_silver(self, code_hash: str, raw_code: str, label: int = None):
    """Extract 32 ML features + 512 DL tokens, push Silver parquet to S3.

    Args:
        code_hash: SHA-256 hash
        raw_code: C++ source code
        label: optional ground truth (0=human, 1=AI)
    """
    t0 = time.perf_counter()

    logger.info(
        module="silver_tasks",
        function="extract_and_push_silver",
        message=f"Starting Silver extraction: {code_hash[:12]}",
    )

    try:
        # --- ML Features (32 static features) ---
        from src.colab_runtime.scripts.feature_extractor import CppFeatureExtractorV8
        extractor = CppFeatureExtractorV8()
        ml_features = extractor.extract(raw_code)

        # --- DL Features (512 token IDs) ---
        # Use shared tokenizer if available, else skip
        dl_token_ids = []
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
            encoding = tokenizer(
                raw_code, max_length=512, truncation=True,
                padding="max_length", return_tensors=None,
            )
            dl_token_ids = encoding["input_ids"]
        except Exception as tok_exc:
            logger.warning(
                module="silver_tasks",
                function="extract_and_push_silver",
                message=f"DL tokenization failed (non-fatal): {tok_exc}",
            )

        # Build Silver record
        now = datetime.utcnow()
        silver_record = {
            "code_hash": code_hash,
            "ml_features": ml_features,
            "dl_token_ids": dl_token_ids,
            "label": label,
            "timestamp": now.isoformat() + "Z",
            "schema_version": "1.0",
        }

        # Push to S3 as JSON (Parquet requires pandas/pyarrow)
        json_bytes = json.dumps(silver_record, ensure_ascii=False).encode("utf-8")
        key = f"silver/{now.year}/{now.month:02d}/{now.day:02d}/{code_hash}.json"

        from src.shared.s3_client import DagsHubS3Client
        s3 = DagsHubS3Client()
        s3.upload_bytes(key=key, data=json_bytes, content_type="application/json")

        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            module="silver_tasks",
            function="extract_and_push_silver",
            message=f"Silver push complete: {key}",
            output_data={
                "s3_key": key,
                "ml_feature_count": len(ml_features),
                "dl_token_count": len(dl_token_ids),
            },
            latency_ms=latency,
        )

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        logger.error(
            module="silver_tasks",
            function="extract_and_push_silver",
            error=f"Silver extraction failed: {exc}",
            latency_ms=latency,
        )
        raise self.retry(exc=exc)
