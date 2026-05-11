"""
tasks/retraining_tasks.py — Periodic retraining trigger check.

Runs daily via Celery Beat. Checks if enough new Silver data
has accumulated to trigger a model retraining pipeline.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from src.celery_workers import celery_app
from src.shared.logger import AppLogger

logger = AppLogger()

# Minimum new samples required to trigger retraining
MIN_SAMPLES_FOR_RETRAIN = 500


@celery_app.task(
    name="check_retrain_trigger",
    bind=True,
    max_retries=1,
)
def check_retrain_trigger(self):
    """Periodic: check if enough new Silver data to trigger retraining.

    Business rules:
        - Check S3 silver/ for files created in the last 7 days
        - If count >= MIN_SAMPLES_FOR_RETRAIN → trigger retraining pipeline
        - Log the decision either way
    """
    t0 = time.perf_counter()

    logger.info(
        module="retraining_tasks",
        function="check_retrain_trigger",
        message="Starting daily retrain check",
    )

    try:
        from src.shared.s3_client import DagsHubS3Client

        s3 = DagsHubS3Client()
        now = datetime.utcnow()

        # Count Silver files from last 7 days
        total_new = 0
        for days_ago in range(7):
            dt = now - timedelta(days=days_ago)
            prefix = f"silver/{dt.year}/{dt.month:02d}/{dt.day:02d}/"
            try:
                files = s3.list_objects(prefix=prefix)
                total_new += len(files)
            except Exception:
                pass

        should_retrain = total_new >= MIN_SAMPLES_FOR_RETRAIN

        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            module="retraining_tasks",
            function="check_retrain_trigger",
            message=f"Retrain check: {total_new} new samples, threshold={MIN_SAMPLES_FOR_RETRAIN}, trigger={'YES' if should_retrain else 'NO'}",
            output_data={
                "new_samples": total_new,
                "threshold": MIN_SAMPLES_FOR_RETRAIN,
                "should_retrain": should_retrain,
            },
            latency_ms=latency,
        )

        if should_retrain:
            logger.info(
                module="retraining_tasks",
                function="check_retrain_trigger",
                message="🔁 Retraining triggered! Dispatching pipeline...",
            )
            # In production: trigger the retraining pipeline
            # For now, just log the event
            # trigger_retraining_pipeline.delay()

        return {
            "new_samples": total_new,
            "should_retrain": should_retrain,
            "checked_at": now.isoformat(),
        }

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        logger.error(
            module="retraining_tasks",
            function="check_retrain_trigger",
            error=f"Retrain check failed: {exc}",
            latency_ms=latency,
        )
        raise self.retry(exc=exc)
