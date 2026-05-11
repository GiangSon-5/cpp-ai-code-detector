"""
tasks/gold_tasks.py — Log prediction results to PostgreSQL Gold layer.
"""

from __future__ import annotations

import time

from src.celery_workers import celery_app
from src.shared.logger import AppLogger

logger = AppLogger()


@celery_app.task(
    name="log_prediction_to_gold",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def log_prediction_to_gold(self, prediction_data: dict):
    """Write prediction result to PostgreSQL Gold table.

    Args:
        prediction_data: dict matching GoldPredictionRecord schema
    """
    t0 = time.perf_counter()

    code_hash = prediction_data.get("code_hash", "unknown")
    logger.info(
        module="gold_tasks",
        function="log_prediction_to_gold",
        message=f"Logging Gold prediction: {code_hash[:12]}",
        input_data={"code_hash": code_hash},
    )

    try:
        # Use sync DB session for Celery (not async)
        from src.shared.database import get_sync_session
        from src.fastapi_service.models.gold_prediction import GoldPrediction

        session = get_sync_session()
        try:
            row = GoldPrediction(
                code_hash=prediction_data.get("code_hash"),
                user_id=prediction_data.get("user_id"),
                model_used=prediction_data.get("model_used", ""),
                classification=prediction_data.get("classification", ""),
                prediction=prediction_data.get("prediction", ""),
                confidence=prediction_data.get("confidence", 0.5),
                perplexity=prediction_data.get("perplexity", 0.0),
                max_ppl=prediction_data.get("max_ppl", 0.0),
                burstiness=prediction_data.get("burstiness", 0.0),
                is_ambiguous=prediction_data.get("is_ambiguous", False),
                retry_count=prediction_data.get("retry_count", 0),
                total_tokens=prediction_data.get("total_tokens", 0),
                total_chunks=prediction_data.get("total_chunks", 0),
                global_critique=prediction_data.get("global_critique", ""),
                top_ai_signals=prediction_data.get("top_ai_signals", []),
                top_hu_signals=prediction_data.get("top_hu_signals", []),
                chunk_details=prediction_data.get("chunk_details", []),
                inference_ms=prediction_data.get("inference_ms", 0),
            )
            session.add(row)
            session.commit()

            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                module="gold_tasks",
                function="log_prediction_to_gold",
                message=f"Gold logged: {code_hash[:12]} → {prediction_data.get('prediction')}",
                output_data={"id": row.id},
                latency_ms=latency,
            )
        finally:
            session.close()

    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        logger.error(
            module="gold_tasks",
            function="log_prediction_to_gold",
            error=f"Gold logging failed: {exc}",
            latency_ms=latency,
        )
        raise self.retry(exc=exc)
