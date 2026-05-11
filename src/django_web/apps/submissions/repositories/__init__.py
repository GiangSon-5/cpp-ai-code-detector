"""
submissions/repositories/bronze_repository.py — Repository pattern for Bronze layer.

All database operations go through this class.
Deep logging on every method.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional

from django.contrib.auth.models import User

from src.django_web.apps.submissions.models import BronzeSubmission
from src.shared.logger import AppLogger

logger = AppLogger()


class BronzeRepository:
    """CRUD operations for Bronze code submissions."""

    @staticmethod
    def save(
        user: Optional[User],
        raw_code: str,
        source: str = "web_upload",
        language: str = "cpp",
        filename: str = "",
        batch=None,
    ) -> tuple[BronzeSubmission, bool]:
        """Save a code submission. Returns (submission, created).

        If code_hash already exists, returns the existing record.
        """
        t0 = time.perf_counter()
        code_hash = hashlib.sha256(raw_code.strip().replace("\r\n", "\n").encode("utf-8")).hexdigest()
        file_size = len(raw_code.encode("utf-8"))

        try:
            obj, created = BronzeSubmission.objects.get_or_create(
                code_hash=code_hash,
                defaults={
                    "user": user,
                    "raw_code": raw_code,
                    "language": language,
                    "source": source,
                    "file_size_bytes": file_size,
                    "filename": filename,
                    "batch": batch,
                },
            )
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                module="bronze_repository",
                function="save",
                message=f"{'Created' if created else 'Found existing'} submission {code_hash[:12]}",
                input_data={
                    "code_hash": code_hash,
                    "user_id": user.id if user else None,
                    "source": source,
                    "file_size": file_size,
                },
                output_data={
                    "submission_id": obj.id,
                    "created": created,
                },
                latency_ms=latency,
            )
            return obj, created
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(
                module="bronze_repository",
                function="save",
                error=f"Failed to save submission: {exc}",
                input_data={"code_hash": code_hash},
                latency_ms=latency,
            )
            raise

    @staticmethod
    def get_by_hash(code_hash: str) -> Optional[BronzeSubmission]:
        """Lookup by code_hash."""
        t0 = time.perf_counter()
        try:
            obj = BronzeSubmission.objects.filter(code_hash=code_hash).first()
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                module="bronze_repository",
                function="get_by_hash",
                input_data={"code_hash": code_hash},
                output_data={"found": obj is not None},
                latency_ms=latency,
            )
            return obj
        except Exception as exc:
            logger.error(
                module="bronze_repository",
                function="get_by_hash",
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return None

    @staticmethod
    def get_user_history(user: User, limit: int = 50) -> list[BronzeSubmission]:
        """Get recent submissions for a user."""
        t0 = time.perf_counter()
        try:
            qs = BronzeSubmission.objects.filter(user=user).order_by("-timestamp")[:limit]
            results = list(qs)
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                module="bronze_repository",
                function="get_user_history",
                input_data={"user_id": user.id, "limit": limit},
                output_data={"count": len(results)},
                latency_ms=latency,
            )
            return results
        except Exception as exc:
            logger.error(
                module="bronze_repository",
                function="get_user_history",
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return []

    @staticmethod
    def update_result(code_hash: str, prediction: str, confidence: float, result_json: dict) -> bool:
        """Update cached prediction result on a submission."""
        t0 = time.perf_counter()
        try:
            updated = BronzeSubmission.objects.filter(code_hash=code_hash).update(
                prediction=prediction,
                confidence=confidence,
                result_json=result_json,
            )
            latency = (time.perf_counter() - t0) * 1000
            logger.info(
                module="bronze_repository",
                function="update_result",
                input_data={"code_hash": code_hash, "prediction": prediction},
                output_data={"rows_updated": updated},
                latency_ms=latency,
            )
            return updated > 0
        except Exception as exc:
            logger.error(
                module="bronze_repository",
                function="update_result",
                error=str(exc),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return False

    @staticmethod
    def count_all() -> int:
        """Total submission count."""
        return BronzeSubmission.objects.count()

    @staticmethod
    def list_recent(limit: int = 50) -> list[BronzeSubmission]:
        """Get most recent submissions (all users)."""
        return list(BronzeSubmission.objects.order_by("-timestamp")[:limit])
