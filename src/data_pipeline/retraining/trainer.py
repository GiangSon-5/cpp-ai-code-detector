"""
retraining/trainer.py — Closed-loop retraining pipeline.

Ported from extracted_with_markdown.py Cells 6, 10.
Handles:
    - α optimization (fusion weight grid search)
    - Threshold optimization (PR curve)
    - MLflow logging to DagsHub
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.shared.logger import AppLogger

logger = AppLogger()


class RetrainingPipeline:
    """Automated retraining pipeline for the AI detection system."""

    @AppLogger.log_function(module="retraining")
    def optimize_alpha(
        self,
        bert_scores: np.ndarray,
        lgbm_scores: np.ndarray,
        labels: np.ndarray,
        step: float = 0.01,
    ) -> dict[str, Any]:
        """Grid search for optimal fusion α.

        Formula: fused = α * bert_score + (1-α) * lgbm_score

        Returns:
            dict with best_alpha, best_accuracy, all_results
        """
        from sklearn.metrics import accuracy_score

        best_alpha = 0.5
        best_acc = 0.0
        results = []

        for alpha_int in range(0, 101):
            alpha = alpha_int * step
            fused = alpha * bert_scores + (1 - alpha) * lgbm_scores
            preds = (fused >= 0.5).astype(int)
            acc = accuracy_score(labels, preds)

            results.append({"alpha": alpha, "accuracy": acc})

            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha

        return {
            "best_alpha": round(best_alpha, 2),
            "best_accuracy": round(best_acc, 4),
            "total_steps": len(results),
        }

    @AppLogger.log_function(module="retraining")
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> dict[str, Any]:
        """Optimize classification threshold via PR curve.

        Returns:
            dict with best_threshold, best_f1
        """
        from sklearn.metrics import precision_recall_curve, f1_score

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        best_f1 = 0.0
        best_thresh = 0.5

        for i, thresh in enumerate(thresholds):
            if i >= len(precisions) or i >= len(recalls):
                break
            p = precisions[i]
            r = recalls[i]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return {
            "best_threshold": round(float(best_thresh), 4),
            "best_f1": round(float(best_f1), 4),
        }

    @AppLogger.log_function(module="retraining")
    def log_to_mlflow(
        self,
        metrics: dict[str, float],
        params: dict[str, Any],
        model_name: str = "cpp_detector",
    ) -> bool:
        """Log experiment to MLflow on DagsHub.

        Returns:
            True if logging succeeded
        """
        try:
            import mlflow

            dagshub_url = "https://dagshub.com"
            mlflow.set_tracking_uri(dagshub_url)

            with mlflow.start_run(run_name=f"retrain_{model_name}"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

            logger.info(
                module="retraining",
                function="log_to_mlflow",
                message=f"MLflow logged: {model_name}",
                output_data=metrics,
            )
            return True

        except Exception as exc:
            logger.error(
                module="retraining",
                function="log_to_mlflow",
                error=f"MLflow logging failed: {exc}",
            )
            return False
