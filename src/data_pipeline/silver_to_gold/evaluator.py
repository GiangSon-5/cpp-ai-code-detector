"""
silver_to_gold/evaluator.py — Model evaluation + SHAP analysis.

Evaluates model performance on Silver data and writes Gold metrics.
Ported from extracted_with_markdown.py Cell 12.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from src.shared.logger import AppLogger

logger = AppLogger()


class SilverToGoldEvaluator:
    """Evaluate model on Silver data and produce Gold metrics."""

    @AppLogger.log_function(module="silver_to_gold")
    def evaluate_model(
        self,
        model_name: str,
        y_true: list[int],
        y_pred: list[int],
        y_proba: list[float],
    ) -> dict[str, Any]:
        """Compute evaluation metrics for a model.

        Returns:
            dict with accuracy, f1, precision, recall, auc_roc, etc.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics = {
            "model_name": model_name,
            "eval_date": datetime.utcnow().isoformat(),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "sample_count": len(y_true),
        }

        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0

        return metrics

    @AppLogger.log_function(module="silver_to_gold")
    def compute_shap_summary(
        self,
        model: Any,
        X_data: Any,
        feature_names: list[str],
        max_samples: int = 100,
    ) -> dict[str, float]:
        """Compute SHAP feature importance summary.

        Returns:
            dict mapping feature_name → mean(|SHAP value|)
        """
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_data[:max_samples])

            # For binary: shap_values[1] or shap_values
            if isinstance(shap_values, list):
                vals = shap_values[1]
            else:
                vals = shap_values

            import numpy as np
            mean_abs = np.abs(vals).mean(axis=0)

            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(mean_abs):
                    importance[name] = float(mean_abs[i])

            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        except Exception as exc:
            logger.error(
                module="silver_to_gold",
                function="compute_shap_summary",
                error=f"SHAP analysis failed: {exc}",
            )
            return {}
