"""
hybrid_evaluator.py — LightGBM Prediction + Fusion Score

Singleton HybridEvaluator loads LightGBM model, scaler, and feature names
from Google Drive. Provides predict_lgbm() and fuse_scores().
Deep logging on all operations.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd

from .config import PATH_SAVED_MODELS, FUSION_ALPHA, colab_log
from .feature_extractor import CppFeatureExtractorV8, strip_metadata_headers


class HybridEvaluator:
    """
    Quản lý mô hình LightGBM và tính toán Hybrid Fusion score.
    """
    def __init__(self):
        self.is_ready = False
        self.lgbm_model = None
        self.scaler = None
        self.feature_names = None
        self.extractor = CppFeatureExtractorV8()

    def load_resources(self):
        """Load các file pkl từ thư mục Saved_Models"""
        t0 = time.perf_counter()
        print("\n⚙️ [HYBRID EVALUATOR] Loading LightGBM & Scaler...")
        colab_log("info", "hybrid_evaluator", "load_resources", "Starting load")

        try:
            model_path = os.path.join(PATH_SAVED_MODELS, 'LightGBM_Regulated.pkl')
            scaler_path = os.path.join(PATH_SAVED_MODELS, 'scaler.pkl')
            feat_path = os.path.join(PATH_SAVED_MODELS, 'final_features.pkl')

            paths = [model_path, scaler_path, feat_path]
            missing = [p for p in paths if not os.path.exists(p)]

            if missing:
                msg = f"Missing files: {missing}"
                print(f"❌ [HYBRID EVALUATOR] {msg}")
                colab_log("warning", "hybrid_evaluator", "load_resources", msg)
                return

            self.lgbm_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feat_path)

            latency = (time.perf_counter() - t0) * 1000
            msg = f"Ready! Loaded {len(self.feature_names)} features."
            print(f"✅ [HYBRID EVALUATOR] {msg}")
            colab_log("info", "hybrid_evaluator", "load_resources", msg,
                      num_features=len(self.feature_names),
                      latency_ms=round(latency, 1))
            self.is_ready = True

        except Exception as e:
            print(f"❌ [HYBRID EVALUATOR] Load Error: {e}")
            colab_log("error", "hybrid_evaluator", "load_resources", str(e))

    def predict_lgbm(self, code_text):
        """Trích xuất 32 features, scale, chọn 20 features và predict"""
        t0 = time.perf_counter()

        if not self.is_ready:
            colab_log("warning", "hybrid_evaluator", "predict_lgbm",
                      "LightGBM not ready")
            return None

        try:
            # 1. Trích xuất 32 đặc trưng
            features_dict = self.extractor.extract(code_text)
            df_feat = pd.DataFrame([features_dict]).fillna(0)

            # 2. Scale
            X_all = df_feat[self.scaler.feature_names_in_]
            X_scaled = self.scaler.transform(X_all)

            # 3. Lấy 20 features (theo index)
            feature_names_list = list(self.feature_names)
            all_features_list = list(self.scaler.feature_names_in_)
            final_indices = [all_features_list.index(f) for f in feature_names_list]
            X_final = X_scaled[:, final_indices]

            # 4. Predict probability
            prob = self.lgbm_model.predict_proba(X_final)[0][1]
            result = float(prob)

            latency = (time.perf_counter() - t0) * 1000
            colab_log("info", "hybrid_evaluator", "predict_lgbm",
                      f"Prediction: {result:.4f}",
                      score=result, latency_ms=round(latency, 1))
            return result

        except Exception as e:
            print(f"⚠️ [HYBRID EVALUATOR] Predict Error: {e}")
            colab_log("error", "hybrid_evaluator", "predict_lgbm", str(e))
            import traceback
            traceback.print_exc()
            return None

    def fuse_scores(self, bert_score, lgbm_score):
        """Tính điểm Hybrid kết hợp RoBERTa và LightGBM"""
        if lgbm_score is None:
            colab_log("info", "hybrid_evaluator", "fuse_scores",
                      "LightGBM unavailable — using BERT only",
                      bert_score=bert_score)
            return bert_score

        fused = FUSION_ALPHA * bert_score + (1.0 - FUSION_ALPHA) * lgbm_score
        colab_log("info", "hybrid_evaluator", "fuse_scores",
                  f"Fused: {fused:.4f} (BERT={bert_score:.4f} × {FUSION_ALPHA} + LGBM={lgbm_score:.4f} × {1-FUSION_ALPHA:.2f})",
                  bert_score=bert_score, lgbm_score=lgbm_score,
                  fused_score=float(fused), alpha=FUSION_ALPHA)
        return float(fused)


# Singleton instance
hybrid_evaluator = HybridEvaluator()
