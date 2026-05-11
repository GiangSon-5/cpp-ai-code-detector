"""
hybrid_evaluator.py — LightGBM + Fusion Score cho Local GPU Server

Copy từ src/local_gpu_agent/hybrid_evaluator.py nhưng import từ local config.
Singleton instance: local_hybrid_evaluator
"""

import os
import joblib
import pandas as pd
from .config import PATH_SAVED_MODELS, FUSION_ALPHA, gpu_log

# Import feature extractor — tái sử dụng từ local_gpu_agent
import sys
_agent_dir = os.path.join(os.path.dirname(__file__), "..", "local_gpu_agent")
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from ..local_gpu_agent.feature_extractor import CppFeatureExtractorV8, strip_metadata_headers


class LocalHybridEvaluator:
    """LightGBM model wrapper với Fusion score cho Local GPU Server."""

    def __init__(self):
        self.is_ready     = False
        self.lgbm_model   = None
        self.scaler       = None
        self.feature_names = None
        self.extractor    = CppFeatureExtractorV8()

    def load_resources(self):
        """Load các file pkl từ PATH_SAVED_MODELS."""
        print("\n⚙️  [LOCAL GPU] Loading LightGBM & Scaler...")
        gpu_log("info", "hybrid_evaluator", "load_resources",
                f"Loading from: {PATH_SAVED_MODELS}")

        # Hỗ trợ cả 2 tên file (typo "Reculated" và đúng "Regulated")
        model_candidates = ["LightGBM_Regulated.pkl", "LightGBM_Reculated.pkl"]
        model_path = None
        for candidate in model_candidates:
            p = os.path.join(PATH_SAVED_MODELS, candidate)
            if os.path.exists(p):
                model_path = p
                break
        if model_path is None:
            model_path = os.path.join(PATH_SAVED_MODELS, "LightGBM_Regulated.pkl")  # fallback tên chuẩn để báo lỗi đúng
        scaler_path = os.path.join(PATH_SAVED_MODELS, "scaler.pkl")
        # Hỗ trợ cả final_features.pkl và feature_names.pkl
        feat_candidates = ["final_features.pkl", "feature_names.pkl"]
        feat_path = None
        for candidate in feat_candidates:
            p = os.path.join(PATH_SAVED_MODELS, candidate)
            if os.path.exists(p):
                feat_path = p
                break
        if feat_path is None:
            feat_path = os.path.join(PATH_SAVED_MODELS, "final_features.pkl")  # fallback

        if not all(os.path.exists(p) for p in [model_path, scaler_path, feat_path]):
            print(
                f"⚠️  [LOCAL GPU] LightGBM files không tìm thấy tại: {PATH_SAVED_MODELS}\n"
                "    → LightGBM sẽ bị bỏ qua, chỉ dùng RoBERTa score."
            )
            gpu_log("warning", "hybrid_evaluator", "load_resources",
                    "LightGBM files missing — will use RoBERTa score only",
                    path=PATH_SAVED_MODELS)
            return

        try:
            self.lgbm_model    = joblib.load(model_path)
            self.scaler        = joblib.load(scaler_path)
            self.feature_names = joblib.load(feat_path)
            self.is_ready      = True
            print(f"✅ [LOCAL GPU] LightGBM ready! {len(self.feature_names)} features.")
            gpu_log("info", "hybrid_evaluator", "load_resources",
                    "LightGBM loaded successfully",
                    num_features=len(self.feature_names))
        except Exception as exc:
            print(f"❌ [LOCAL GPU] LightGBM load error: {exc}")
            gpu_log("error", "hybrid_evaluator", "load_resources",
                    f"Load error: {exc}")

    def predict_lgbm(self, code_text: str) -> float | None:
        """Trích xuất features → Scale → Predict probability."""
        if not self.is_ready:
            return None
        try:
            features_dict = self.extractor.extract(code_text)
            df = pd.DataFrame([features_dict]).fillna(0)

            X_all    = df[self.scaler.feature_names_in_]
            X_scaled = self.scaler.transform(X_all)

            feat_list = list(self.feature_names)
            all_feat  = list(self.scaler.feature_names_in_)
            indices   = [all_feat.index(f) for f in feat_list]
            X_final   = X_scaled[:, indices]

            prob = float(self.lgbm_model.predict_proba(X_final)[0][1])
            return prob
        except Exception as exc:
            print(f"⚠️  [LOCAL GPU] LightGBM predict error: {exc}")
            gpu_log("warning", "hybrid_evaluator", "predict_lgbm", f"Error: {exc}")
            return None

    def fuse_scores(self, bert_score: float, lgbm_score: float | None) -> float:
        """Hybrid Fusion: α×BERT + (1-α)×LGBM. Fallback to BERT nếu LGBM fail."""
        if lgbm_score is None:
            return bert_score
        return float(FUSION_ALPHA * bert_score + (1.0 - FUSION_ALPHA) * lgbm_score)


# Singleton
local_hybrid_evaluator = LocalHybridEvaluator()
