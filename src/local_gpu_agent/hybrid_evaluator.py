import os
import joblib
import pandas as pd
import numpy as np
from .config import PATH_SAVED_MODELS, FUSION_ALPHA
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
        print("\n⚙️ [HYBRID EVALUATOR] Loading LightGBM & Scaler...")
        try:
            model_path = os.path.join(PATH_SAVED_MODELS, 'LightGBM_Regulated.pkl')
            scaler_path = os.path.join(PATH_SAVED_MODELS, 'scaler.pkl')
            feat_path = os.path.join(PATH_SAVED_MODELS, 'final_features.pkl')
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, feat_path]):
                print(f"❌ [HYBRID EVALUATOR] Missing files in {PATH_SAVED_MODELS}. LightGBM won't be used.")
                return
                
            self.lgbm_model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(feat_path)
            
            print(f"✅ [HYBRID EVALUATOR] Ready! Loaded {len(self.feature_names)} features.")
            self.is_ready = True
        except Exception as e:
            print(f"❌ [HYBRID EVALUATOR] Load Error: {e}")
            
    def predict_lgbm(self, code_text):
        """Trích xuất 32 features, scale, chọn 20 features và predict"""
        if not self.is_ready:
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
            return float(prob)
        except Exception as e:
            print(f"⚠️ [HYBRID EVALUATOR] Predict Error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def fuse_scores(self, bert_score, lgbm_score):
        """Tính điểm Hybrid kết hợp RoBERTa và LightGBM"""
        if lgbm_score is None:
            return bert_score # Fallback if LightGBM fails
            
        fused = FUSION_ALPHA * bert_score + (1.0 - FUSION_ALPHA) * lgbm_score
        return float(fused)

# Singleton instance
hybrid_evaluator = HybridEvaluator()
