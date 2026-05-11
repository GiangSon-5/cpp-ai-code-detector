"""
engine/fingerprint_engine.py — Feature-based XAI using LightGBM and SHAP.
Replaces the old LIG heatmap with explainable features.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from src.fastapi_service.engine.feature_extractor.extractor import CppFeatureExtractorV8
from src.shared.data_contracts import FingerprintResult, ShapFeature
from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()

# Behavioral insights mapping
BEHAVIORAL_INSIGHTS = {
    "comment_ratio": {
        "human": "Sinh viên thường lười comment hoặc comment ngắn gọn. Đôi khi có comment rác.",
        "ai": "AI sinh ra comment chi tiết, chuẩn mực ngữ pháp và giải thích rõ ràng."
    },
    "empty_line_ratio": {
        "human": "Con người xuống dòng tùy tiện, để trống nhiều dòng liên tiếp do thói quen.",
        "ai": "AI phân chia hàm và khối lệnh bằng đúng 1 dòng trống cực kỳ nhất quán."
    },
    "tab_vs_space_ratio": {
        "human": "Code thường trộn lẫn lộn xộn giữa Tab và Space do copy-paste hoặc code nhóm.",
        "ai": "AI sinh code với định dạng thụt lề chuẩn mực (thường 100% Space)."
    },
    "trailing_space_ratio": {
        "human": "Người lập trình hay gõ thừa dấu cách ở cuối dòng mà không xóa đi.",
        "ai": "AI tối ưu văn bản sinh ra, rất hiếm khi để thừa khoảng trắng vô nghĩa."
    },
    "brace_style_consistency": {
        "human": "Mức độ đồng nhất kém, lúc mở ngoặc cùng dòng, lúc xuống dòng.",
        "ai": "Tính nhất quán cực cao, luôn tuân thủ một phong cách duy nhất."
    },
    "single_char_var_ratio": {
        "human": "Lạm dụng biến 1 ký tự (i, j, n) là thói quen kinh điển để code nhanh.",
        "ai": "AI thích đặt tên biến đầy đủ ý nghĩa theo nguyên tắc Clean Code."
    },
    "avg_identifier_length": {
        "human": "Độ dài tên biến trung bình thấp do thói quen viết tắt.",
        "ai": "Tên biến dài và mô tả chính xác chức năng."
    },
    "keyword_to_identifier_ratio": {
        "human": "Tỉ lệ cao do dùng ít biến tự định nghĩa mà lạm dụng cấu trúc cơ bản.",
        "ai": "Tạo ra nhiều cấu trúc, hàm phụ trợ làm tăng số định danh độc lập."
    },
    "avg_cyclomatic_complexity": {
        "human": "Thường dồn toàn bộ logic vào hàm main, for/if lồng nhau sâu hoắm.",
        "ai": "Chia nhỏ các hàm để giữ độ phức tạp thấp và dễ đọc hơn."
    },
    "halstead_difficulty": {
        "human": "Độ khó cao bất thường do sử dụng toán tử lặp lại thiếu tối ưu.",
        "ai": "Mức độ duy trì cân đối, tối ưu hóa để dễ hiểu nhất."
    },
    "total_includes": {
        "human": "Thường include hàng loạt thư viện thừa hoặc thiếu do thói quen.",
        "ai": "Include chính xác những thư viện cần thiết cho thuật toán."
    },
    "macro_count": {
        "human": "Đặc sản của sinh viên: lạm dụng macro (#define pb push_back).",
        "ai": "Rất ít lạm dụng macro vì đi ngược triết lý Modern C++."
    },
    "shannon_entropy": {
        "human": "Entropy thấp do copy-paste hoặc lặp lại cấu trúc cơ học.",
        "ai": "Entropy cao vì từ vựng phong phú, comment đa dạng."
    },
    "whitespace_entropy": {
        "human": "Sự ngẫu nhiên cao trong việc dùng Space/Tab ở các vị trí khác nhau.",
        "ai": "Tuân thủ nghiêm ngặt quy tắc khoảng trắng (entropy thấp)."
    }
}

class FingerprintEngine:
    def __init__(self):
        self.extractor = CppFeatureExtractorV8()
        
        # Load LightGBM model & artifacts from local_models/saved_models
        base_dir = os.path.join(settings.BASE_DIR, "local_models", "saved_models")
        
        try:
            self.model = joblib.load(os.path.join(base_dir, "LightGBM_Regulated.pkl"))
            self.scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
            self.final_features = joblib.load(os.path.join(base_dir, "final_features.pkl"))
            
            with open(os.path.join(base_dir, "baselines.json"), "r", encoding="utf-8") as f:
                self.baselines = json.load(f)
                
            self.explainer = shap.TreeExplainer(self.model)
            self._loaded = True
            logger.info(module="fingerprint_engine", function="__init__", message="Loaded LightGBM model and SHAP explainer successfully.")
        except Exception as exc:
            self._loaded = False
            logger.error(module="fingerprint_engine", function="__init__", error=f"Failed to load LightGBM artifacts: {exc}")

    @AppLogger.log_function(module="fingerprint_engine")
    def analyze(self, code: str) -> FingerprintResult | None:
        if not self._loaded:
            return None
            
        try:
            # 1. Trích xuất đặc trưng
            features = self.extractor.extract(code)
            
            # 2. Xử lý dữ liệu
            original_columns = self.scaler.feature_names_in_
            df_feat = pd.DataFrame([features])
            df_feat = df_feat.reindex(columns=original_columns, fill_value=0)
            
            # Scale
            scaled_feat = self.scaler.transform(df_feat)
            df_scaled = pd.DataFrame(scaled_feat, columns=original_columns)
            
            # Predict
            X_infer = df_scaled[self.final_features]
            pred_proba = self.model.predict_proba(X_infer)[0, 1]
            
            is_ai = pred_proba >= 0.5
            prediction_label = "AI GENERATED" if is_ai else "HUMAN WRITTEN"
            
            # 3. Tính SHAP
            shap_values = self.explainer.shap_values(X_infer)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # For AI class
            shap_values = shap_values[0] # First sample
            
            # 4. Gom nhóm kết quả SHAP
            shap_df = pd.DataFrame({
                'Feature': self.final_features,
                'SHAP': shap_values,
                'Abs_SHAP': np.abs(shap_values),
                'Value': [features.get(f, 0) for f in self.final_features]
            }).sort_values(by='Abs_SHAP', ascending=False)
            
            all_shap_features = []
            
            for _, row in shap_df.iterrows():
                f_name = row['Feature']
                f_val = row['Value']
                s_val = row['SHAP']
                
                direction = "AI" if s_val > 0 else "HUMAN"
                
                # Get baselines
                baseline = self.baselines.get(f_name, {"ai": 0.0, "human": 0.0})
                
                # Get insights
                insight_dict = BEHAVIORAL_INSIGHTS.get(f_name, {})
                insight = insight_dict.get(
                    direction.lower(), 
                    f"Đặc trưng này ({f_name}) có xu hướng kéo quyết định về phía {direction}."
                )
                
                shap_feat = ShapFeature(
                    name=f_name,
                    display_name=f_name.replace("_", " ").title(),
                    value=float(f_val),
                    shap_value=float(s_val),
                    direction=direction,
                    human_baseline=float(baseline["human"]),
                    ai_baseline=float(baseline["ai"]),
                    insight=insight
                )
                all_shap_features.append(shap_feat)
                
            top_features = all_shap_features[:5]
            
            # Generate Executive Summary
            categories = {
                "Layout & Formatting": ["comment_ratio", "empty_line_ratio", "avg_line_length", "max_line_length", "tab_vs_space_ratio", "trailing_space_ratio", "brace_style_consistency"],
                "Naming Conventions": ["avg_identifier_length", "identifier_length_variance", "single_char_var_ratio", "unique_identifier_ratio", "keyword_to_identifier_ratio"],
                "Structural Complexity": ["avg_cyclomatic_complexity", "num_functions", "avg_function_loc", "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs", "maintainability_index", "code_to_comment_ratio", "max_nesting_depth"],
                "Coding Habits": ["total_includes", "has_bits_stdc", "macro_count", "modern_cpp_ratio", "const_usage_ratio", "has_fast_io", "newline_style_ratio"],
                "Information Theory": ["shannon_entropy", "bigram_entropy", "whitespace_entropy"]
            }
            feat_to_group = {f: g for g, fs in categories.items() for f in fs}
            
            def get_group_vn(group):
                return {
                    "Layout & Formatting": "cách trình bày và định dạng (Layout)",
                    "Naming Conventions": "thói quen đặt tên biến (Naming)",
                    "Structural Complexity": "cấu trúc và độ phức tạp (Complexity)",
                    "Coding Habits": "các thói quen mã hóa (Habits)",
                    "Information Theory": "độ nhiễu loạn thông tin (Entropy)"
                }.get(group, "các thói quen chung")

            if len(top_features) >= 2:
                top_1 = top_features[0]
                top_2 = top_features[1]
                
                g1 = get_group_vn(feat_to_group.get(top_1.name))
                g2 = get_group_vn(feat_to_group.get(top_2.name))
                
                summary_text = f"Dựa trên phân tích hình thái (Fingerprint), đoạn mã này bộc lộ rất rõ bản chất của **{'thuật toán máy học' if is_ai else 'con người'}**."
                if is_ai:
                    summary_text += f" Quyết định này được hệ thống đưa ra phần lớn là do sự hoàn hảo bất thường trong **{g1}** ({top_1.display_name}), kết hợp với những đặc trưng máy móc ở **{g2}** ({top_2.display_name})."
                else:
                    summary_text += f" Sự ngẫu hứng đặc trưng của con người thể hiện rất rõ qua **{g1}** ({top_1.display_name}), cũng như những dấu vết để lại trong **{g2}** ({top_2.display_name})."
            else:
                summary_text = "Phân tích Fingerprint hoàn tất."
            
            return FingerprintResult(
                lgbm_score=float(pred_proba),
                lgbm_prediction=prediction_label,
                executive_summary=summary_text,
                top_features=top_features,
                all_features=all_shap_features
            )
            
        except Exception as exc:
            logger.error(module="fingerprint_engine", function="analyze", error=f"Analysis failed: {exc}")
            return None
