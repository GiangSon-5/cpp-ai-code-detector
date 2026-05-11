"""
engine/fingerprint_engine.py — Feature-based XAI using LightGBM and SHAP.
Replaces the old LIG heatmap with human-readable feature-level explanations.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.fastapi_service.engine.feature_extractor.extractor import CppFeatureExtractorV8
from src.shared.config import settings
from src.shared.data_contracts import FingerprintResult, ShapFeature
from src.shared.logger import AppLogger

logger = AppLogger()

# ---------------------------------------------------------------------------
# Human-readable behavioral insights per feature (Vietnamese)
# ---------------------------------------------------------------------------
BEHAVIORAL_INSIGHTS: dict[str, dict[str, str]] = {
    "comment_ratio": {
        "human": "Sinh viên thường lười comment hoặc comment ngắn gọn. Đôi khi có comment rác.",
        "ai": "AI sinh ra comment chi tiết, chuẩn mực ngữ pháp và giải thích rõ ràng.",
    },
    "empty_line_ratio": {
        "human": "Con người xuống dòng tùy tiện, để trống nhiều dòng liên tiếp do thói quen.",
        "ai": "AI phân chia hàm và khối lệnh bằng đúng 1 dòng trống cực kỳ nhất quán.",
    },
    "avg_line_length": {
        "human": "Độ dài dòng biến động lớn, lúc rất ngắn lúc rất dài do phong cách không nhất quán.",
        "ai": "AI duy trì độ dài dòng vừa phải và đồng đều theo quy chuẩn 80-100 ký tự.",
    },
    "tab_vs_space_ratio": {
        "human": "Code thường trộn lẫn lộn xộn giữa Tab và Space do copy-paste hoặc code nhóm.",
        "ai": "AI sinh code với định dạng thụt lề chuẩn mực (thường 100% Space).",
    },
    "trailing_space_ratio": {
        "human": "Người lập trình hay gõ thừa dấu cách ở cuối dòng mà không xóa đi.",
        "ai": "AI tối ưu văn bản sinh ra, rất hiếm khi để thừa khoảng trắng vô nghĩa.",
    },
    "brace_style_consistency": {
        "human": "Mức độ đồng nhất kém, lúc mở ngoặc cùng dòng, lúc xuống dòng.",
        "ai": "Tính nhất quán cực cao, luôn tuân thủ một phong cách duy nhất.",
    },
    "single_char_var_ratio": {
        "human": "Lạm dụng biến 1 ký tự (i, j, n) là thói quen kinh điển để code nhanh.",
        "ai": "AI thích đặt tên biến đầy đủ ý nghĩa theo nguyên tắc Clean Code.",
    },
    "avg_identifier_length": {
        "human": "Độ dài tên biến trung bình thấp do thói quen viết tắt.",
        "ai": "Tên biến dài và mô tả chính xác chức năng.",
    },
    "unique_identifier_ratio": {
        "human": "Tái sử dụng cùng một tên biến nhiều lần trong các vòng lặp lồng nhau.",
        "ai": "Đặt tên riêng biệt, có ngữ nghĩa rõ ràng cho từng định danh.",
    },
    "keyword_to_identifier_ratio": {
        "human": "Tỉ lệ cao do dùng ít biến tự định nghĩa mà lạm dụng cấu trúc cơ bản.",
        "ai": "Tạo ra nhiều cấu trúc, hàm phụ trợ làm tăng số định danh độc lập.",
    },
    "avg_cyclomatic_complexity": {
        "human": "Thường dồn toàn bộ logic vào hàm main, for/if lồng nhau sâu hoắm.",
        "ai": "Chia nhỏ các hàm để giữ độ phức tạp thấp và dễ đọc hơn.",
    },
    "avg_function_loc": {
        "human": "Hàm dài, thường nhét toàn bộ xử lý vào 1 hàm duy nhất.",
        "ai": "Chia nhỏ hàm, mỗi hàm chỉ làm 1 việc (Single Responsibility).",
    },
    "halstead_volume": {
        "human": "Volume thấp do logic đơn giản, ít toán tử và toán hạng.",
        "ai": "Volume cao hơn, code phức tạp hơn với nhiều phép tính và biểu thức.",
    },
    "halstead_difficulty": {
        "human": "Độ khó cao bất thường do sử dụng toán tử lặp lại thiếu tối ưu.",
        "ai": "Mức độ duy trì cân đối, tối ưu hóa để dễ hiểu nhất.",
    },
    "maintainability_index": {
        "human": "Chỉ số bảo trì thấp do code phức tạp, thiếu comment và cấu trúc.",
        "ai": "Chỉ số bảo trì cao, code sạch sẽ và có thể bảo trì tốt.",
    },
    "code_to_comment_ratio": {
        "human": "Rất nhiều code nhưng rất ít comment, tỷ lệ code/comment cao.",
        "ai": "Cân đối giữa code và comment, giải thích logic phức tạp rõ ràng.",
    },
    "max_nesting_depth": {
        "human": "Độ lồng nhau sâu (4-5 cấp) do thói quen viết nested if/for.",
        "ai": "Độ lồng nhau nông, dùng early return hoặc tách hàm để giảm độ sâu.",
    },
    "total_includes": {
        "human": "Thường include hàng loạt thư viện thừa hoặc thiếu do thói quen.",
        "ai": "Include chính xác những thư viện cần thiết cho thuật toán.",
    },
    "macro_count": {
        "human": "Đặc sản của sinh viên: lạm dụng macro (#define pb push_back).",
        "ai": "Rất ít lạm dụng macro vì đi ngược triết lý Modern C++.",
    },
    "shannon_entropy": {
        "human": "Entropy thấp do copy-paste hoặc lặp lại cấu trúc cơ học.",
        "ai": "Entropy cao vì từ vựng phong phú, comment đa dạng.",
    },
    "whitespace_entropy": {
        "human": "Sự ngẫu nhiên cao trong việc dùng Space/Tab ở các vị trí khác nhau.",
        "ai": "Tuân thủ nghiêm ngặt quy tắc khoảng trắng (entropy thấp).",
    },
}

_CATEGORY_MAP: dict[str, str] = {
    f: g
    for g, feats in {
        "Layout & Formatting": [
            "empty_line_ratio", "avg_line_length", "max_line_length",
            "tab_vs_space_ratio", "trailing_space_ratio", "brace_style_consistency",
        ],
        "Naming Conventions": [
            "avg_identifier_length", "identifier_length_variance",
            "single_char_var_ratio", "unique_identifier_ratio",
            "keyword_to_identifier_ratio",
        ],
        "Structural Complexity": [
            "avg_cyclomatic_complexity", "num_functions", "avg_function_loc",
            "halstead_volume", "halstead_difficulty", "halstead_effort",
            "halstead_bugs", "maintainability_index",
            "code_to_comment_ratio", "max_nesting_depth",
        ],
        "Coding Habits": [
            "total_includes", "has_bits_stdc", "macro_count",
            "modern_cpp_ratio", "const_usage_ratio", "has_fast_io",
            "newline_style_ratio",
        ],
        "Information Theory": [
            "shannon_entropy", "bigram_entropy", "whitespace_entropy",
        ],
        "Comment & Style": ["comment_ratio", "code_to_comment_ratio"],
    }.items()
    for f in feats
}

_CATEGORY_VN: dict[str, str] = {
    "Layout & Formatting":  "cách trình bày và định dạng (Layout)",
    "Naming Conventions":   "thói quen đặt tên biến (Naming)",
    "Structural Complexity": "cấu trúc và độ phức tạp (Complexity)",
    "Coding Habits":        "các thói quen mã hóa (Habits)",
    "Information Theory":   "độ nhiễu loạn thông tin (Entropy)",
    "Comment & Style":      "phong cách bình luận (Comment)",
}


class FingerprintEngine:
    """Local LightGBM + SHAP engine — no GPU needed."""

    def __init__(self) -> None:
        self.extractor = CppFeatureExtractorV8()
        base_dir = str(settings.PROJECT_ROOT / "local_models" / "saved_models")

        try:
            self.model = joblib.load(os.path.join(base_dir, "LightGBM_Regulated.pkl"))
            self.scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
            self.final_features = joblib.load(os.path.join(base_dir, "final_features.pkl"))
            with open(os.path.join(base_dir, "baselines.json"), encoding="utf-8") as f:
                self.baselines: dict = json.load(f)
            self.explainer = shap.TreeExplainer(self.model)
            self._loaded = True
            logger.info(
                module="fingerprint_engine",
                function="__init__",
                message="FingerprintEngine loaded successfully.",
            )
        except Exception as exc:
            self._loaded = False
            logger.error(
                module="fingerprint_engine",
                function="__init__",
                error=f"Failed to load artifacts: {exc}",
            )

    def analyze(self, code: str) -> FingerprintResult | None:
        """Extract features → LightGBM predict → SHAP explain."""
        if not self._loaded:
            logger.error(
                module="fingerprint_engine",
                function="analyze",
                error="Model not loaded, skipping fingerprint analysis.",
            )
            return None

        try:
            # 1. Feature extraction
            features: dict = self.extractor.extract(code)

            # 2. Scale
            scaler_cols = list(self.scaler.feature_names_in_)
            df = pd.DataFrame([features]).reindex(columns=scaler_cols, fill_value=0)
            df_scaled = pd.DataFrame(
                self.scaler.transform(df), columns=scaler_cols
            )

            # 3. Predict
            X = df_scaled[self.final_features]
            prob_ai = float(self.model.predict_proba(X)[0, 1])
            is_ai = prob_ai >= 0.5
            label = "AI GENERATED" if is_ai else "HUMAN WRITTEN"

            # 4. SHAP
            raw_shap = self.explainer.shap_values(X)
            # LightGBM binary: returns array of shape (1, n_features)
            if isinstance(raw_shap, list):
                raw_shap = raw_shap[1]
            shap_vals: np.ndarray = np.asarray(raw_shap).flatten()

            feat_names = list(self.final_features)

            shap_df = pd.DataFrame({
                "Feature": feat_names,
                "SHAP": shap_vals,
                "AbsSHAP": np.abs(shap_vals),
                "Value": [features.get(f, 0.0) for f in feat_names],
            }).sort_values("AbsSHAP", ascending=False)

            # 5. Build ShapFeature objects
            all_features: list[ShapFeature] = []
            for _, row in shap_df.iterrows():
                fname = str(row["Feature"])
                fval  = float(row["Value"])
                sval  = float(row["SHAP"])
                direction = "AI" if sval > 0 else "HUMAN"
                baseline = self.baselines.get(fname, {"ai": 0.0, "human": 0.0})
                insights = BEHAVIORAL_INSIGHTS.get(fname, {})
                insight = insights.get(
                    direction.lower(),
                    f"Đặc trưng '{fname}' kéo quyết định về phía {direction}.",
                )
                all_features.append(ShapFeature(
                    name=fname,
                    display_name=fname.replace("_", " ").title(),
                    value=fval,
                    shap_value=sval,
                    direction=direction,
                    human_baseline=float(baseline.get("human", 0.0)),
                    ai_baseline=float(baseline.get("ai", 0.0)),
                    insight=insight,
                ))

            top5 = all_features[:5]

            # 6. Executive summary
            summary = self._make_summary(top5, is_ai)

            return FingerprintResult(
                lgbm_score=prob_ai,
                lgbm_prediction=label,
                executive_summary=summary,
                top_features=top5,
                all_features=all_features,
            )

        except Exception as exc:
            logger.error(
                module="fingerprint_engine",
                function="analyze",
                error=f"Analysis failed: {exc}",
            )
            return None

    def _make_summary(self, top: list[ShapFeature], is_ai: bool) -> str:
        if len(top) < 2:
            return "Phân tích Fingerprint hoàn tất."
        t1, t2 = top[0], top[1]
        g1 = _CATEGORY_VN.get(_CATEGORY_MAP.get(t1.name, ""), "đặc trưng chung")
        g2 = _CATEGORY_VN.get(_CATEGORY_MAP.get(t2.name, ""), "đặc trưng chung")
        nature = "thuật toán máy học" if is_ai else "con người"
        summary = (
            f"Dựa trên phân tích hình thái (Fingerprint), đoạn mã này bộc lộ rất rõ "
            f"bản chất của <b>{nature}</b>."
        )
        if is_ai:
            summary += (
                f" Quyết định này được hệ thống đưa ra phần lớn là do sự hoàn hảo bất thường "
                f"trong <b>{g1}</b> ({t1.display_name}), kết hợp với những đặc trưng máy móc "
                f"ở <b>{g2}</b> ({t2.display_name})."
            )
        else:
            summary += (
                f" Sự ngẫu hứng đặc trưng của con người thể hiện rất rõ qua <b>{g1}</b> "
                f"({t1.display_name}), cũng như những dấu vết để lại trong <b>{g2}</b> "
                f"({t2.display_name})."
            )
        return summary
