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
    "max_line_length": {
        "human": "Người viết hay để các dòng code dài lê thê không ngắt dòng.",
        "ai": "AI định dạng thụt lề chuẩn, ngắt dòng đều đặn để giữ code dễ đọc.",
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
    "avg_identifier_length": {
        "human": "Độ dài tên biến trung bình thấp do thói quen viết tắt.",
        "ai": "Tên biến dài và mô tả chính xác chức năng.",
    },
    "identifier_length_variance": {
        "human": "Sự chênh lệch lớn giữa tên biến cực ngắn và biến rất dài trong cùng một file.",
        "ai": "Tên định danh có độ dài đồng đều và nhất quán theo quy chuẩn clean code.",
    },
    "single_char_var_ratio": {
        "human": "Lạm dụng biến 1 ký tự (i, j, n) là thói quen kinh điển để code nhanh.",
        "ai": "AI thích đặt tên biến đầy đủ ý nghĩa theo nguyên tắc Clean Code.",
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
    "num_functions": {
        "human": "Số lượng hàm ít, thường viết code tuyến tính không chia module.",
        "ai": "Phân chia mã nguồn thành nhiều hàm độc lập thực hiện chức năng riêng biệt.",
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
    "halstead_effort": {
        "human": "Công sức gõ code/suy nghĩ phân phối không đồng đều, thiếu tối ưu toán tử.",
        "ai": "Công sức lập trình được tối ưu hóa thông qua các câu lệnh chuẩn mực.",
    },
    "halstead_bugs": {
        "human": "Mã nguồn tự do dễ phát sinh lỗi tiềm ẩn do cấu trúc lỏng lẻo.",
        "ai": "Khả năng phát sinh lỗi thấp nhờ cấu trúc và định kiểu chặt chẽ.",
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
    "has_bits_stdc": {
        "human": "Đặc sản của coder thi đấu giải thuật: dùng thư viện vạn năng bits/stdc++.",
        "ai": "Rất ít khi dùng bits/stdc++ vì đi ngược lại tiêu chuẩn phát triển dự án thực tế.",
    },
    "macro_count": {
        "human": "Đặc sản của sinh viên: lạm dụng macro (#define pb push_back).",
        "ai": "Rất ít lạm dụng macro vì đi ngược triết lý Modern C++.",
    },
    "modern_cpp_ratio": {
        "human": "Lập trình viên thường viết code theo chuẩn C++ cũ, ít dùng các tính năng mới.",
        "ai": "Sử dụng các cú pháp C++ hiện đại (auto, nullptr, constexpr, lambda) rất tự nhiên.",
    },
    "const_usage_ratio": {
        "human": "Hiếm khi khai báo const hoặc constexpr cho biến hay tham số hàm.",
        "ai": "Tích cực sử dụng const/constexpr để tối ưu hóa hiệu năng và bảo vệ dữ liệu.",
    },
    "has_fast_io": {
        "human": "Thường thêm dòng tối ưu nhập xuất (ios_base::sync_with_stdio) khi code thi đấu.",
        "ai": "Hầu như không sử dụng lệnh tối ưu nhập xuất này trong code ứng dụng thông thường.",
    },
    "newline_style_ratio": {
        "human": "Sử dụng pha trộn giữa '\\n' và 'endl' không đồng nhất.",
        "ai": "Lựa chọn và duy trì một phong cách xuống dòng cực kỳ nhất quán.",
    },
    "shannon_entropy": {
        "human": "Entropy thấp do copy-paste hoặc lặp lại cấu trúc cơ học.",
        "ai": "Entropy cao vì từ vựng phong phú, comment đa dạng.",
    },
    "bigram_entropy": {
        "human": "Entropy bigram thấp, biểu thị các cặp ký tự lặp lại thường xuyên theo thói quen cũ.",
        "ai": "Entropy bigram cân bằng, biểu thị sự đa dạng trong cách ghép từ và lệnh.",
    },
    "whitespace_entropy": {
        "human": "Sự ngẫu nhiên cao trong việc dùng Space/Tab ở các vị trí khác nhau.",
        "ai": "Tuân thủ nghiêm ngặt quy tắc khoảng trắng (entropy thấp).",
    },
    "class_count": {
        "human": "Hiếm khi thiết kế class cho các bài code giải thuật ngắn hoặc viết kiểu thủ tục.",
        "ai": "Cấu trúc hóa chương trình bằng class hướng đối tượng rất bài bản.",
    },
    "struct_count": {
        "human": "Lạm dụng struct để gom nhóm nhanh thuộc tính mà không dùng phương thức.",
        "ai": "Sử dụng struct đúng mục đích chứa dữ liệu thô, ưu tiên class cho hành vi.",
    },
    "has_inheritance": {
        "human": "Hầu như không sử dụng kế thừa class trong các bài tập code đơn giản.",
        "ai": "Chủ động thiết kế sơ đồ kế thừa lớp để tái sử dụng mã nguồn chuẩn OOP.",
    },
    "access_specifier_ratio": {
        "human": "Để mặc định phạm vi truy cập (thường là public) hoặc viết lộn xộn.",
        "ai": "Phân chia rõ ràng phạm vi truy cập (public, private, protected) để đóng gói dữ liệu.",
    },
    "virtual_override_ratio": {
        "human": "Không bao giờ dùng cơ chế đa hình ảo (virtual/override) trong code cơ bản.",
        "ai": "Sử dụng linh hoạt cơ chế virtual và override để xây dựng các phương thức đa hình.",
    },
    "using_std_ratio": {
        "human": "Hơn 80% con người dùng `using namespace std;` ở đầu file để đỡ phải gõ nhiều.",
        "ai": "Viết rõ tiền tố `std::` cho từng định danh để tránh ô nhiễm vùng tên (namespace pollution).",
    },
    "try_catch_ratio": {
        "human": "Bỏ qua hoàn toàn việc bắt lỗi/ngoại lệ bằng khối lệnh try-catch.",
        "ai": "Viết code phòng thủ tốt, chủ động đặt try-catch ở các vùng có nguy cơ phát sinh lỗi.",
    },
    "raw_pointer_ratio": {
        "human": "Sử dụng con trỏ thô (raw pointer `*`) để cấp phát động và quản lý bộ nhớ trực tiếp.",
        "ai": "Hạn chế tối đa con trỏ thô, ưu tiên dùng smart pointer hoặc tham chiếu an toàn.",
    },
    "getter_setter_ratio": {
        "human": "Thường cho thuộc tính ở dạng public để truy cập trực tiếp thay vì viết getter/setter.",
        "ai": "Tuân thủ nguyên lý OOP bằng cách đóng gói private thuộc tính và viết getter/setter đầy đủ.",
    },
    "std_prefix_ratio": {
        "human": "Hạn chế gõ tiền tố `std::` do lười hoặc đã dùng namespace global.",
        "ai": "Viết đầy đủ và nhất quán tiền tố `std::` cho tất cả các thư viện chuẩn C++.",
    },
    "cpp_cast_ratio": {
        "human": "Sử dụng ép kiểu kiểu C cổ điển `(int)value` vì dễ viết.",
        "ai": "Ưu tiên dùng các toán tử ép kiểu an toàn của C++ (`static_cast`, `const_cast`).",
    },
    "emoji_marker_score": {
        "human": "Thêm các ký hiệu cảm xúc hoặc icon cá nhân vào comment biểu lộ cảm xúc.",
        "ai": "Mã nguồn nghiêm túc, chuẩn dự án chuyên nghiệp, không bao giờ chứa emoji tự do.",
    },
}

FEATURE_DISPLAY_NAMES_VN: dict[str, str] = {
    "comment_ratio": "Mật độ chú thích",
    "empty_line_ratio": "Tỷ lệ dòng trắng",
    "avg_line_length": "Độ dài dòng trung bình",
    "max_line_length": "Độ dài dòng tối đa",
    "tab_vs_space_ratio": "Tỷ lệ Tab/Space",
    "trailing_space_ratio": "Khoảng trắng thừa cuối dòng",
    "brace_style_consistency": "Nhất quán mở ngoặc nhọn",
    "avg_identifier_length": "Độ dài tên định danh trung bình",
    "identifier_length_variance": "Biến động độ dài tên định danh",
    "single_char_var_ratio": "Tỷ lệ biến một ký tự",
    "unique_identifier_ratio": "Tỷ lệ định danh độc nhất",
    "keyword_to_identifier_ratio": "Tỷ lệ từ khóa/tên định danh",
    "avg_cyclomatic_complexity": "Độ phức tạp nhánh điều kiện trung bình",
    "num_functions": "Số lượng hàm",
    "avg_function_loc": "Độ dài hàm trung bình",
    "halstead_volume": "Độ phức tạp thuật toán (Halstead Volume)",
    "halstead_difficulty": "Độ khó lập trình (Halstead Difficulty)",
    "halstead_effort": "Công sức lập trình (Halstead Effort)",
    "halstead_bugs": "Dự báo số lỗi (Halstead Bugs)",
    "maintainability_index": "Chỉ số dễ bảo trì",
    "code_to_comment_ratio": "Tỷ lệ Code/Comment",
    "max_nesting_depth": "Độ sâu lồng nhau tối đa",
    "total_includes": "Số lượng thư viện import",
    "has_bits_stdc": "Sử dụng thư viện bits/stdc++.h",
    "macro_count": "Số lượng Macro khai báo",
    "modern_cpp_ratio": "Tỷ lệ sử dụng Modern C++",
    "const_usage_ratio": "Tỷ lệ sử dụng từ khóa const",
    "has_fast_io": "Tối ưu hóa nhập xuất Fast I/O",
    "newline_style_ratio": "Nhất quán phong cách xuống dòng",
    "shannon_entropy": "Độ đa dạng từ vựng (Entropy)",
    "bigram_entropy": "Độ biến động cặp ký tự",
    "whitespace_entropy": "Nhất quán khoảng trắng",
    "class_count": "Số lượng lớp (Class)",
    "struct_count": "Số lượng cấu trúc (Struct)",
    "has_inheritance": "Sử dụng tính kế thừa lớp",
    "access_specifier_ratio": "Tỷ lệ chỉ thị truy cập (OOP)",
    "virtual_override_ratio": "Tỷ lệ đa hình (virtual/override)",
    "using_std_ratio": "Sử dụng using namespace std",
    "try_catch_ratio": "Tỷ lệ khối try-catch bắt lỗi",
    "raw_pointer_ratio": "Tỷ lệ sử dụng con trỏ thô",
    "getter_setter_ratio": "Sử dụng Getter/Setter",
    "std_prefix_ratio": "Sử dụng tiền tố std::",
    "cpp_cast_ratio": "Tỷ lệ ép kiểu C++ an toàn",
    "emoji_marker_score": "Sử dụng biểu tượng cảm xúc (Emoji)",
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
            "max_nesting_depth",
        ],
        "Coding Habits": [
            "total_includes", "has_bits_stdc", "macro_count",
            "modern_cpp_ratio", "const_usage_ratio", "has_fast_io",
            "newline_style_ratio", "using_std_ratio", "try_catch_ratio",
            "raw_pointer_ratio", "std_prefix_ratio", "cpp_cast_ratio",
        ],
        "Information Theory": [
            "shannon_entropy", "bigram_entropy", "whitespace_entropy",
        ],
        "Comment & Style": [
            "comment_ratio", "code_to_comment_ratio", "emoji_marker_score",
        ],
        "OOP Structure": [
            "class_count", "struct_count", "has_inheritance",
            "access_specifier_ratio", "virtual_override_ratio",
            "getter_setter_ratio",
        ],
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
    "OOP Structure":        "thiết kế hướng đối tượng (OOP)",
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
                    display_name=FEATURE_DISPLAY_NAMES_VN.get(fname, fname.replace("_", " ").title()),
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

            # Calculate scorecard groupings (Surface vs Deep)
            surface_categories = {"Layout & Formatting", "Naming Conventions", "Comment & Style", "Coding Habits"}
            deep_categories = {"Structural Complexity", "OOP Structure", "Information Theory"}

            total_abs_shap = 0.0
            surface_abs_shap = 0.0
            deep_abs_shap = 0.0
            sum_shap_surface = 0.0
            sum_shap_deep = 0.0

            for feat in all_features:
                fname = feat.name
                sval = feat.shap_value
                abs_sval = abs(sval)
                cat = _CATEGORY_MAP.get(fname, "")

                total_abs_shap += abs_sval
                if cat in surface_categories:
                    surface_abs_shap += abs_sval
                    sum_shap_surface += sval
                elif cat in deep_categories:
                    deep_abs_shap += abs_sval
                    sum_shap_deep += sval
                else:
                    surface_abs_shap += abs_sval / 2.0
                    deep_abs_shap += abs_sval / 2.0
                    sum_shap_surface += sval / 2.0
                    sum_shap_deep += sval / 2.0

            if total_abs_shap > 0:
                surface_pct = round((surface_abs_shap / total_abs_shap) * 100, 1)
                deep_pct = round((deep_abs_shap / total_abs_shap) * 100, 1)
                total_pct = surface_pct + deep_pct
                if total_pct > 0:
                    surface_pct = round((surface_pct / total_pct) * 100, 1)
                    deep_pct = round(100.0 - surface_pct, 1)
            else:
                surface_pct = 50.0
                deep_pct = 50.0

            surface_label = "Đặc trưng AI" if sum_shap_surface > 0 else "Giống người"
            deep_label = "Đặc trưng AI" if sum_shap_deep > 0 else "Giống người"

            return FingerprintResult(
                lgbm_score=prob_ai,
                lgbm_prediction=label,
                executive_summary=summary,
                top_features=top5,
                all_features=all_features,
                surface_pct=surface_pct,
                surface_label=surface_label,
                deep_pct=deep_pct,
                deep_label=deep_label,
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
