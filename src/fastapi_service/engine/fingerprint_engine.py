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

    # ── Comment & Style ──────────────────────────────────────────────────
    "comment_ratio": {
        "human": (
            "Mã nguồn do người viết thường có rất ít chú thích, hoặc chỉ comment "
            "vài từ mang tính đánh dấu tạm thời như '// TODO' hay '// fix later'. "
            "Mật độ thấp này phản ánh thói quen ưu tiên chạy được code trước, "
            "giải thích sau."
        ),
        "ai": (
            "AI có xu hướng tạo ra chú thích chi tiết, rõ ràng cho hầu hết các "
            "khối logic — bao gồm cả những đoạn code đơn giản không thực sự cần "
            "giải thích. Mật độ comment đồng đều bất thường này là dấu hiệu đặc trưng."
        ),
    },
    "code_to_comment_ratio": {
        "human": (
            "Người lập trình thường để tỷ lệ code/comment rất cao — nhiều dòng "
            "xử lý nhưng gần như không có giải thích đi kèm, đặc biệt trong "
            "các bài tập thuật toán ngắn."
        ),
        "ai": (
            "AI duy trì tỷ lệ code/comment cân đối hơn: sau mỗi hàm hoặc khối "
            "logic phức tạp đều có phần mô tả chức năng, khiến tỷ lệ này thấp "
            "hơn đáng kể so với code thuần thủ công."
        ),
    },
    "emoji_marker_score": {
        "human": (
            "Một số lập trình viên — đặc biệt khi làm việc cá nhân — chèn ký "
            "tự cảm xúc hoặc ký hiệu đặc biệt vào comment để đánh dấu ('// ✅ done', "
            "'// ⚠️ cẩn thận'). Đây là dấu ấn cá nhân hiếm gặp trong code chuyên nghiệp."
        ),
        "ai": (
            "AI luôn tạo ra mã nguồn theo chuẩn dự án chuyên nghiệp — comment "
            "thuần văn bản, không chứa emoji hay ký hiệu cảm xúc tự do. "
            "Sự vắng mặt hoàn toàn của các ký tự này cũng là một tín hiệu."
        ),
    },

    # ── Layout & Formatting ───────────────────────────────────────────────
    "empty_line_ratio": {
        "human": (
            "Người viết code thường xuống dòng theo thói quen và cảm giác — "
            "đôi khi để trống 2–3 dòng liên tiếp giữa các hàm, đôi khi không "
            "có dòng trống nào giữa các khối logic dài."
        ),
        "ai": (
            "AI phân tách các hàm và khối lệnh bằng đúng một dòng trống, "
            "cực kỳ nhất quán xuyên suốt toàn bộ file — tạo ra tỷ lệ dòng "
            "trống ổn định và dễ nhận biết."
        ),
    },
    "avg_line_length": {
        "human": (
            "Độ dài dòng trong code thủ công biến động lớn: có dòng chỉ vài ký "
            "tự (khai báo biến đơn), lại có dòng dài bất thường do lười ngắt "
            "hoặc viết inline expression phức tạp."
        ),
        "ai": (
            "AI duy trì độ dài dòng trung bình đồng đều, thường trong khoảng "
            "60–90 ký tự — không quá ngắn, không quá dài — theo đúng quy "
            "chuẩn style guide mà không cần cưỡng ép."
        ),
    },
    "max_line_length": {
        "human": (
            "Khi viết vội, người lập trình hay để nguyên các biểu thức điều "
            "kiện dài hoặc chuỗi nối thành một dòng thay vì ngắt xuống, "
            "dẫn đến những dòng vượt quá 120 ký tự."
        ),
        "ai": (
            "AI tự động ngắt dòng ở những điểm hợp lý — sau toán tử, trước "
            "tham số — để không có dòng nào quá dài, giữ toàn bộ code trong "
            "giới hạn hiển thị tiêu chuẩn."
        ),
    },
    "tab_vs_space_ratio": {
        "human": (
            "Code do nhiều người viết hoặc copy-paste từ nhiều nguồn thường "
            "trộn lẫn Tab và Space trong cùng một file — một vấn đề quen thuộc "
            "trong các dự án nhóm hoặc code học tập."
        ),
        "ai": (
            "AI luôn sử dụng một kiểu thụt lề duy nhất (thường là 4 spaces) "
            "xuyên suốt toàn bộ file, không có ngoại lệ — tỷ lệ Tab/Space "
            "bằng 0 hoặc 1 tuyệt đối."
        ),
    },
    "trailing_space_ratio": {
        "human": (
            "Khoảng trắng thừa ở cuối dòng là dấu vết tự nhiên của quá trình "
            "gõ phím thủ công — thêm rồi xóa, chỉnh sửa nhiều lần mà không "
            "bao giờ dọn sạch."
        ),
        "ai": (
            "Văn bản do AI sinh ra không có khoảng trắng thừa ở cuối dòng — "
            "output được tạo ra đã ở trạng thái 'sạch' ngay từ đầu, "
            "không qua quá trình gõ và xóa của con người."
        ),
    },
    "brace_style_consistency": {
        "human": (
            "Phong cách mở ngoặc nhọn thường không nhất quán trong cùng một "
            "file: lúc mở cùng dòng với điều kiện, lúc xuống dòng mới — "
            "phụ thuộc vào thói quen cá nhân từng thời điểm viết."
        ),
        "ai": (
            "AI tuân thủ một phong cách ngoặc nhọn duy nhất xuyên suốt "
            "toàn bộ file mà không có ngoại lệ — mức độ nhất quán này "
            "gần như không thể đạt được bằng cách gõ tay."
        ),
    },
    "newline_style_ratio": {
        "human": (
            "Trong cùng một file, thường xuất hiện cả '\\n' lẫn 'endl' xen kẽ "
            "nhau — phản ánh việc code được viết ở nhiều thời điểm khác nhau "
            "hoặc chắp vá từ nhiều nguồn."
        ),
        "ai": (
            "AI chọn một phong cách xuống dòng duy nhất ('\\n' hoặc 'endl') "
            "và áp dụng nhất quán từ đầu đến cuối file — không có sự "
            "pha trộn ngẫu nhiên."
        ),
    },

    # ── Naming Conventions ────────────────────────────────────────────────
    "avg_identifier_length": {
        "human": (
            "Người lập trình hay viết tắt tên biến để gõ nhanh hơn: 'cnt' "
            "thay vì 'count', 'res' thay vì 'result', 'tmp' thay vì 'temporary'. "
            "Độ dài tên định danh trung bình vì vậy thường ngắn hơn."
        ),
        "ai": (
            "AI ưu tiên đặt tên đầy đủ, có nghĩa rõ ràng: 'studentCount', "
            "'calculatedResult', 'temporaryBuffer'. Độ dài trung bình cao hơn "
            "và phản ánh nguyên tắc clean code một cách máy móc."
        ),
    },
    "identifier_length_variance": {
        "human": (
            "Tên biến trong code thủ công có độ dài biến động lớn: biến vòng "
            "lặp cực ngắn ('i', 'j'), biến kết quả trung bình ('result'), "
            "biến cấu trúc dài hơn ('adjacencyMatrix') — sự đa dạng tự nhiên."
        ),
        "ai": (
            "AI đặt tên với độ dài đồng đều hơn nhiều — kể cả biến vòng lặp "
            "cũng được đặt tên như 'index' hoặc 'iterator'. Phương sai thấp "
            "bất thường là dấu hiệu của việc áp dụng quy tắc một cách cứng nhắc."
        ),
    },
    "single_char_var_ratio": {
        "human": (
            "Dùng biến một ký tự ('i', 'j', 'k', 'n', 'x') là thói quen cực "
            "kỳ phổ biến trong code thuật toán và vòng lặp — nhanh, quen tay, "
            "và hoàn toàn hợp lý trong ngữ cảnh toán học."
        ),
        "ai": (
            "AI hiếm khi dùng biến một ký tự, kể cả trong vòng lặp đơn giản. "
            "Thay vào đó sẽ là 'index', 'row', 'col' — tuân thủ clean code "
            "đến mức đôi khi trở nên dài dòng không cần thiết."
        ),
    },
    "unique_identifier_ratio": {
        "human": (
            "Người lập trình thường tái sử dụng cùng tên biến cho các biến "
            "tạm thời trong nhiều vòng lặp hoặc hàm khác nhau — 'temp', 'aux', "
            "'buf' xuất hiện nhiều lần với vai trò khác nhau."
        ),
        "ai": (
            "AI tạo ra tên định danh đa dạng và riêng biệt cho từng ngữ cảnh, "
            "khiến tỷ lệ định danh không trùng lặp cao hơn đáng kể so với "
            "code được viết thủ công trong thời gian ngắn."
        ),
    },
    "keyword_to_identifier_ratio": {
        "human": (
            "Code viết nhanh thường dựa nhiều vào các cấu trúc ngôn ngữ có "
            "sẵn (if, for, while, return) và ít khai báo hàm hay biến tự định "
            "nghĩa — dẫn đến tỷ lệ từ khóa/định danh cao."
        ),
        "ai": (
            "AI phân tách logic thành nhiều hàm con và cấu trúc phụ trợ, "
            "tạo ra nhiều định danh tự định nghĩa hơn — kéo tỷ lệ "
            "từ khóa/định danh xuống thấp hơn."
        ),
    },

    # ── Structural Complexity ─────────────────────────────────────────────
    "avg_cyclomatic_complexity": {
        "human": (
            "Người lập trình hay dồn toàn bộ logic vào một hoặc vài hàm lớn "
            "với nhiều nhánh if/else và vòng lặp lồng nhau — khiến độ phức "
            "tạp nhánh điều kiện trung bình trên mỗi hàm cao hơn."
        ),
        "ai": (
            "AI phân tách điều kiện phức tạp thành các hàm nhỏ chuyên biệt, "
            "giữ cho mỗi hàm chỉ xử lý một vài nhánh — độ phức tạp trung "
            "bình thấp và đồng đều hơn trên toàn bộ codebase."
        ),
    },
    "num_functions": {
        "human": (
            "Code học tập hoặc bài tập ngắn thường có ít hàm — đôi khi chỉ "
            "có hàm 'main' chứa toàn bộ logic, viết theo phong cách tuyến "
            "tính từ trên xuống dưới."
        ),
        "ai": (
            "AI tự động phân tách mã nguồn thành nhiều hàm nhỏ có chức năng "
            "rõ ràng — số lượng hàm nhiều hơn đáng kể, mỗi hàm chỉ "
            "thực hiện một nhiệm vụ cụ thể."
        ),
    },
    "avg_function_loc": {
        "human": (
            "Hàm trong code thủ công thường dài hơn vì người viết có xu "
            "hướng nhét nhiều xử lý vào cùng một chỗ thay vì tách ra — "
            "số dòng trung bình mỗi hàm vì thế cao hơn."
        ),
        "ai": (
            "AI tuân thủ nguyên tắc hàm ngắn gọn: mỗi hàm thường chỉ "
            "10–30 dòng, làm đúng một việc. Số dòng trung bình thấp và "
            "đồng đều là kết quả của việc áp dụng cứng nhắc quy tắc này."
        ),
    },
    "halstead_volume": {
        "human": (
            "Code đơn giản hóa logic bằng cách dùng ít toán tử và toán hạng "
            "độc lập — thường viết thẳng vào kết quả thay vì chia thành "
            "nhiều bước trung gian. Halstead Volume vì thế thường thấp hơn."
        ),
        "ai": (
            "AI tạo ra các biểu thức đầy đủ, rõ ràng với nhiều bước trung gian "
            "và tên biến mô tả — dẫn đến Halstead Volume cao hơn, phản ánh "
            "độ phong phú của từ vựng lập trình được dùng."
        ),
    },
    "halstead_difficulty": {
        "human": (
            "Phong cách code thủ công thường lặp lại cùng một toán tử "
            "trên nhiều toán hạng khác nhau (ví dụ: dùng '+' hoặc '==' "
            "nhiều lần) — điều này làm tăng Halstead Difficulty do "
            "tỷ lệ sử dụng toán tử không đa dạng."
        ),
        "ai": (
            "AI phân phối toán tử đa dạng hơn và giảm tần suất lặp lại "
            "của các toán tử thông dụng — Halstead Difficulty được giữ "
            "ở mức cân bằng, không quá cao cũng không quá thấp."
        ),
    },
    "halstead_effort": {
        "human": (
            "Code viết theo kiểu tuyến tính với các phép tính lặp lại "
            "thủ công thay vì dùng hàm tiện ích — khiến Halstead Effort "
            "phân phối không đều: vài đoạn rất phức tạp, phần còn lại rất đơn giản."
        ),
        "ai": (
            "AI phân bổ độ phức tạp đều hơn trên toàn bộ file thông qua "
            "việc dùng hàm phụ trợ và biểu thức chuẩn hóa — Halstead Effort "
            "có xu hướng đồng đều và dễ dự đoán hơn."
        ),
    },
    "halstead_bugs": {
        "human": (
            "Cấu trúc code lỏng lẻo — if lồng nhau sâu, logic dài trong "
            "một hàm, biến dùng lại nhiều mục đích — tạo ra nhiều điểm "
            "tiềm ẩn phát sinh lỗi theo ước lượng của Halstead."
        ),
        "ai": (
            "Việc chia nhỏ hàm và sử dụng biến có tên rõ ràng giúp "
            "AI tạo ra code có chỉ số lỗi ước tính thấp hơn — không "
            "phải vì AI không mắc lỗi, mà vì cấu trúc ít phức tạp hơn."
        ),
    },
    "maintainability_index": {
        "human": (
            "Hàm dài, lồng nhau sâu và thiếu chú thích là ba yếu tố chính "
            "kéo Maintainability Index xuống thấp — cả ba đều phổ biến "
            "trong code học tập và code viết nhanh dưới áp lực."
        ),
        "ai": (
            "Nhờ hàm ngắn, thụt lề nhất quán và comment đầy đủ, MI của "
            "code do AI tạo ra thường cao hơn ngưỡng 65 — mức được coi "
            "là 'dễ bảo trì' theo tiêu chuẩn công nghiệp."
        ),
    },
    "max_nesting_depth": {
        "human": (
            "Viết nhanh thường dẫn đến các khối if/for lồng nhau 4–5 cấp "
            "mà không dừng lại để tái cấu trúc — đây là dấu hiệu điển hình "
            "của code giải quyết vấn đề theo kiểu 'cứ chạy được là được'."
        ),
        "ai": (
            "AI áp dụng kỹ thuật early return và tách hàm để giữ độ lồng "
            "nhau tối đa ở 2–3 cấp — cấu trúc phẳng hơn và dễ đọc hơn, "
            "nhưng đôi khi cũng mang lại cảm giác 'quá chỉn chu'."
        ),
    },

    # ── Coding Habits ─────────────────────────────────────────────────────
    "total_includes": {
        "human": (
            "Người lập trình hay thêm thư viện theo kiểu 'phòng thủ' — "
            "include những gì mình quen dùng ngay từ đầu, kể cả khi "
            "chưa chắc cần dùng đến trong bài cụ thể này."
        ),
        "ai": (
            "AI include chính xác những thư viện cần thiết cho bài toán "
            "đang giải quyết — không thừa, không thiếu. Tính chính xác "
            "này đôi khi trở thành dấu hiệu nhận biết."
        ),
    },
    "has_bits_stdc": {
        "human": (
            "'#include <bits/stdc++.h>' là đặc sản của cộng đồng competitive "
            "programming — một dòng thay thế toàn bộ thư viện chuẩn, "
            "phổ biến đến mức nhiều người dùng theo phản xạ."
        ),
        "ai": (
            "AI hầu như không dùng <bits/stdc++.h> vì thư viện này không "
            "tương thích với nhiều compiler và đi ngược nguyên tắc 'include "
            "chỉ những gì cần thiết' của lập trình dự án thực tế."
        ),
    },
    "macro_count": {
        "human": (
            "Macro như '#define pb push_back', '#define ll long long', "
            "'#define INF 1e18' là công cụ gõ nhanh quen thuộc của "
            "người học qua competitive programming — tiện nhưng khó debug."
        ),
        "ai": (
            "AI sử dụng 'constexpr', 'using', hoặc viết đầy đủ tên "
            "thay vì macro — theo triết lý Modern C++ vốn xem macro "
            "là di sản của C cần tránh trong code mới."
        ),
    },
    "modern_cpp_ratio": {
        "human": (
            "Code học tập thường theo các ví dụ cũ hoặc giáo trình viết "
            "theo chuẩn C++03/C++11 — hiếm khi dùng 'auto', 'nullptr', "
            "range-based for, hay lambda trừ khi được yêu cầu rõ ràng."
        ),
        "ai": (
            "AI sử dụng các tính năng Modern C++ (auto, nullptr, constexpr, "
            "structured bindings, lambda) một cách tự nhiên và nhất quán — "
            "phản ánh việc được huấn luyện trên codebase hiện đại."
        ),
    },
    "const_usage_ratio": {
        "human": (
            "Khai báo 'const' thường bị bỏ qua trong code viết nhanh — "
            "người lập trình ít nghĩ đến tính bất biến của biến trừ khi "
            "bị yêu cầu hoặc gặp lỗi liên quan."
        ),
        "ai": (
            "AI tích cực đánh dấu 'const' cho mọi biến và tham số không "
            "thay đổi giá trị — hành vi này nhất quán đến mức tỷ lệ "
            "const/tổng biến cao hơn hẳn so với code thủ công."
        ),
    },
    "has_fast_io": {
        "human": (
            "'ios_base::sync_with_stdio(false); cin.tie(NULL);' là hai dòng "
            "quen thuộc trong code thi đấu — nhiều người thêm vào theo thói "
            "quen ngay cả khi bài không có yêu cầu xử lý input lớn."
        ),
        "ai": (
            "AI thường không thêm tối ưu hóa nhập/xuất này trong code "
            "ứng dụng thông thường — chỉ xuất hiện khi prompt yêu cầu "
            "rõ ràng về hiệu năng I/O."
        ),
    },
    "using_std_ratio": {
        "human": (
            "'using namespace std;' ở đầu file là lựa chọn của hơn 80% "
            "người học C++ — tiết kiệm gõ phím và đơn giản hóa code "
            "trong bối cảnh học tập và thi đấu."
        ),
        "ai": (
            "AI viết rõ tiền tố 'std::' cho từng định danh thay vì dùng "
            "namespace toàn cục — tránh ô nhiễm namespace là nguyên tắc "
            "chuẩn trong lập trình dự án thực tế."
        ),
    },
    "try_catch_ratio": {
        "human": (
            "Xử lý ngoại lệ thường bị bỏ qua hoàn toàn trong code học tập "
            "và thuật toán — người viết tập trung vào logic chính, "
            "coi việc bắt lỗi là overhead không cần thiết."
        ),
        "ai": (
            "AI chủ động đặt khối try-catch ở các vùng có nguy cơ phát "
            "sinh lỗi — đặc biệt khi làm việc với file, network, hay "
            "dynamic cast. Mật độ try-catch cao hơn là dấu hiệu rõ ràng."
        ),
    },
    "raw_pointer_ratio": {
        "human": (
            "Con trỏ thô ('int* p = new int[n]') vẫn phổ biến trong code "
            "học C++ vì được dạy trước smart pointer — quản lý bộ nhớ "
            "thủ công là kỹ năng cơ bản nhưng dễ gây memory leak."
        ),
        "ai": (
            "AI hạn chế tối đa con trỏ thô, ưu tiên 'unique_ptr', "
            "'shared_ptr', hoặc tham chiếu thông thường — phản ánh "
            "best practice của Modern C++ về quản lý bộ nhớ an toàn."
        ),
    },
    "std_prefix_ratio": {
        "human": (
            "Khi đã dùng 'using namespace std;', tiền tố 'std::' biến mất "
            "hoàn toàn. Khi không dùng namespace, đôi khi vẫn quên thêm "
            "tiền tố — tạo ra tỷ lệ sử dụng 'std::' không nhất quán."
        ),
        "ai": (
            "AI viết đầy đủ và nhất quán 'std::' cho tất cả định danh "
            "từ thư viện chuẩn, ngay cả những cái rất phổ biến như "
            "'std::endl', 'std::string' — tỷ lệ 100% hoặc gần tuyệt đối."
        ),
    },
    "cpp_cast_ratio": {
        "human": (
            "Ép kiểu theo phong cách C '(int)value' hoặc '(double)n' "
            "vẫn phổ biến vì ngắn gọn và dễ viết — người học thường "
            "không phân biệt giữa các loại cast trong C++."
        ),
        "ai": (
            "AI sử dụng các toán tử ép kiểu an toàn của C++ "
            "('static_cast<int>()', 'dynamic_cast<>()') — dài hơn nhưng "
            "rõ ý định hơn và được compiler kiểm tra chặt chẽ hơn."
        ),
    },

    # ── Information Theory ────────────────────────────────────────────────
    "shannon_entropy": {
        "human": (
            "Code viết trong môi trường áp lực (bài tập, thi cử) thường "
            "dùng từ vựng hạn chế và lặp lại cấu trúc — entropy thấp "
            "phản ánh sự đồng nhất về phong cách trong khoảng thời gian ngắn."
        ),
        "ai": (
            "AI tạo ra từ vựng phong phú hơn: tên biến đa dạng, comment "
            "mô tả chi tiết, cấu trúc câu lệnh nhiều dạng — entropy cao "
            "hơn là dấu hiệu của việc tổng hợp từ nhiều nguồn khác nhau."
        ),
    },
    "bigram_entropy": {
        "human": (
            "Cặp ký tự lặp đi lặp lại theo thói quen gõ phím cá nhân "
            "(vd: tên viết tắt yêu thích, pattern vòng lặp quen thuộc) "
            "làm giảm bigram entropy so với văn bản đa dạng hơn."
        ),
        "ai": (
            "Bigram entropy cân bằng hơn trong code do AI tạo ra — sự "
            "đa dạng trong cách ghép từ, lệnh và identifier phản ánh "
            "việc tổng hợp từ nhiều phong cách viết code khác nhau."
        ),
    },
    "whitespace_entropy": {
        "human": (
            "Cách dùng khoảng trắng của người lập trình mang tính ngẫu "
            "nhiên cao — lúc thêm space quanh toán tử, lúc không; "
            "entropy khoảng trắng cao phản ánh sự tùy tiện này."
        ),
        "ai": (
            "AI tuân thủ quy tắc khoảng trắng nhất quán (luôn space "
            "quanh toán tử, không space trước dấu chấm phẩy) — "
            "entropy thấp là kết quả của sự nhất quán hoàn hảo."
        ),
    },

    # ── OOP Structure ─────────────────────────────────────────────────────
    "class_count": {
        "human": (
            "Code giải thuật ngắn và bài tập thường được viết theo "
            "phong cách thủ tục — không có class nào, hoặc chỉ dùng "
            "struct để gom nhóm dữ liệu."
        ),
        "ai": (
            "AI tổ chức chương trình bằng class kể cả với bài toán đơn "
            "giản — tạo ra số lượng class nhiều hơn kỳ vọng, phản ánh "
            "việc áp dụng cứng nhắc nguyên tắc OOP."
        ),
    },
    "struct_count": {
        "human": (
            "Struct được dùng rộng rãi trong code thuật toán để gom nhóm "
            "các thuộc tính liên quan (tọa độ, cạnh đồ thị, thông tin node) "
            "theo phong cách C/C++ truyền thống."
        ),
        "ai": (
            "AI phân biệt rõ giữa struct (chứa dữ liệu thuần túy) và "
            "class (có hành vi và phương thức) — số lượng struct thường "
            "ít hơn và được dùng đúng mục đích hơn."
        ),
    },
    "has_inheritance": {
        "human": (
            "Kế thừa class hầu như không xuất hiện trong code bài tập "
            "và thuật toán — người học thường chưa đến giai đoạn thiết "
            "kế hệ thống phân cấp lớp."
        ),
        "ai": (
            "AI chủ động sử dụng kế thừa khi bài toán phù hợp, tạo ra "
            "hierarchy lớp rõ ràng — sự hiện diện của inheritance trong "
            "code đơn giản là dấu hiệu đáng chú ý."
        ),
    },
    "access_specifier_ratio": {
        "human": (
            "Phân chia public/private/protected thường không được chú "
            "trọng — many người để toàn bộ thuộc tính public cho "
            "tiện truy cập trực tiếp mà không nghĩ đến encapsulation."
        ),
        "ai": (
            "AI phân chia access specifier rõ ràng theo nguyên tắc OOP: "
            "private cho dữ liệu nội bộ, public cho interface — "
            "tỷ lệ sử dụng access specifier cao và nhất quán."
        ),
    },
    "virtual_override_ratio": {
        "human": (
            "Cơ chế đa hình ảo (virtual/override) hầu như không xuất "
            "hiện trong code học tập và thuật toán — đây là tính năng "
            "nâng cao dành cho thiết kế hệ thống phức tạp."
        ),
        "ai": (
            "AI dùng virtual và override khi thiết kế class hierarchy, "
            "kể cả trong những ví dụ không nhất thiết cần đến — "
            "sự hiện diện của chúng trong code đơn giản là tín hiệu rõ ràng."
        ),
    },
    "getter_setter_ratio": {
        "human": (
            "Thay vì viết getter/setter, người lập trình thường để "
            "thuộc tính public và truy cập trực tiếp — nhanh hơn "
            "khi viết, nhưng phá vỡ nguyên tắc encapsulation."
        ),
        "ai": (
            "AI viết getter/setter đầy đủ cho hầu hết thuộc tính private "
            "kể cả khi không cần thiết — mật độ cao của getX()/setX() "
            "là một trong những dấu hiệu OOP cứng nhắc đặc trưng của AI."
        ),
    },
}

# ---------------------------------------------------------------------------
# Feature display names (Vietnamese)
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES_VN: dict[str, str] = {
    # Comment & Style
    "comment_ratio":             "Mật độ chú thích",
    "code_to_comment_ratio":     "Tỷ lệ code / chú thích",
    "emoji_marker_score":        "Dấu ấn cá nhân (emoji/ký hiệu)",

    # Layout & Formatting
    "empty_line_ratio":          "Tỷ lệ dòng trắng",
    "avg_line_length":           "Độ dài dòng trung bình",
    "max_line_length":           "Độ dài dòng tối đa",
    "tab_vs_space_ratio":        "Kiểu thụt lề (Tab / Space)",
    "trailing_space_ratio":      "Khoảng trắng thừa cuối dòng",
    "brace_style_consistency":   "Tính nhất quán ngoặc nhọn",
    "newline_style_ratio":       "Tính nhất quán xuống dòng",

    # Naming Conventions
    "avg_identifier_length":         "Độ dài tên định danh trung bình",
    "identifier_length_variance":    "Độ biến động độ dài tên định danh",
    "single_char_var_ratio":         "Tỷ lệ biến một ký tự",
    "unique_identifier_ratio":       "Tỷ lệ định danh không trùng lặp",
    "keyword_to_identifier_ratio":   "Tỷ lệ từ khóa / định danh tự định nghĩa",

    # Structural Complexity
    "avg_cyclomatic_complexity": "Độ phức tạp nhánh điều kiện trung bình",
    "num_functions":             "Số lượng hàm",
    "avg_function_loc":          "Số dòng trung bình mỗi hàm",
    "halstead_volume":           "Độ phong phú từ vựng (Halstead Volume)",
    "halstead_difficulty":       "Mức độ phức tạp toán tử (Halstead Difficulty)",
    "halstead_effort":           "Ước tính công sức lập trình (Halstead Effort)",
    "halstead_bugs":             "Dự báo lỗi tiềm ẩn (Halstead Bugs)",
    "maintainability_index":     "Chỉ số khả năng bảo trì",
    "max_nesting_depth":         "Độ sâu lồng nhau tối đa",

    # Coding Habits
    "total_includes":      "Số lượng thư viện được import",
    "has_bits_stdc":       "Dùng <bits/stdc++.h>",
    "macro_count":         "Số lượng macro (#define)",
    "modern_cpp_ratio":    "Mức độ dùng tính năng Modern C++",
    "const_usage_ratio":   "Tần suất dùng từ khóa const",
    "has_fast_io":         "Tối ưu hóa nhập/xuất (Fast I/O)",
    "using_std_ratio":     "Dùng 'using namespace std'",
    "try_catch_ratio":     "Tần suất khối xử lý ngoại lệ (try-catch)",
    "raw_pointer_ratio":   "Tỷ lệ sử dụng con trỏ thô",
    "std_prefix_ratio":    "Tần suất viết tiền tố std::",
    "cpp_cast_ratio":      "Tần suất dùng ép kiểu C++ an toàn",

    # Information Theory
    "shannon_entropy":    "Độ đa dạng từ vựng (Shannon Entropy)",
    "bigram_entropy":     "Độ biến động cặp ký tự (Bigram Entropy)",
    "whitespace_entropy": "Tính nhất quán khoảng trắng",

    # OOP Structure
    "class_count":              "Số lượng lớp (class)",
    "struct_count":             "Số lượng cấu trúc (struct)",
    "has_inheritance":          "Sử dụng kế thừa lớp",
    "access_specifier_ratio":   "Tỷ lệ chỉ thị truy cập (public/private/protected)",
    "virtual_override_ratio":   "Tỷ lệ sử dụng đa hình (virtual/override)",
    "getter_setter_ratio":      "Tần suất dùng getter/setter",
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
