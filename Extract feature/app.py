import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import shap
from core.feature_extractor.extractor import CppFeatureExtractorV8
from utils.shared.helpers import strip_metadata_headers

# Cấu hình trang
st.set_page_config(
    page_title="C++ Code Fingerprint Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện Premium
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextArea textarea {
        font-family: 'Fira Code', monospace;
        font-size: 14px;
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3d425c;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .human-box {
        background-color: #1c3d33;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2ecc71;
    }
    .ai-box {
        background-color: #3d1c1c;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #e74c3c;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_extractor():
    return CppFeatureExtractorV8()

@st.cache_resource
def load_models():
    model = joblib.load("models/zoo/LightGBM_Regulated.pkl")
    scaler = joblib.load("models/zoo/scaler.pkl")
    final_features = joblib.load("models/zoo/final_features.pkl")
    import json
    with open("models/zoo/baselines.json", "r", encoding="utf-8") as f:
        baselines = json.load(f)
    return model, scaler, final_features, baselines

BEHAVIORAL_INSIGHTS = {
    # Layout & Formatting
    "comment_ratio": {
        "human": "Sinh viên thường lười comment hoặc comment rất ngắn gọn bằng tiếng Việt lóng ngóng. Đôi khi có những comment rác (comment-out code cũ).",
        "ai": "AI thường sinh ra các comment rất chi tiết, chuẩn mực ngữ pháp, giải thích rõ ràng từng khối code nhỏ và hiếm khi để lại code thừa bị comment."
    },
    "empty_line_ratio": {
        "human": "Con người thường xuống dòng tùy tiện, có thể để trống nhiều dòng liên tiếp do thói quen nhìn hoặc copy-paste lộn xộn.",
        "ai": "AI thường phân chia các hàm, các block code bằng đúng 1 dòng trống một cách cực kỳ nhất quán và gọn gàng."
    },
    "tab_vs_space_ratio": {
        "human": "Đoạn code có sự trộn lẫn lộn xộn giữa dấu Tab và Space. Đây là lỗi định dạng kinh điển của con người khi làm việc nhóm hoặc copy từ web.",
        "ai": "AI luôn sinh ra code với định dạng thụt lề chuẩn mực (thường là 100% Space). Sự hoàn hảo tuyệt đối này là đặc trưng của máy móc."
    },
    "trailing_space_ratio": {
        "human": "Người lập trình thường gõ thừa dấu cách ở cuối dòng khi code nhanh mà không xóa đi.",
        "ai": "AI tạo ra văn bản một cách tối ưu, rất hiếm khi sinh ra các khoảng trắng vô nghĩa ở cuối dòng."
    },
    "brace_style_consistency": {
        "human": "Mức độ đồng nhất kém. Người viết có thể lúc thì mở ngoặc nhọn `{` ở cùng dòng, lúc thì xuống dòng (đặc biệt khi copy-paste).",
        "ai": "Tính nhất quán cực cao, AI thường chọn đúng 1 style (như Allman hoặc K&R) và tuân thủ nó từ đầu đến cuối."
    },
    
    # Naming Conventions
    "single_char_var_ratio": {
        "human": "Việc lạm dụng nhiều biến 1 ký tự (i, j, n, x, s) để làm toán hay chạy vòng lặp là thói quen kinh điển để code cho nhanh.",
        "ai": "AI được huấn luyện dựa trên nguyên tắc Clean Code nên có xu hướng đặt tên biến đầy đủ ý nghĩa (VD: sum_of_array) thay vì dùng biến viết tắt."
    },
    "avg_identifier_length": {
        "human": "Độ dài tên biến trung bình thấp do thói quen viết tắt.",
        "ai": "Tên biến thường dài và mô tả chính xác chức năng do AI chọn từ vựng phong phú và tuân thủ chuẩn mực."
    },
    "keyword_to_identifier_ratio": {
        "human": "Tỉ lệ này thường cao do lập trình viên dùng ít biến tự định nghĩa mà lạm dụng nhiều cấu trúc cơ bản.",
        "ai": "AI thường tạo ra các cấu trúc đối tượng, hàm phụ trợ phong phú làm tăng số lượng định danh (identifier)."
    },

    # Structural Complexity
    "avg_cyclomatic_complexity": {
        "human": "Thường dồn toàn bộ logic vào hàm `main` với các vòng lặp `for` và `if-else` lồng nhau sâu hoắm, độ phức tạp lộn xộn.",
        "ai": "AI có thói quen chia nhỏ các hàm để giảm độ phức tạp lặp lại, giữ code có độ phức tạp thấp và dễ đọc hơn."
    },
    "halstead_difficulty": {
        "human": "Độ khó cao bất thường do cách sử dụng toán tử và biến lặp lại thiếu tối ưu, người đọc khó theo dõi luồng dữ liệu.",
        "ai": "Mức Halstead Difficulty thường được duy trì ở mức cân đối, tối ưu hóa để đạt hiệu năng hiểu tốt nhất."
    },

    # Habits & Idioms
    "total_includes": {
        "human": "Thường include hàng loạt thư viện cơ bản do thói quen, hoặc chỉ dùng `<iostream>` và code mọi thứ thủ công.",
        "ai": "Include chính xác, vừa đủ những thư viện cần thiết cho các hàm thuật toán cụ thể (VD: `<algorithm>`, `<vector>`)."
    },
    "macro_count": {
        "human": "Đặc sản của sinh viên thi đấu thuật toán (Competitive Programming): lạm dụng hàng loạt macro như `#define pb push_back`.",
        "ai": "Ít khi lạm dụng macro vì nó đi ngược với triết lý Modern C++. (Trừ phi AI bị yêu cầu ép buộc)."
    },
    "has_bits_stdc": {
        "human": "Sử dụng `#include <bits/stdc++.h>` là thói quen kinh điển của con người để đỡ phải nhớ nhiều thư viện.",
        "ai": "AI thường tránh dùng thư viện gom chung này trong môi trường chuyên nghiệp, nó thích include rõ ràng từng module hơn."
    },
    "has_fast_io": {
        "human": "Dùng `ios::sync_with_stdio(0)` là kỹ thuật 'học vẹt' rất phổ biến để tăng tốc độ I/O khi nộp bài.",
        "ai": "AI hiếm khi tự động thêm các lệnh tối ưu tốc độ này nếu không bị yêu cầu cụ thể."
    },

    # Information Theory
    "shannon_entropy": {
        "human": "Entropy thấp (tính ngẫu nhiên thấp) do việc copy-paste nhiều dòng code giống hệt nhau hoặc lặp lại cấu trúc một cách cơ học.",
        "ai": "Entropy thường cao hơn vì từ vựng nó sử dụng rất đa dạng, comment phong phú, cấu trúc không lặp lại thừa thãi."
    },
    "whitespace_entropy": {
        "human": "Sự ngẫu nhiên cao trong việc dùng Space và Tab ở các vị trí khác nhau (trước toán tử, sau dấu phẩy...) phản ánh sự tùy hứng.",
        "ai": "AI tuân thủ nghiêm ngặt quy tắc khoảng trắng (vd: luôn có 1 space quanh toán tử `=`), khiến chuỗi whitespace mất đi tính ngẫu nhiên (Entropy thấp)."
    }
}

FEATURE_DETAILS = {
    # Layout
    "comment_ratio": {
        "desc": "Tỷ lệ ký tự chú thích trên tổng ký tự.",
        "formula": r"\text{Ratio} = \frac{\text{Total Comment Chars}}{\text{Total File Chars}}",
        "in": "// HI\nint x;", 
        "calc": [
            "Ký tự chú thích: '// HI' (5 ký tự)",
            "Ký tự ngắt dòng: '\\n' (1 ký tự)",
            "Ký tự mã nguồn: 'int x;' (6 ký tự)",
            "Tổng cộng file: 5 + 1 + 6 = 12 ký tự",
            "Phép tính: 5 / 12 = 0.4167"
        ],
        "out": "0.4167"
    },
    "empty_line_ratio": {
        "desc": "Tỷ lệ dòng trống trên tổng số dòng.",
        "formula": r"\text{Ratio} = \frac{\text{Empty Lines}}{\text{Total Lines}}",
        "in": "int x;\n\nint y;",
        "calc": ["1 dòng trống", "3 dòng (tổng cộng)", "Phép tính: 1 / 3 = 0.3333"],
        "out": "0.3333"
    },
    "avg_line_length": {
        "desc": "Độ dài trung bình một dòng code (không tính dòng trống).",
        "formula": r"\text{Avg} = \frac{\sum \text{len(line)}}{\text{Num Lines}}",
        "in": "int a;\nreturn a;",
        "calc": ["Dòng 1: 6 ký tự", "Dòng 2: 9 ký tự", "Trung bình: (6 + 9) / 2 = 7.5"],
        "out": "7.5"
    },
    "max_line_length": {
        "desc": "Độ dài của dòng dài nhất trong file.",
        "in": "int a;\nlong long b = 100;",
        "calc": ["Dòng 1: 6", "Dòng 2: 18", "Kết quả: max(6, 18) = 18"],
        "out": "18.0"
    },
    "tab_vs_space_ratio": {
        "desc": "Tỷ lệ sử dụng phím Tab so với tổng số ký tự khoảng trắng (Tab + Space).",
        "formula": r"\text{Ratio} = \frac{\text{Count(Tab)}}{\text{Count(Tab)} + \text{Count(Space)}}",
        "in": "	int a; // 1 tab ở đầu\n		int b; // 2 tab ở đầu\n    int c; // 4 dấu cách (space)",
        "calc": [
            "Dòng 1: 1 Tab",
            "Dòng 2: 2 Tab",
            "Dòng 3: 4 Space",
            "Tổng: 3 Tab và 4 Space",
            "Phép tính: 3 / (3 + 4) = 0.4286"
        ],
        "out": "0.4286"
    },
    "trailing_space_ratio": {
        "desc": "Tỷ lệ các dòng có khoảng trắng (Space/Tab) thừa ở cuối dòng.",
        "formula": r"\text{Ratio} = \frac{\text{Số dòng có khoảng trắng thừa}}{\text{Tổng số dòng (tất cả)}}",
        "in": "int a;   \n\nreturn 0;",
        "calc": [
            "Dòng 1: 'int a;   ' (Có khoảng trắng thừa) ➔ Đếm 1",
            "Dòng 2: '' (Dòng trống) ➔ Không có khoảng trắng thừa ➔ Đếm 0",
            "Dòng 3: 'return 0;' ➔ Không có khoảng trắng thừa ➔ Đếm 0",
            "Tổng số dòng: 3",
            "Phép tính: 1 / 3 = 0.3333"
        ],
        "out": "0.3333"
    },
    "brace_style_consistency": {
        "desc": "Độ nhất quán trong việc sử dụng phong cách đặt dấu ngoặc {.",
        "formula": r"\text{Consistency} = | \frac{\text{Số dấu Allman}}{\text{Tổng số dấu}} - 0.5 | \times 2",
        "in": "1: if(x > 0) {  // Dấu { cùng dòng (K&R)\n2:     ...\n3: }\n4: if(x < 0) \n5: {            // Dấu { xuống dòng (Allman)\n6:     ...\n7: }",
        "calc": [
            "Bước 1: Kiểm tra dòng 1 ➔ Thấy dấu { cùng dòng ➔ Kiểu K&R",
            "Bước 2: Kiểm tra dòng 5 ➔ Thấy dấu { đứng riêng ➔ Kiểu Allman",
            "Bước 3: Tổng số dấu { tìm thấy = 2 (Dòng 1 và 5)",
            "Bước 4: Số dấu kiểu Allman = 1 (Dòng 5)",
            "Bước 5: Áp dụng công thức ➔ |(1/2) - 0.5| * 2 = 0.0",
            "Kết luận: Do dùng 2 kiểu khác nhau nên độ nhất quán = 0"
        ],
        "out": "0.0"
    },
    
    # Naming
    "avg_identifier_length": {
        "desc": "Độ dài trung bình các định danh (biến, hàm, class).",
        "formula": r"\text{Avg} = \frac{\sum \text{len(id)}}{\text{Count(id)}}",
        "in": "int count; int i;",
        "calc": ["Biến 'count': 5", "Biến 'i': 1", "Trung bình: (5 + 1) / 2 = 3.0"],
        "out": "3.0"
    },
    "identifier_length_variance": {
        "desc": "Phương sai độ dài định danh.",
        "formula": r"\sigma^2 = \frac{\sum (x_i - \mu)^2}{N}",
        "in": "int a; int count;",
        "calc": ["Độ dài: 1 và 5", "Trung bình: 3", "Phương sai: ((1-3)^2 + (5-3)^2) / 2 = 4.0"],
        "out": "4.0"
    },
    "single_char_var_ratio": {
        "desc": "Tỷ lệ dùng biến 1 ký tự (i, n, x).",
        "in": "int i, n, count;",
        "calc": ["Biến 1 ký tự: i, n (2)", "Tổng biến: i, n, count (3)", "Phép tính: 2 / 3 = 0.6667"],
        "out": "0.6667"
    },
    "unique_identifier_ratio": {
        "desc": "Tỷ lệ tên định danh duy nhất / tổng số định danh.",
        "in": "int a, b; a=b;",
        "calc": ["Unique: a, b (2)", "Total: a, b, a, b (4)", "Phép tính: 2 / 4 = 0.5"],
        "out": "0.5"
    },
    "keyword_to_identifier_ratio": {
        "desc": "Tỷ lệ từ khóa C++ (int, for...) / tên định danh.",
        "in": "int a = 10;",
        "calc": ["Từ khóa: int (1)", "Định danh: a (1)", "Phép tính: 1 / 1 = 1.0"],
        "out": "1.0"
    },

    # Structural
    "avg_cyclomatic_complexity": {
        "desc": "Đo lường số lượng các con đường độc lập trong code.",
        "formula": r"CC = E - N + 2P",
        "in": "void f() { if(a) x=1; else x=2; return x; }\nvoid g() { return; }",
        "calc": [
            "🔍 Chi tiết N, E, P đến từ đâu:",
            "1. Xác định Đỉnh (N=5):",
            "   - Hàm f: (1) Lệnh if, (2) Lệnh x=1, (3) Lệnh x=2, (4) Lệnh return",
            "   - Hàm g: (5) Lệnh return",
            "   - *Ghi chú: 'else' chỉ là từ khóa chỉ hướng đi, không phải một lệnh xử lý nên không tính là đỉnh riêng.",
            "2. Xác định Cạnh (E=4):",
            "   - (1->2) Đường đi khi if đúng",
            "   - (1->3) Đường đi khi if sai (nhánh else)",
            "   - (2->4) và (3->4) Các đường hội tụ về return",
            "3. Xác định P=2: Có 2 hàm độc lập.",
            "➔ Kết quả: 4(E) - 5(N) + 2*2(P) = 3.0",
            "➔ Trung bình: 1.5"
        ],
        "out": "1.5"
    },
    "num_functions": {
        "desc": "Tổng số lượng hàm được định nghĩa trong file.",
        "formula": r"\text{Value} = \sum \text{Definitions}",
        "in": "void f(){} void g(){}",
        "calc": ["Hàm f: 1", "Hàm g: 1", "Tổng: 2.0"],
        "out": "2.0"
    },
    "avg_function_loc": {
        "desc": "Số dòng code (LOC) trung bình của các hàm.",
        "formula": r"\text{Avg} = \frac{\sum \text{LOC}(\text{func}_i)}{\text{Num Functions}}",
        "in": "void f() {\n  int x;\n}\n\nvoid g() {\n  x = 1;\n  return;\n}",
        "calc": [
            "1. Hàm f: 3 dòng (dòng 1-3)",
            "2. Hàm g: 4 dòng (dòng 5-8)",
            "3. Tổng số dòng: 3 + 4 = 7",
            "4. Tổng số hàm: 2",
            "➔ Phép tính: 7 / 2 = 3.5"
        ],
        "out": "3.5"
    },
    "halstead_volume": {
        "desc": "Khối lượng mã nguồn (đo lường kích thước thông tin của chương trình).",
        "formula": r"V = (N_1 + N_2) \log_2 (n_1 + n_2)",
        "in": "a = b + c + d;",
        "calc": [
            "1. Định nghĩa: n (duy nhất), N (tổng số), 1 (toán tử), 2 (toán hạng)",
            "2. Phân tích Toán tử (Operators): '=', '+', '+', ';'",
            "   ➔ n1 (duy nhất) = 3 | N1 (tổng) = 4",
            "3. Phân tích Toán hạng (Operands): 'a', 'b', 'c', 'd'",
            "   ➔ n2 (duy nhất) = 4 | N2 (tổng) = 4",
            "4. Tính toán:",
            "   - N = 4 + 4 = 8 | n = 3 + 4 = 7",
            "➔ Phép tính: 8 * log2(7) = 22.46",
            "💡 Giải thích: Volume đo độ 'nặng' của thông tin trong file."
        ],
        "out": "22.46"
    },
    "halstead_difficulty": {
        "desc": "Độ khó lập trình (Độ phức tạp để duy trì mã nguồn).",
        "formula": r"D = \frac{n_1}{2} \times \frac{N_2}{n_2}",
        "in": "a = b + c + d;",
        "calc": [
            "Sử dụng kết quả từ Volume:",
            "- n1 (Toán tử duy nhất): 3",
            "- n2 (Toán hạng duy nhất): 4",
            "- N2 (Tổng số toán hạng): 4",
            "➔ Phép tính: (3 / 2) * (4 / 4) = 1.5 * 1.0 = 1.5",
            "💡 Tại sao không dùng N1? Halstead cho rằng việc lặp lại toán tử (N1) không gây khó hiểu bằng việc lặp lại các biến số (N2/n2)."
        ],
        "out": "1.5"
    },
    "halstead_effort": {
        "desc": "Nỗ lực (Công sức) cần thiết để viết mã nguồn.",
        "formula": r"E = V \times D",
        "in": "a = b + c + d;",
        "calc": [
            "- V (Volume): 22.46",
            "- D (Difficulty): 1.5",
            "➔ Phép tính: 22.46 * 1.5 = 33.69",
            "💡 Giải thích: Effort đo thời gian và trí lực cần bỏ ra để viết code này."
        ],
        "out": "33.69"
    },
    "halstead_bugs": {
        "desc": "Ước lượng số lượng lỗi (Bugs) tiềm ẩn trong code.",
        "formula": r"B = V / 3000",
        "in": "a = b + c + d;",
        "calc": [
            "- V (Volume): 22.46",
            "➔ Phép tính: 22.46 / 3000 = 0.0075",
            "💡 Giải thích: Một hàm phức tạp (Volume lớn) sẽ có xác suất lỗi cao hơn."
        ],
        "out": "0.0075"
    },
    "maintainability_index": {
        "desc": "Chỉ số đánh giá độ 'dễ nuôi' (bảo trì) của mã nguồn.",
        "formula": r"MI = \max(0, 171 - 5.2\ln(V) - 0.23CC - 16.2\ln(LOC))",
        "in": "void f() {\n  if (a) x = 1;\n  return;\n}",
        "calc": [
            "1. Các thông số đầu vào (lấy từ các đặc trưng trên):",
            "   - LOC (Số dòng code): 4 dòng",
            "   - CC (Độ phức tạp): 2.0 (Gốc + 1 lệnh if)",
            "   - V (Volume): ~30.0 (Tính dựa trên số token)",
            "2. Giải mã công thức (Trừ dần từ 171):",
            "   - Hình phạt Volume (-5.2 * ln(30))",
            "   - Hình phạt Complexity (-0.23 * 2.0)",
            "   - Hình phạt Độ dài (-16.2 * ln(4))",
            "➔ MI = 171 - 17.6 - 0.46 - 22.4 = 130.54",
            "➔ Kết quả: Giới hạn tối đa là 100.0"
        ],
        "out": "100.0"
    },
    "code_to_comment_ratio": {
        "desc": "Tỷ lệ dòng code / dòng comment.",
        "in": "5 lines code, 1 line comment",
        "calc": ["Code: 5 dòng", "Comment: 1 dòng", "Phép tính: 5 / 1 = 5.0"],
        "out": "5.0"
    },
    "max_nesting_depth": {
        "desc": "Độ sâu lồng nhau tối đa (dấu {).",
        "in": "if{ if{ } }",
        "calc": ["Lớp 1: if", "Lớp 2: if lồng", "Kết quả: 2.0"],
        "out": "2.0"
    },

    # Habits
    "total_includes": {
        "desc": "Số lượng thư viện được include.",
        "in": "#include <iostream>\n#include <vector>",
        "calc": ["Đếm số dòng #include: 2"],
        "out": "2.0"
    },
    "has_bits_stdc": {
"desc": "Có sử dụng bits/stdc++.h hay không.",
        "in": "#include <bits/stdc++.h>",
        "calc": ["Khớp mẫu 'bits/stdc++.h': Có"],
        "out": "1.0"
    },
    "macro_count": {
        "desc": "Số lượng macro #define.",
        "in": "#define MAX 100",
        "calc": ["Đếm số dòng #define: 1"],
        "out": "1.0"
    },
    "modern_cpp_ratio": {
        "desc": "Tỷ lệ từ khóa C++ hiện đại (auto, nullptr, constexpr...) trên tổng số từ.",
        "in": "auto x = 5;",
        "calc": [
            "1. Từ khóa hiện đại tìm thấy: 'auto' (1)",
            "2. Tổng số từ (words) trong code: 3 ('auto', 'x', '5')",
            "➔ Phép tính: 1 / 3 = 0.3333"
        ],
        "out": "0.3333"
    },
    "const_usage_ratio": {
        "desc": "Tỷ lệ sử dụng const/constexpr trên tổng số từ.",
        "in": "const int x = 0;",
        "calc": [
            "1. Từ khóa tìm thấy: 'const' (1)",
            "2. Tổng số từ (words) trong code: 4 ('const', 'int', 'x', '0')",
            "➔ Phép tính: 1 / 4 = 0.25"
        ],
        "out": "0.25"
    },
    "has_fast_io": {
        "desc": "Sử dụng các lệnh tối ưu I/O.",
        "in": "std::ios::sync_with_stdio(false);",
        "calc": ["Khớp mẫu 'sync_with_stdio': Có"],
        "out": "1.0"
    },
    "newline_style_ratio": {
        "desc": "Thói quen dùng \\n thay cho endl.",
        "in": "cout << '\\n';",
        "calc": ["Số lượng '\\n': 1", "Số lượng 'endl': 0", "Phép tính: 1 / (1+0) = 1.0"],
"out": "1.0"
    },

    # Info Theory
    "shannon_entropy": {
        "desc": "Đo lường độ hỗn loạn (độ ngẫu nhiên) của các ký tự trong code.",
        "formula": r"H = -\sum p_i \log_2 p_i",
        "in": "int a=0;",
        "calc": [
            "1. Tổng số ký tự: 8 ('i','n','t',' ','a','=','0',';')",
            "2. Mỗi ký tự xuất hiện 1 lần ➔ p = 1/8",
            "➔ H = -8 * (1/8 * log2(1/8)) = 3.0 bits",
            "💡 Giải thích: Nếu code lặp lại nhiều (vd: aaaaaa), Entropy sẽ thấp hơn."
        ],
        "out": "3.0"
    },
    "bigram_entropy": {
        "desc": "Entropy dựa trên các cặp ký tự liền kề (Bigrams).",
        "in": "a+a+a",
        "calc": [
            "1. Trích xuất các cặp: 'a+', '+a', 'a+', '+a'",
            "2. Tần suất: p(a+)=0.5, p(+a)=0.5",
            "➔ H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = 1.0 bit",
            "💡 Giải thích: Đo độ lặp lại của các cụm ký tự (thói quen gõ phím)."
        ],
        "out": "1.0"
    },
    "whitespace_entropy": {
        "desc": "Entropy của chuỗi khoảng trắng (Space, Tab, Newline).",
        "in": "int a;\nint b;",
        "calc": [
            "1. Chuỗi khoảng trắng trích xuất: 'S' (giữa int/a), 'N' (xuống dòng), 'S' (giữa int/b) ➔ 'SNS'",
            "2. Tần suất: p(S)=2/3, p(N)=1/3",
            "➔ H = -(2/3*log2(2/3) + 1/3*log2(1/3)) = 0.918 bit",
            "💡 Giải thích: Nếu bạn luôn dùng Space sau mỗi dòng, Entropy sẽ rất thấp."
        ],
        "out": "0.918"
    },
}

def render_analysis_page(extractor, model, scaler, final_features, baselines):
    st.title("🔍 C++ Source Code AI Detection & Fingerprint")
    st.markdown("Hệ thống phát hiện mã nguồn AI dựa trên **32 đặc trưng định danh** và mô hình **Machine Learning (LightGBM)**.")

    # Sidebar Settings
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Analysis Settings")
    clean_metadata = st.sidebar.checkbox("Clean AI Metadata Headers", value=True)
    
    # Layout chính
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Source Code")
        code_input = st.text_area("Dán mã nguồn C++ vào đây:", height=500, placeholder="int main() {\\n    // Write your code here...\\n}")
        analyze_btn = st.button("🚀 Analyze & Detect", use_container_width=True)

    if analyze_btn and code_input:
        final_code = strip_metadata_headers(code_input) if clean_metadata else code_input
        
        with st.spinner("Đang mổ xẻ mã nguồn và chạy Inference..."):
            # 1. Trích xuất đặc trưng
            features = extractor.extract(final_code)
            
            # 2. Xử lý dữ liệu cho Model
            original_columns = scaler.feature_names_in_
            df_feat = pd.DataFrame([features])
            df_feat = df_feat.reindex(columns=original_columns, fill_value=0)
            
            # Scale
            scaled_feat = scaler.transform(df_feat)
            df_scaled = pd.DataFrame(scaled_feat, columns=original_columns)
            
            # Predict
            X_infer = df_scaled[final_features]
            pred_proba = model.predict_proba(X_infer)[0, 1]
            
            # 3. Tính SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_infer)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Lấy values cho class 1 (AI)
            shap_values = shap_values[0] # Lấy sample đầu tiên

        with col2:
            st.subheader("🎯 Final Prediction Result")
            is_ai = pred_proba >= 0.5
            
            if is_ai:
                st.markdown(f"<div class='ai-box'><h2 style='color:#e74c3c; margin:0;'>🤖 AI GENERATED</h2><p style='margin:0; font-size:18px;'>Confidence: <b>{pred_proba*100:.2f}%</b></p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='human-box'><h2 style='color:#2ecc71; margin:0;'>🧑‍💻 HUMAN WRITTEN</h2><p style='margin:0; font-size:18px;'>Confidence: <b>{(1 - pred_proba)*100:.2f}%</b></p></div>", unsafe_allow_html=True)
            
            st.progress(float(pred_proba))
            
            st.markdown("### Lực lượng chi phối quyết định (Top Features)")
            
            # Phân tích top features
            shap_df = pd.DataFrame({
                'Feature': final_features,
                'SHAP': shap_values,
                'Abs_SHAP': np.abs(shap_values),
                'Value': [features.get(f, 0) for f in final_features]
            }).sort_values(by='Abs_SHAP', ascending=False)
            
            top_n = 5
            top_features = shap_df.head(top_n)
            
            # Phân loại đặc trưng cho NLG
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

            # Lời bình tự động (Executive Summary NLG)
            top_1 = top_features.iloc[0]
            top_2 = top_features.iloc[1]
            
            g1 = get_group_vn(feat_to_group.get(top_1['Feature']))
            g2 = get_group_vn(feat_to_group.get(top_2['Feature']))
            
            summary_text = f"Dựa trên phân tích hình thái (Fingerprint), đoạn mã này bộc lộ rất rõ bản chất của **{'thuật toán máy học' if is_ai else 'con người'}**."
            if is_ai:
                summary_text += f" Quyết định này được hệ thống đưa ra phần lớn là do sự hoàn hảo bất thường trong **{g1}** ({top_1['Feature'].replace('_', ' ').title()}), kết hợp với những đặc trưng máy móc ở **{g2}** ({top_2['Feature'].replace('_', ' ').title()})."
            else:
                summary_text += f" Sự ngẫu hứng đặc trưng của con người thể hiện rất rõ qua **{g1}** ({top_1['Feature'].replace('_', ' ').title()}), cũng như những dấu vết để lại trong **{g2}** ({top_2['Feature'].replace('_', ' ').title()})."
            
            st.info("💡 **Tổng quan Hành vi (Executive Summary):**\n\n" + summary_text)

            for _, row in top_features.iterrows():
                f_name = row['Feature']
                f_val = row['Value']
                s_val = row['SHAP']
                
                b_info = BEHAVIORAL_INSIGHTS.get(f_name, {})
                f_info = FEATURE_DETAILS.get(f_name, {})
                
                # Lấy insight hành vi tương ứng với nhãn
                if s_val > 0:
                    insight = b_info.get("ai", f"Thống kê cho thấy thói quen {f_info.get('desc', 'này').lower()} ở mức này nghiêng về phía AI.")
                else:
                    insight = b_info.get("human", f"Thống kê cho thấy thói quen {f_info.get('desc', 'này').lower()} ở mức này nghiêng về phía Human.")
                
                direction = "Kéo dự đoán về phía AI 🤖" if s_val > 0 else "Kéo dự đoán về phía Con Người 🧑‍💻"
                color = "#e74c3c" if s_val > 0 else "#2ecc71"
                
                # Format value
                val_str = f"{f_val:.4f}" if isinstance(f_val, float) else str(f_val)
                
                # Contextual Baselines
                baseline = baselines.get(f_name, {"ai": 0.0, "human": 0.0})
                ai_mean = baseline["ai"]
                human_mean = baseline["human"]
                baseline_str = f"Mốc chuẩn: Human ≈ {human_mean:.4f} | AI ≈ {ai_mean:.4f}"

                st.markdown(f"""
                <div style='padding: 15px; border-left: 5px solid {color}; background-color: #1e2130; margin-bottom: 15px; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <strong style='color:#00d4ff; font-size: 18px;'>{f_name.replace('_', ' ').title()}</strong>
                        <span style='background-color: #2b3040; padding: 4px 10px; border-radius: 20px; font-family: monospace; font-size: 14px; border: 1px solid #3d425c;'>Giá trị đo được: {val_str}</span>
                    </div>
                    <div style='margin-bottom: 8px; color: #aaaaaa; font-size: 13px;'>
                        <i>📊 {baseline_str}</i>
                    </div>
                    <div style='margin-bottom: 10px; color: {color}; font-weight: bold;'>
                        <i>👉 {direction}</i>
                    </div>
                    <div style='color: #e0e0e0; font-size: 15px; line-height: 1.6;'>
                        {insight}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        with st.expander("📊 Xem chi tiết toàn bộ các đặc trưng và Fingerprint Radar"):
            categories = {
                "Layout & Formatting": ["comment_ratio", "empty_line_ratio", "avg_line_length", "max_line_length", "tab_vs_space_ratio", "trailing_space_ratio", "brace_style_consistency"],
                "Naming Conventions": ["avg_identifier_length", "identifier_length_variance", "single_char_var_ratio", "unique_identifier_ratio", "keyword_to_identifier_ratio"],
                "Structural Complexity": ["avg_cyclomatic_complexity", "num_functions", "avg_function_loc", "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs", "maintainability_index", "code_to_comment_ratio", "max_nesting_depth"],
                "Coding Habits": ["total_includes", "has_bits_stdc", "macro_count", "modern_cpp_ratio", "const_usage_ratio", "has_fast_io", "newline_style_ratio"],
                "Information Theory": ["shannon_entropy", "bigram_entropy", "whitespace_entropy"]
            }

            radar_data = []
            for cat, f_list in categories.items():
                avg_val = sum([features.get(f, 0) for f in f_list]) / len(f_list)
                radar_data.append(dict(Category=cat, Value=min(avg_val * 10, 100)))
            
            df_radar = pd.DataFrame(radar_data)
            fig = px.line_polar(df_radar, r='Value', theta='Category', line_close=True, range_r=[0,100])
            fig.update_traces(fill='toself', line_color='#00d4ff')
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            cat_tabs = st.tabs(list(categories.keys()))
            for i, (cat, f_list) in enumerate(categories.items()):
                with cat_tabs[i]:
                    cols = st.columns(3)
                    for j, f_name in enumerate(f_list):
                        val = features.get(f_name, 0)
                        with cols[j % 3]:
                            st.metric(label=f_name.replace("_", " ").title(), value=f"{val:.4f}" if isinstance(val, float) else val)

    else:
        st.info("👈 Hãy dán mã nguồn C++ vào ô bên trái và nhấn 'Analyze & Detect' để bắt đầu.")


def render_glossary_page():
    st.title("📖 Từ điển 32 Đặc trưng (Chi tiết)")
    st.markdown("Khám phá chi tiết từng đặc trưng và cách phân biệt giữa **Người** và **AI**.")

    feature_details = FEATURE_DETAILS

    # Sidebar để chọn nhanh đặc trưng
    st.sidebar.divider()
    search_query = st.sidebar.text_input("🔍 Tìm nhanh đặc trưng:", placeholder="Ví dụ: entropy, halstead...")

    def render_feature_list(features_list, category_name):
        st.header(category_name)
        for f in features_list:
            if search_query.lower() in f.lower():
                with st.expander(f"🔹 {f.replace('_', ' ').title()}"):
                    info = feature_details.get(f, {})
                    st.write(f"**Mô tả:** {info.get('desc', 'N/A')}")
                    if "formula" in info:
                        st.latex(info["formula"])
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info("Ví dụ Input (Code):")
                        st.code(info.get('in', '// sample code'), language="cpp")
                    with c2:
                        if "calc" in info:
                            st.warning("**📊 Phép tính:**")
                            if isinstance(info["calc"], list):
                                for step in info["calc"]: st.write(f"- {step}")
                            else: st.write(info["calc"])
                        st.success(f"**✅ Kết quả:** Value = {info.get('out', '0.0')}")

    render_feature_list(["comment_ratio", "empty_line_ratio", "avg_line_length", "max_line_length", "tab_vs_space_ratio", "trailing_space_ratio", "brace_style_consistency"], "🎨 Nhóm A: Layout & Formatting")
    render_feature_list(["avg_identifier_length", "identifier_length_variance", "single_char_var_ratio", "unique_identifier_ratio", "keyword_to_identifier_ratio"], "🏷️ Nhóm B: Naming Conventions")
    render_feature_list(["avg_cyclomatic_complexity", "num_functions", "avg_function_loc", "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs", "maintainability_index", "code_to_comment_ratio", "max_nesting_depth"], "🏗️ Nhóm C: Structural Complexity")
    render_feature_list(["total_includes", "has_bits_stdc", "macro_count", "modern_cpp_ratio", "const_usage_ratio", "has_fast_io", "newline_style_ratio"], "🧠 Nhóm D: Coding Habits & Idioms")
    render_feature_list(["shannon_entropy", "bigram_entropy", "whitespace_entropy"], "🎲 Nhóm E: Information Theory")

    st.divider()
    st.subheader("🚀 Quy trình biến đổi: Code ➔ Vector")
    p1, p2, p3, p4, p5 = st.columns([1, 0.2, 1, 0.2, 1])
    with p1: st.markdown("<div style='text-align:center; padding:10px; border-radius:10px; background-color:#1e2130; border:1px solid #00d4ff;'><b>📝 Source Code</b><br><small>Mã nguồn C++ thô</small></div>", unsafe_allow_html=True)
    with p2: st.markdown("<div style='text-align:center; padding-top:15px;'>➡️</div>", unsafe_allow_html=True)
    with p3: st.markdown("<div style='text-align:center; padding:10px; border-radius:10px; background-color:#1e2130; border:1px solid #00d4ff;'><b>🔍 Phân tích Tĩnh</b><br><small>Regex & Lizard</small></div>", unsafe_allow_html=True)
    with p4: st.markdown("<div style='text-align:center; padding-top:15px;'>➡️</div>", unsafe_allow_html=True)
    with p5: st.markdown("<div style='text-align:center; padding:10px; border-radius:10px; background-color:#00d4ff; color:black; border:1px solid #00d4ff;'><b>🔢 Vector (32)</b><br><small>Dữ liệu số hóa</small></div>", unsafe_allow_html=True)

    st.subheader("🔍 Ví dụ Trace chi tiết")
    trace_code = """
#include <iostream>
int main() {
    // Tinh tong 1..10
    int s = 0;
    for(int i=0; i<10; i++) s += i;
    return 0;
}"""
    t_col1, t_col2 = st.columns([1, 1.5])
    with t_col1: st.code(trace_code, language="cpp")
    with t_col2:
        with st.expander("Bước 1: Quét Regex (Habits)", expanded=True):
            st.write("- Tìm thấy `#include` ➔ `total_includes = 1`")
            st.write("- Không thấy `bits/stdc++.h` ➔ `has_bits_stdc = 0`")
        with st.expander("Bước 2: Cấu trúc (Complexity)"):
            st.write("- 1 hàm main, CC=2 (1 for), 5 dòng code")
def render_feature_selection_page():
    st.title("⚙️ Quy trình Tuyển chọn Đặc trưng (Triple Filter)")
    st.markdown("Hệ thống lọc 3 tầng giúp loại bỏ nhiễu và chống **Overfitting** (Học vẹt).")

    tabs = st.tabs(["🛡️ Tầng 1: Correlation", "📊 Tầng 2: Mutual Info", "🎯 Tầng 3: Lasso L1", "🚀 Kết quả cuối cùng"])

    with tabs[0]:
        st.subheader("🛡️ Tầng 1: Lọc Tương quan (Pearson Correlation)")
        st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.info("**Logic trinh sát:**")
            st.write("1. Tính ma trận tương quan giữa tất cả các cặp đặc trưng.")
            st.write("2. Tìm các cặp có **|r| > 0.85**.")
            st.write("3. Loại bỏ 1 trong 2 đặc trưng để tránh **Đa cộng tuyến**.")
        with c2:
            st.warning("**Truy vết thực tế:**")
            st.write("Loại bỏ các đặc trưng có tương quan cực cao (>0.95) để tránh gây nhiễu cho mô hình:")
            st.code("- halstead_effort (Tương quan với Volume)\n- halstead_bugs (Tương quan với Volume)\n- num_functions (Tương quan với LOC/Complexity)\n- bigram_entropy (Tương quan với Shannon Entropy)", language="python")
        st.success("➔ **Kết quả:** Giảm từ 32 đặc trưng xuống còn 28 đặc trưng mạnh nhất.")

    with tabs[1]:
        st.subheader("📊 Tầng 2: Mutual Information (K-Best)")
        st.latex(r"I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right)")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.info("**Logic trinh sát:**")
            st.write("1. Đo lường mức độ 'giảm bớt sự không chắc chắn' của Nhãn (Human/AI) khi biết Đặc trưng.")
            st.write("2. Đặc trưng nào có **Information Gain** cao nhất sẽ được giữ lại.")
            st.write("3. Chọn **Top 20** đặc trưng mạnh nhất.")
        with c2:
            st.warning("**Truy vết thực tế:**")
            st.write("Loại bỏ 8 đặc trưng có điểm Mutual Information thấp nhất (ít đóng góp vào việc phân loại):")
            st.write("- `has_bits_stdc`, `has_fast_io`, `newline_style_ratio` (Thói quen quá biến thiên)")
            st.write("- `modern_cpp_ratio`, `const_usage_ratio` (Ít xuất hiện trong dataset)")
            st.write("- `shannon_entropy`, `comment_ratio`, `trailing_space_ratio` (Bị nhiễu)")
        st.success("➔ **Kết quả:** Giữ lại đúng 20 ứng viên có lượng thông tin 'tinh khiết' nhất.")

    with tabs[2]:
        st.subheader("🎯 Tầng 3: Lasso L1 Regularization")
        st.latex(r"\min_{\beta} \left( \sum (y_i - X_i \beta)^2 + \lambda \sum |\beta_j| \right)")
        
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.info("**Logic trinh sát:**")
            st.write("1. Đưa 20 đặc trưng vào mô hình Logistic Regression.")
            st.write("2. Áp dụng hình phạt **L1 (Lasso)** vào các trọng số.")
            st.write("3. Những đặc trưng không quan trọng sẽ bị ép trọng số về **0.0**.")
        with c2:
            st.warning("**Truy vết thực tế:**")
            st.write("Kiểm tra trọng số 20 đặc trưng cuối cùng:")
            st.code("- whitespace_entropy: 1.42 (Quan trọng nhất)\n- avg_identifier_length: 0.98\n- single_char_var_ratio: 0.85\n- ... 17 features khác có trọng số > 0", language="python")
        st.success("➔ **Kết quả:** Chốt hạ bộ khung 20 đặc trưng tối ưu (Lưu vào final_features.pkl).")

    with tabs[3]:
        st.subheader("🚀 Minh họa Phễu lọc dữ liệu (Funnel)")
        funnel_data = dict(
            number=[32, 28, 20, 20],
            stage=["1. Trích xuất thô (32)", "2. Lọc Đa cộng tuyến (28)", "3. Mutual Info (20)", "4. Lasso L1 (20)"]
        )
        fig = px.funnel(funnel_data, x='number', y='stage', color_discrete_sequence=['#00d4ff'])
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        final_list = [
            'empty_line_ratio', 'avg_line_length', 'max_line_length', 'tab_vs_space_ratio', 
            'brace_style_consistency', 'avg_identifier_length', 'identifier_length_variance', 
            'single_char_var_ratio', 'unique_identifier_ratio', 'keyword_to_identifier_ratio', 
            'avg_cyclomatic_complexity', 'avg_function_loc', 'halstead_volume', 
            'halstead_difficulty', 'maintainability_index', 'code_to_comment_ratio', 
            'max_nesting_depth', 'total_includes', 'macro_count', 'whitespace_entropy'
        ]
        st.info("**Danh sách 20 đặc trưng 'tinh túy' cuối cùng:**\\n" + ", ".join([f"`{f}`" for f in final_list]))

def main():
    # Sidebar Navigation
    st.sidebar.title("🎮 Navigator")
    page = st.sidebar.radio("Chọn chức năng:", ["🚀 Phân tích Fingerprint", "📖 Từ điển 32 Đặc trưng", "⚙️ Quy trình Tuyển chọn"], index=0)
    extractor = get_extractor()
    model, scaler, final_features, baselines = load_models()
    
    if page == "🚀 Phân tích Fingerprint": render_analysis_page(extractor, model, scaler, final_features, baselines)
    elif page == "📖 Từ điển 32 Đặc trưng": render_glossary_page()
    elif page == "⚙️ Quy trình Tuyển chọn": render_feature_selection_page()
    st.sidebar.divider()
    st.sidebar.caption("System: Anti-Gravity V8 (Anti-Overfit)")
    st.sidebar.caption("© 2026 LVTN: AI Code Detection")

if __name__ == "__main__":
    main()
