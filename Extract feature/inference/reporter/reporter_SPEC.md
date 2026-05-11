# Module: Inference & Reporting (SPEC)

## 1. Module Overview
Module này thực hiện nhiệm vụ dự đoán trên thực tế. Nó có khả năng quét qua các thư mục lồng nhau, đọc hàng nghìn file mã nguồn, sử dụng model đã huấn luyện để phân loại và tổng hợp thành báo cáo tỷ lệ AI/Human theo từng thư mục.

## 2. Data Contracts & Examples
**Input:**
- Thư mục gốc (`PARENT_DIR`) chứa các file `.c`, `.cpp`, `.h`.
- Model artifact và Scaler object từ bước trước.

**Output:**
- Bảng tổng kết (DataFrame) tỷ lệ AI/Human.
- Biểu đồ phân phối xác suất (KDE Plot).
- Danh sách các file dự đoán sai để phân tích (Error Analysis).

Example Summary Table:
| Thư mục | Tổng số file | Tỷ lệ AI | Tỷ lệ Human |
|---------|--------------|-----------|-------------|
| FromAI-v1 | 100 | 98.00% | 2.00% |
| Student_HW| 50 | 10.00% | 90.00% |

## 3. Core Logic & Formulas
- **Directory Scanning:** Sử dụng `os.walk` để duyệt đệ quy tất cả các cấp thư mục.
- **Classification Threshold:** Mặc định sử dụng $0.5$. Tuy nhiên, có thể điều chỉnh (`CUSTOM_THRESHOLD`) để tăng tính khắt khe (ví dụ: $0.7$ nếu muốn chắc chắn 100% là AI).
- **Probability Distribution:** Tính toán xác suất của lớp 1 (AI) thông qua `predict_proba`.

## 4. End-to-End Trace Example
- **Execution Trace:**
    1. Duyệt thư mục `/test-set/gpt4o`.
    2. Tìm thấy 50 file `.cpp`.
    3. Với mỗi file: Clean metadata -> Extract -> Scale -> Predict.
    4. Ghi nhận: 48 file có xác suất > 0.5.
    5. Tính tỷ lệ AI cho thư mục `gpt4o` = 96%.
- **Sample Output:** Bảng dashboard tổng kết cho 10+ thư mục model AI khác nhau.

## 5. LLM Prompts/Integrations
N/A.

## 6. Edge Cases
- **Encoding Errors:** File code (đặc biệt là từ Human) có thể chứa ký tự lạ (UTF-8 lỗi). Cần dùng `errors='ignore'`.
- **Thư mục trống:** Cần skip để tránh lỗi chia cho 0 khi tính tỷ lệ.
- **File format lạ:** Một số file không phải code C++ nhưng có đuôi mở rộng tương tự.
