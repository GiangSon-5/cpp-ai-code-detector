# FastAPI AI Service — Đặc tả Nghiệp vụ (SRS)

### UC: Phân tích mã nguồn C++ - Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module FastAPI AI Service nhận đoạn mã C++ từ người dùng, tự động phân loại loại code (OOP/Normal), chạy ensemble model AI (RoBERTa + Perplexity), tự sửa sai nếu kết quả nhập nhằng, và trả về báo cáo phân tích chi tiết kèm giải thích XAI.

| Primary Actor: | End User (Student/Developer) | Secondary Actor: | LLM Service (Gemini/OpenAI/Qwen) |
|----------------|------------------------------|------------------|-----------------------------------|
| **Description:** | Phân tích code C++ để xác định do AI hay Human viết |
| **Trigger:** | POST /api/analyze_stream với payload Base64 |
| **Preconditions:** | PRE1: Server đã khởi động và load model thành công. PRE2: Code input không rỗng. |
| **Post-conditions:** | POST1: Kết quả lưu vào Gold layer (PostgreSQL). POST2: Raw code push lên Bronze (DagsHub S3). |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** Đoạn code C++ 50 dòng dạng OOP (có class, virtual, inheritance)
- **Hệ thống xử lý:** Router phân loại → OOP Model → RoBERTa Ensemble 5-fold → LIG Attribution → Perplexity → Judge → LLM Critique
- **Kết quả nhận được:** JSON với `final_pred: "AI GENERATED"`, `final_score: 0.92`, kèm heatmap HTML và giải thích per-chunk

**Normal Flow:**

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | User gửi POST /api/analyze_stream với code Base64 | Validate input, decode Base64, tạo code_hash |
| 2 | — | Router Node: LLM phân loại OOP/NORMAL (fallback heuristic) |
| 3 | — | Analyzer Node: Chạy RoBERTa Ensemble + LIG + Perplexity |
| 4 | — | Judge Node: Đánh giá confidence, trigger self-correction nếu cần |
| 5 | — | Critique Node: LLM phân tích Map-Reduce per chunk |
| 6 | — | SSE stream kết quả real-time (progress 5→20→65→75→95→100%) |
| 7 | User nhận SSE events | Hiển thị kết quả + heatmap + critique |

**Exception:**

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Code rỗng | HTTP 400: `{"error": "Empty code"}` |
| 2 | Base64 không hợp lệ | HTTP 422: Validation Error |
| 3 | Model chưa load | HTTP 503: `{"error": "Model not loaded"}` |
| 4 | GPU OOM | Fallback inference-only, log warning |
| 5 | LLM timeout | Fallback text: "Analysis unavailable" |

**Business Rules:**

| No | Rule |
|----|------|
| 1 | Threshold mặc định = 0.5. Score ≥ 0.5 → "AI GENERATED" |
| 2 | Vùng nhập nhằng: 0.40 ≤ score ≤ 0.60 → trigger self-correction |
| 3 | PPL conflict: score > 0.6 AND PPL > 5.0 → suspicious, self-correct |
| 4 | Maximum 1 lần retry (self-correction). Sau đó accept kết quả |
| 5 | Cache theo code_hash (MD5). Max cache = 100 entries |
| 6 | Code dài > 510 tokens → chia chunk stride=256, weighted average |

**Bảng mô tả Data Mapping:**

| Tên trường | Mô tả | Kiểu dữ liệu | Mặc định | Bắt buộc | Ví dụ | Direct to |
|------------|-------|---------------|----------|----------|-------|-----------|
| code_base64 | Code C++ mã hóa Base64 | string | — | Y | "I2luY2x1ZGU..." | Input decode |
| final_pred | Nhãn phân loại cuối | string | — | Y | "AI GENERATED" | Dashboard |
| final_score | Điểm AI probability | float | 0.5 | Y | 0.8742 | Dashboard |
| model_used | Model đã chạy | string | — | Y | "C++ OOP Model" | Gold DB |
| perplexity | Điểm PPL | float | 0.0 | Y | 2.34 | Gold DB |
| is_ambiguous | Cờ nhập nhằng | boolean | false | Y | false | Gold DB |
| chunks | Chi tiết per-chunk | JSON array | [] | Y | [{index:1,...}] | Dashboard |
| global_critique | Tổng kết LLM | text | "" | N | "Code is AI..." | Dashboard |
| global_html | Heatmap HTML | text | "" | N | "<html>..." | Frontend |
