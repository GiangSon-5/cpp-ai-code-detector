# Django Web Application — Đặc tả Nghiệp vụ (SRS)

### UC: Quản lý người dùng và nộp code - Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Django Web cung cấp giao diện người dùng để đăng ký/đăng nhập, nộp code C++ để phân tích, xem lịch sử phân tích, và dashboard thống kê tổng hợp.

| Primary Actor: | End User (Student/Dev) | Secondary Actor: | FastAPI AI Service |
|----------------|------------------------|------------------|--------------------|
| **Description:** | Giao diện web chính cho hệ thống AI Code Detector |
| **Trigger:** | User truy cập web, nộp code, xem kết quả |
| **Preconditions:** | PRE1: PostgreSQL đang chạy. PRE2: FastAPI service available. |
| **Post-conditions:** | POST1: Code lưu Bronze layer. POST2: Kết quả hiển thị cho user. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** File .cpp hoặc paste code C++ 30 dòng
- **Hệ thống xử lý:** Lưu Bronze → Forward FastAPI → Stream kết quả
- **Kết quả nhận được:** Trang kết quả với AI Score, Prediction, Heatmap, Chunk details

**Normal Flow:**

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | User đăng nhập | Xác thực session, redirect dashboard |
| 2 | User paste/upload code C++ | Validate input (không rỗng, < 500KB) |
| 3 | User click "Phân tích" | Save Bronze → Celery push S3 → Forward FastAPI |
| 4 | — | SSE stream progress events → render real-time |
| 5 | User xem kết quả | Hiển thị score, prediction, heatmap, chunks |
| 6 | User xem lịch sử | Query Bronze + Gold → list submissions |

**Exception:**

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Chưa đăng nhập | Redirect login page |
| 2 | Code rỗng | Flash message: "Vui lòng nhập code" |
| 3 | File > 500KB | Flash message: "File quá lớn" |
| 4 | FastAPI timeout | Error page + retry button |

**Business Rules:**

| No | Rule |
|----|------|
| 1 | Mỗi user có lịch sử phân tích riêng (filtered by user FK) |
| 2 | Duplicate code (same hash) → trả kết quả cũ, không chạy lại model |
| 3 | Dashboard hiển thị thống kê Gold: tổng submissions, AI ratio, avg confidence |
| 4 | Admin có thể xem tất cả submissions qua Django Admin |

**Bảng mô tả giao diện (UI) / Data Mapping:**

| Tên trường | Mô tả | Kiểu dữ liệu / Control | Mặc định | Bắt buộc | Ví dụ | Direct to |
|------------|-------|------------------------|----------|----------|-------|-----------|
| code_input | Textarea nhập code | textarea (rows=15) | "" | Y | `#include <iostream>...` | /submit |
| file_upload | Upload file .cpp | file input (.cpp,.c,.h,.txt) | — | N | code.cpp | /submit |
| result_score | Điểm AI | readonly display | — | — | 0.8742 | /result/{id} |
| result_pred | Nhãn phân loại | badge (red/green) | — | — | "AI GENERATED" | /result/{id} |
| result_ppl | Perplexity | readonly display | — | — | 2.34 | /result/{id} |
| heatmap | Visualization | iframe/embedded HTML | — | — | (HTML heatmap) | /result/{id} |
