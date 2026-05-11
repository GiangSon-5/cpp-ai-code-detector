### UC: Trích xuất đặc trưng mã nguồn - Module Core Feature Extractor

**Mô tả chức năng tổng quan**
Hệ thống tự động phân tích cấu trúc và thói quen lập trình trong file C++ để chuyển đổi thành dữ liệu số mà máy tính có thể hiểu được.

| Primary Actor: | AI Agent / Developer | Secondary Actor: | Lizard Static Analyzer |
|----------------|---------------|------------------|---------------------|
| **Description:** | Phân tích file code để lấy các chỉ số về độ phức tạp, phong cách và thói quen. |
| **Trigger:** | Yêu cầu trích xuất từ Pipeline huấn luyện hoặc Inference. |
| **Preconditions:** | PRE1: Mã nguồn phải là chuỗi văn bản (String). |
| **Post-conditions:** | POST1: Trả về một vector đặc trưng đầy đủ 32 thuộc tính. |

**Business Scenario Walkthrough:**
- **Khách hàng/Hệ thống đưa vào:** Một đoạn mã C++ giải bài toán Competitive Programming.
- **Module xử lý:** Đếm số lượng từ khóa, đo độ dài biến, tính toán entropy của các khoảng trắng, gọi thư viện Lizard đo độ phức tạp vòng lặp.
- **Kết quả nhận được:** Một bảng dữ liệu số thể hiện "dấu vân tay" (fingerprint) của tác giả đoạn code đó.

**Normal Flow:**
| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Truyền chuỗi code vào hàm `extract()` | Hệ thống khởi tạo các bộ lọc Regex |
| 2 | Hệ thống chạy phân tích cấu trúc | Trả về các chỉ số Layout (comment, khoảng trắng...) |
| 3 | Hệ thống chạy phân tích Halstead | Trả về chỉ số về khối lượng và độ khó của thuật toán |
| 4 | Hệ thống gọi Lizard | Trả về số lượng hàm và độ phức tạp Cyclomatic |
| 5 | Tổng hợp dữ liệu | Trả về Dictionary kết quả |

**Exception:**
| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Code rỗng hoặc quá ngắn | Trả về vector với các giá trị mặc định (0 hoặc 1) |
| 2 | Lỗi parse từ Lizard | Ghi log lỗi và trả về CC mặc định là 1.0 |

**Business Rules:**
| No | Rule |
|----|------|
| 1 | Phải loại bỏ chuỗi (strings) và comment trước khi tính toán Halstead để tránh nhiễu. |
| 2 | Các giá trị vô cùng (Infinity) hoặc NaN phải được xử lý về 0. |

**Bảng mô tả giao diện (UI) / Data Mapping:**

**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|------------|-------|-----------------|------------------|----------|------------------|-----------|
| comment_ratio | Tỷ lệ chú thích | Float (0.0 - 1.0) | 0.0 | Y | 0.25 | Model Input |
| avg_cyclomatic_complexity | Độ phức tạp TB | Float | 1.0 | Y | 4.5 | Model Input |
| has_bits_stdc | Sử dụng thư viện tổng | Boolean (0/1) | 0 | Y | 1 | Model Input |
| modern_cpp_ratio | Tỷ lệ C++11+ | Float | 0.0 | Y | 0.05 | Model Input |

**Danh mục 32 Đặc trưng trích xuất:**
- **Nhóm Layout (7):** Tỷ lệ comment, dòng trống, chiều dài dòng, Tab/Space, brace style.
- **Nhóm Naming (5):** Độ dài định danh, biến 1 ký tự, tỷ lệ từ khóa.
- **Nhóm Complexity (10):** Cyclomatic Complexity, Halstead Metrics (Volume, Difficulty, Effort, Bugs), MI, Nesting Depth.
- **Nhóm Habits (7):** Include count, bits/stdc++, Macro, Modern C++, Const, Fast I/O, Newline style.
- **Nhóm Information Theory (3):** Shannon Entropy, Bigram Entropy, Whitespace Entropy.
