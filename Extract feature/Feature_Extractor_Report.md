# BÁO CÁO KỸ THUẬT: HỆ THỐNG TRÍCH XUẤT ĐẶC TRƯNG MÃ NGUỒN C++ 

## 1. Tổng quan (Overview)
Hệ thống được thiết kế để chuyển đổi mã nguồn C++ thô thành một tập hợp các đặc trưng định lượng (Numerical Features). Hệ thống tập trung vào việc tối ưu hóa khả năng chống học tủ (Anti-overfitting) và tích hợp các chỉ số tiêu chuẩn trong lĩnh vực đo lường phần mềm.

## 2. Phương pháp luận (Methodology)
Hệ thống sử dụng kết hợp bốn phương pháp tiếp cận chính để trích xuất dữ liệu:
1.  **Phân tích dựa trên Biểu thức chính quy (Regex):** Nhận diện các mẫu hình cú pháp, từ khóa và thói quen đặt tên.
2.  **Phân tích tĩnh nâng cao (Static Analysis):** Sử dụng thư viện `lizard` để tính toán độ phức tạp vòng (Cyclomatic Complexity).
3.  **Số học phần mềm Halstead (Halstead Software Metrics):** Đo lường khối lượng công việc, độ khó và lỗi dựa trên toán tử (Operators) và toán hạng (Operands).
4.  **Lý thuyết thông tin (Information Theory):** Sử dụng Entropy (Shannon & Bigram) để đánh giá mức độ hỗn loạn và tính lặp lại của mã nguồn.

---

## 3. Danh mục Đặc trưng Chi tiết (Feature Taxonomy)

Hệ thống trích xuất tổng cộng **32 đặc trưng**, được chia thành 5 nhóm chính:

### Nhóm A: Bố cục và Định dạng (Layout & Formatting)
*Mục đích: Xác định phong cách trình bày trực quan của lập trình viên.*

| STT | Đặc trưng | Mô tả |
|:---:|---|---|
| 1 | `comment_ratio` | Tỷ lệ ký tự chú thích trên tổng số ký tự mã nguồn. |
| 2 | `empty_line_ratio` | Tỷ lệ dòng trống (thể hiện thói quen phân đoạn code). |
| 3 | `avg_line_length` | Độ dài trung bình của một dòng mã. |
| 4 | `max_line_length` | Độ dài lớn nhất của một dòng mã. |
| 5 | `tab_vs_space_ratio` | Tỷ lệ giữa phím Tab và dấu cách (Space) trong thụt đầu dòng. |
| 6 | `trailing_space_ratio` | Tỷ lệ các dòng có khoảng trắng thừa ở cuối dòng. |
| 7 | `brace_style_consistency` | Độ nhất quán trong việc đặt dấu `{` (theo kiểu K&R hay Allman). |

### Nhóm B: Quy ước đặt tên (Naming Conventions)
*Mục đích: Phân tích thói quen tư duy và cách đặt tên biến/hàm.*

| STT | Đặc trưng | Mô tả |
|:---:|---|---|
| 8 | `avg_identifier_length` | Độ dài trung bình của các định danh (tên biến, hàm). |
| 9 | `identifier_length_variance` | Phương sai độ dài định danh (đo lường sự đa dạng trong cách đặt tên). |
| 10 | `single_char_var_ratio` | Tỷ lệ sử dụng biến 1 ký tự (thường gặp trong lập trình thi đấu). |
| 11 | `unique_identifier_ratio` | Tỷ lệ định danh độc nhất (đo lường mức độ tái sử dụng biến). |
| 12 | `keyword_to_identifier_ratio` | Tương quan giữa từ khóa ngôn ngữ và tên do người dùng đặt. |

### Nhóm C: Độ phức tạp cấu trúc (Structural Complexity)
*Mục đích: Đo lường mức độ logic và khả năng bảo trì của chương trình.*

| STT | Đặc trưng | Mô tả |
|:---:|---|---|
| 13 | `avg_cyclomatic_complexity` | Độ phức tạp vòng trung bình (số lượng đường đi logic qua code). |
| 14 | `num_functions` | Tổng số lượng hàm được định nghĩa. |
| 15 | `avg_function_loc` | Số dòng mã trung bình trên mỗi hàm. |
| 16 | `halstead_volume` | Khối lượng thông tin của thuật toán. |
| 17 | `halstead_difficulty` | Độ khó của mã nguồn khi đọc hoặc viết lại. |
| 18 | `halstead_effort` | Nỗ lực cần thiết để lập trình thuật toán này. |
| 19 | `halstead_bugs` | Ước lượng số lượng lỗi tiềm ẩn dựa trên cấu trúc. |
| 20 | `maintainability_index` | Chỉ số khả năng bảo trì (tổng hợp từ Halstead và CC). |
| 21 | `code_to_comment_ratio` | Tỷ lệ giữa dòng code thực thi và dòng chú thích. |
| 22 | `max_nesting_depth` | Độ sâu lồng nhau tối đa của các khối lệnh `{...}`. |



### Nhóm D: Thói quen lập trình (Coding Habits & Idioms)
*Mục đích: Nhận diện các "vân tay" đặc thù của lập trình viên hoặc môi trường (AI vs Human).*

| STT | Đặc trưng | Mô tả |
|:---:|---|---|
| 23 | `total_includes` | Số lượng thư viện được sử dụng (`#include`). |
| 24 | `has_bits_stdc` | Sử dụng thư viện tổng hợp `bits/stdc++.h` (đặc trưng của CP). |
| 25 | `macro_count` | Số lượng các macro `#define`. |
| 26 | `modern_cpp_ratio` | Tỷ lệ dùng từ khóa C++ hiện đại (`auto`, `nullptr`, `lambda`). |
| 27 | `const_usage_ratio` | Mức độ sử dụng tính đóng gói qua `const` và `constexpr`. |
| 28 | `has_fast_io` | Có sử dụng các kỹ thuật tối ưu hóa nhập xuất (Fast I/O) hay không. |
| 29 | `newline_style_ratio` | Thói quen sử dụng `\n` thay vì `endl`. |

### Nhóm E: Lý thuyết thông tin (Information Theory)
*Mục đích: Phân tích dấu vết thống kê của các ký tự.*

| STT | Đặc trưng | Mô tả |
|:---:|---|---|
| 30 | `shannon_entropy` | Độ hỗn loạn của các ký tự trong mã nguồn. |
| 31 | `bigram_entropy` | Độ hỗn loạn dựa trên cặp ký tự (phát hiện cấu trúc lặp). |
| 32 | `whitespace_entropy` | Entropy của các khoảng trắng (phát hiện cấu trúc thụt lề đặc thù). |

---

## 4. Cơ chế chống rò rỉ dữ liệu (Anti-Leakage Mechanism)
Hệ thống tích hợp hàm `strip_metadata_headers` để loại bỏ các thông tin rác hoặc thông tin "gợi ý" thường xuất hiện trong dữ liệu tổng hợp (đặc biệt là dữ liệu từ AI như GPT/Gemini). Các dòng comment chứa từ khóa như `DATASET`, `MODEL`, `CATEGORY` sẽ bị loại bỏ trước khi trích xuất để đảm bảo mô hình Machine Learning học từ phong cách lập trình thực sự, không phải từ các nhãn metadata.

