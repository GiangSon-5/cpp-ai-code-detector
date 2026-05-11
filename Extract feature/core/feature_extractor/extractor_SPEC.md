# Module: Core Feature Extractor (SPEC)

## 1. Module Overview
Module này chịu trách nhiệm phân tích mã nguồn C++ thô và chuyển đổi nó thành một vector số (numerical vector) gồm 32 đặc trưng. Đây là "trái tim" của hệ thống, cung cấp dữ liệu đầu vào cho các mô hình Machine Learning.

## 2. Data Contracts & Examples
**Input:** Chuỗi string chứa mã nguồn C++ (đã được làm sạch metadata).

**Output:** Dictionary chứa các cặp Key-Value của đặc trưng.
Example:
```json
{
  "comment_ratio": 0.15,
  "avg_cyclomatic_complexity": 3.4,
  "halstead_volume": 1250.5,
  "has_bits_stdc": 1,
  "shannon_entropy": 4.82
}
```

## 3. Core Logic & Formulas
- **Halstead Metrics:**
    - Vocabulary ($n$) = $n1 + n2$
    - Length ($N$) = $N1 + N2$
    - Volume ($V$) = $N \times \log_2(n)$
    - Difficulty ($D$) = $(n1 / 2) \times (N2 / n2)$
- **Maintainability Index (MI):**
    - $MI = 171 - 5.2 \times \ln(V) - 0.23 \times CC - 16.2 \times \ln(LOC)$
- **Shannon Entropy:**
    - $H(X) = -\sum p(x) \log_2 p(x)$

## 4. End-to-End Trace Example
- **Sample Input:**
```cpp
#include <iostream>
int main() {
    // Hello World
    std::cout << "Hello";
    return 0;
}
```
- **Execution Trace:**
    1. `re.findall` tìm thấy toán tử: `<<`, `=`, `-`, etc.
    2. `lizard` phân tích hàm `main`, CC = 1.
    3. `re.search` tìm thấy `#include`.
    4. Tính toán tỷ lệ comment (1 dòng comment / 6 dòng tổng).
- **Sample Output:** `{"num_functions": 1, "has_bits_stdc": 0, "comment_ratio": 0.166, ...}`

## 5. LLM Prompts/Integrations
Module này thuần túy là Static Analysis, không cần LLM Integration. Tuy nhiên, có thể tích hợp LLM để giải thích *tại sao* một đặc trưng lại cao bất thường (ví dụ: Entropy cực cao thường do code obfuscated).

## 6. Edge Cases
- **Code không hợp lệ (Syntax Error):** `lizard` có thể crash hoặc trả về kết quả rỗng. Cần try-except.
- **Code cực ngắn:** Halstead Volume có thể bằng 0 hoặc gây lỗi $\log(0)$.
- **Macro phức tạp:** Regex có thể không bắt được hết các macro lồng nhau.

## 7. Detailed Feature List (32 Features)

### [A] Layout & Formatting (Định dạng & Bố cục) - 7 đặc trưng
1. `comment_ratio`: Tỷ lệ comment trên tổng số ký tự.
2. `empty_line_ratio`: Tỷ lệ dòng trống trên tổng số dòng.
3. `avg_line_length`: Chiều dài trung bình của các dòng code.
4. `max_line_length`: Chiều dài tối đa của một dòng code.
5. `tab_vs_space_ratio`: Tỷ lệ sử dụng Tab so với Space.
6. `trailing_space_ratio`: Tỷ lệ các dòng có khoảng trắng thừa ở cuối.
7. `brace_style_consistency`: Độ nhất quán trong phong cách đặt dấu ngoặc nhọn (K&R vs Allman).

### [B] Naming Conventions (Quy tắc đặt tên) - 5 đặc trưng
8. `avg_identifier_length`: Chiều dài trung bình của các định danh (tên biến, hàm...).
9. `identifier_length_variance`: Phương sai chiều dài của các định danh.
10. `single_char_var_ratio`: Tỷ lệ sử dụng biến có 1 ký tự.
11. `unique_identifier_ratio`: Tỷ lệ định danh độc nhất.
12. `keyword_to_identifier_ratio`: Tỷ lệ từ khóa C++ so với các định danh tự đặt.

### [C] Structural Complexity (Độ phức tạp cấu trúc) - 10 đặc trưng
13. `avg_cyclomatic_complexity`: Độ phức tạp Cyclomatic trung bình của các hàm.
14. `num_functions`: Tổng số lượng hàm.
15. `avg_function_loc`: Số dòng code trung bình của một hàm.
16. `halstead_volume`: Chỉ số Khối lượng (Volume) theo Halstead.
17. `halstead_difficulty`: Chỉ số Độ khó (Difficulty) theo Halstead.
18. `halstead_effort`: Chỉ số Nỗ lực (Effort) theo Halstead.
19. `halstead_bugs`: Ước lượng số lượng lỗi (Bugs) theo Halstead.
20. `maintainability_index`: Chỉ số khả năng bảo trì.
21. `code_to_comment_ratio`: Tỷ lệ số dòng code thực tế so với số dòng comment.
22. `max_nesting_depth`: Độ sâu lồng nhau tối đa.

### [D] Coding Habits & Idioms (Thói quen lập trình) - 7 đặc trưng
23. `total_includes`: Tổng số lượng thư viện được include.
24. `has_bits_stdc`: Có sử dụng thư viện gom `bits/stdc++.h` hay không.
25. `macro_count`: Số lượng macro (`#define`) được sử dụng.
26. `modern_cpp_ratio`: Tỷ lệ sử dụng các từ khóa C++ hiện đại (auto, nullptr...).
27. `const_usage_ratio`: Tỷ lệ sử dụng từ khóa `const` hoặc `constexpr`.
28. `has_fast_io`: Có sử dụng các câu lệnh tối ưu I/O.
29. `newline_style_ratio`: Thói quen xuống dòng (Tỷ lệ dùng `\n` so với `endl`).

### [E] Information Theory (Lý thuyết thông tin) - 3 đặc trưng
30. `shannon_entropy`: Độ hỗn loạn (Entropy) tính trên toàn bộ code thuần.
31. `bigram_entropy`: Độ hỗn loạn tính trên các cặp ký tự liền kề (Bigram).
32. `whitespace_entropy`: Độ hỗn loạn của các khoảng trắng.
