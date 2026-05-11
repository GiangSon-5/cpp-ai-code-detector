# Module: Shared Utilities (SPEC)

## 1. Module Overview
Module này chứa các hàm dùng chung (helper functions) không thuộc về logic nghiệp vụ chính nhưng cần thiết để hệ thống vận hành trơn tru, bao gồm quản lý file, giải nén và cấu hình.

## 2. Data Contracts & Examples
**Input/Output:** Các tham số tùy biến tùy theo hàm (Đường dẫn file, buffer, etc).

## 3. Core Logic & Formulas
- **Zip Management:** Sử dụng thư viện `zipfile` để giải nén các tập dataset lớn từ Google Drive/Local.
- **Path Handling:** Sử dụng `os` và `glob` để quản lý đường dẫn file linh hoạt trên các hệ điều hành khác nhau.

## 4. End-to-End Trace Example
- **Hàm `unzip_file`:**
    - Input: `test_set.zip`, `target_dir`.
    - Trace: Mở zip -> Extract all -> In thông báo thành công.
- **Hàm `load_jsonl`:**
    - Input: `data.jsonl`.
    - Trace: Đọc từng dòng -> `json.loads` -> Append vào list.

## 5. LLM Prompts/Integrations
N/A.

## 6. Edge Cases
- **Disk Full:** Khi giải nén dataset lớn (ví dụ 100k samples).
- **File Locked:** File đang được mở bởi tiến trình khác.
