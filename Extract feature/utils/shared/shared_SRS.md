### UC: Tiện ích dùng chung - Module Shared Utilities

**Mô tả chức năng tổng quan**
Cung cấp các công cụ bổ trợ cho việc quản lý tài nguyên hệ thống, xử lý tệp tin và các thiết lập cấu hình chung.

| Primary Actor: | System Modules | Secondary Actor: | OS File System |
|----------------|---------------|------------------|---------------------|
| **Description:** | Các hàm bổ trợ giúp hệ thống đọc/ghi và giải nén dữ liệu hiệu quả. |
| **Trigger:** | Các module khác gọi đến khi cần xử lý file. |
| **Preconditions:** | PRE1: Có quyền truy cập đọc/ghi vào hệ thống tệp tin. |
| **Post-conditions:** | POST1: Thực hiện thành công các tác vụ quản trị file. |

**Business Scenario Walkthrough:**
- **Hệ thống đưa vào:** Một file nén `.zip` chứa 100.000 mẫu code bài tập.
- **Module xử lý:** Tự động giải nén vào thư mục tạm, kiểm tra dung lượng và cấu trúc thư mục sau khi giải nén.
- **Kết quả nhận được:** Một môi trường dữ liệu sẵn sàng để các module Core và Pipeline làm việc.

**Normal Flow:**
| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Gọi hàm giải nén | Hệ thống kiểm tra sự tồn tại của file zip |
| 2 | Thực hiện giải nén | Hệ thống bung dữ liệu ra thư mục đích |
| 3 | Load dữ liệu JSONL | Chuyển đổi định dạng văn bản thành danh sách Object trong Python |

**Exception:**
| No | Cause | System Response |
|----|-------|-----------------|
| 1 | File Zip bị lỗi (Corrupted) | Thông báo lỗi "Bad Zip File" |

**Business Rules:**
| No | Rule |
|----|------|
| 1 | Các đường dẫn (Paths) nên được quản lý tập trung để dễ dàng thay đổi khi chuyển môi trường (Colab sang Local). |
