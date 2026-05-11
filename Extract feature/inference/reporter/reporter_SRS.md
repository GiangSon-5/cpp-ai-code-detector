### UC: Dự đoán và Báo cáo kết quả - Module Inference & Reporting

**Mô tả chức năng tổng quan**
Sử dụng mô hình đã huấn luyện để quét và đánh giá nguồn gốc của hàng loạt file mã nguồn trong thực tế, cung cấp cái nhìn tổng quan về tỷ lệ sử dụng AI trong các tập dữ liệu khác nhau.

| Primary Actor: | End User / Auditor | Secondary Actor: | Matplotlib, Seaborn |
|----------------|---------------|------------------|---------------------|
| **Description:** | Quét thư mục và đưa ra kết luận về nguồn gốc của mã nguồn (AI hay Human). |
| **Trigger:** | Người dùng cung cấp đường dẫn thư mục cần kiểm tra. |
| **Preconditions:** | PRE1: Đã có mô hình (Model) và bộ chuẩn hóa (Scaler) đã được huấn luyện thành công. |
| **Post-conditions:** | POST1: Hiển thị bảng tổng kết và các biểu đồ trực quan hóa. |

**Business Scenario Walkthrough:**
- **Khách hàng/Hệ thống đưa vào:** Một thư mục chứa bài làm của 50 sinh viên.
- **Module xử lý:** Tự động đi sâu vào từng thư mục con của từng sinh viên, chạy mô hình dự đoán cho từng file code. Tổng hợp xem sinh viên nào có tỷ lệ code giống AI cao bất thường.
- **Kết quả nhận được:** Một bảng xếp hạng các thư mục có nguy cơ cao là AI, kèm theo biểu đồ phân phối xác suất để người dùng tự đánh giá.

**Normal Flow:**
| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Cấu hình PARENT_DIR và Threshold | Hệ thống bắt đầu quét cây thư mục |
| 2 | Chạy tiến trình Inference | Hệ thống xử lý song song hoặc tuần tự các file code tìm thấy |
| 3 | Tổng hợp kết quả theo Folder | Hệ thống tính toán Tỷ lệ % cho từng đơn vị thư mục |
| 4 | Trực quan hóa kết quả | Hiển thị Heatmap Confusion Matrix và KDE Plot xác suất |
| 5 | Phân tích lỗi | Hiển thị các trường hợp dự đoán sai (nếu có Ground Truth) |

**Exception:**
| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Không tìm thấy file code nào | Thông báo "Không tìm thấy file C/C++" |
| 2 | File bị lỗi format nặng | Bỏ qua file đó và ghi nhận vào Error Count |

**Business Rules:**
| No | Rule |
|----|------|
| 1 | Phải hỗ trợ quét đệ quy (recursive scan) để xử lý các cấu trúc dự án phức tạp. |
| 2 | Kết quả dự đoán phải đi kèm với xác suất (Confidence Score) để người dùng có căn cứ tin cậy. |

**Bảng mô tả giao diện (UI) / Data Mapping:**

**1. Thông tin đầu ra (Dashboard):**
| Tên trường | Mô tả | Định dạng hiển thị |
|------------|-------|-----------------|
| Folder Name | Tên thư mục đang xét | Text |
| Total Files | Tổng số file code | Integer |
| AI Ratio | Tỷ lệ dự đoán là AI | Percentage (xx.xx%) |
| Confidence | Mức độ tự tin trung bình của model | Float (0.0 - 1.0) |
