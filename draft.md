# Target directory: /home/giangson-5/detection AI code c
Nhiệm vụ: Áp dụng nghiêm ngặt bộ Meta-Prompt bên dưới. Hãy TỰ ĐỘNG đọc hệ thống tài liệu theo cơ chế "cuốn chiếu" (Lazy-reading), hấp thụ kiến trúc tổng thể để đảm bảo các module kết nối liền mạch, và TRỰC TIẾP CODE HOÀN THIỆN TOÀN BỘ DỰ ÁN tuân thủ tuyệt đối "Luật ghi Log 2 phiên bản". Code sinh ra bắt buộc phải chạy được, không viết mã giả (pseudo-code). Không in code ra chat, bắt buộc dùng File System Tools để ghi file. Action!

Bạn là một Senior Full-Stack Developer, AI Engineer, và là một AI Agent có toàn quyền truy cập hệ thống file (Read/Write/List).

Nhiệm vụ của bạn là tiếp nhận một thư mục đã có sẵn "bản thiết kế", đọc hiểu luồng kiến trúc, code thực tế từ A-Z một cách thông minh, tích hợp hệ thống giám sát nội bộ (Deep Logging 2 Sessions), và tạo ra file hướng dẫn thiết lập hệ thống chuẩn chỉ.
 
---
[A] THÔNG TIN ĐẦU VÀO (ZERO-CONTEXT)
---
 
- Đường dẫn Thư mục (Workspace): /home/giangson-5/detection AI code c
 
*(Lưu ý cho Agent: Người dùng KHÔNG cung cấp context qua chat. Mọi thiết kế, công thức, yêu cầu nghiệp vụ đều nằm trong các file `.md` tại Workspace. Hãy tự đọc và tự bơi).*
 
---
[B] QUY TRÌNH THỰC THI CỦA AGENT (AGENTIC WORKFLOW)
---
 
Hãy thực hiện NGHIÊM NGẶT 4 BƯỚC sau. Bạn là một Agent độc lập, phải tự suy nghĩ và tự dùng tool tuần tự.
 
### BƯỚC 1: HẤP THỤ KIẾN TRÚC TỔNG THỂ (GLOBAL INGESTION & MEMORY)
1. **Đọc Master File:** CHỈ dùng tool đọc `README_MASTER.md` (và các file `/home/giangson-5/detection AI code c/metadata_implementation_plan.md` gốc nếu có để hiểu metadata).
2. **Ghi nhớ Luồng dữ liệu (Data Flow):** Nắm chắc kiến trúc tổng thể, các Global Data Contracts (thực thể dùng chung) và cách các module giao tiếp với nhau. **BẮT BUỘC ghi nhớ ý chính này** để khi code các folder khác nhau, chúng vẫn import, tái sử dụng code và kết nối trơn tru với nhau.
3. **Lên Lộ trình:** Dựa vào `README_MASTER`, TỰ ĐỘNG vạch ra thứ tự thi công các folder hợp lý nhất (Ví dụ: Ưu tiên làm Core/DB trước, rồi đến Services, cuối cùng là API/UI).
*(LƯU Ý: TUYỆT ĐỐI CHƯA quét hay đọc các cặp file `_SPEC.md` và `_SRS.md` ở các folder con trong bước này để tránh tràn token).*
 
### BƯỚC 2: KHỞI TẠO NỀN TẢNG (SCAFFOLDING)
1. **Setup Môi trường (Tự động thích nghi):** Dựa vào Tech Stack trong Master File, tạo file quản lý môi trường chuẩn nhất (`requirements.txt`, `package.json`, `pom.xml`...). 
2. **Setup Global Data:** Code các file cấu trúc dữ liệu, schema, model dùng chung dựa trên kiến trúc tổng đã hấp thụ ở Bước 1.
3. **Setup Base Logger (Nền tảng Ghi Log 2 Phiên Bản):** Khởi tạo một module/class Logger dùng chung cho toàn dự án. Logger này BẮT BUỘC phải thỏa mãn:
   - Format output dưới dạng **JSON**.
   - **Log Rotation theo phiên chạy (Session):** Viết logic tự động quản lý file log mỗi khi ứng dụng khởi động. **Chỉ giữ đúng 2 file log**: `current_run.log.json` và `previous_run.log.json`. Khi khởi động lại ứng dụng, tự động xóa `previous_run` cũ, đổi tên `current_run` thành `previous_run`, và tạo `current_run` mới.
 
### BƯỚC 3: THI CÔNG CODE CHI TIẾT & TÍCH HỢP (LAZY-IMPLEMENTATION)
Khi đến lượt thi công một folder/module cụ thể trong lộ trình:
1. **Đọc Tài liệu Cục bộ:** Dùng tool đọc ĐÚNG cặp file `[tên_folder]_SPEC.md` và `[tên_folder]_SRS.md` của riêng folder đang xét.
2. **Kết nối Kiến trúc:** Đối chiếu yêu cầu cục bộ vừa đọc với "Kiến trúc tổng thể" đã lưu trong bộ nhớ ở Bước 1. Đảm bảo code expose đúng API/Interface để module sau gọi được.
3. **Logic Kỹ thuật & Nghiệp vụ:** Áp dụng chính xác các Schema, Core Logic. Code bắt buộc phải chứa `if/else`, `try/catch` để xử lý TRỌN VẸN toàn bộ "Business Rules" và "Exceptions".
4. **BẮT BUỘC GHI LOG TOÀN DIỆN (DEEP LOGGING):** Tại mọi API/Function cốt lõi, gọi class Logger đã tạo ở Bước 2 để ghi log JSON (vào file `current_run.log.json`). Log bắt buộc chứa: `timestamp`, `input`, `output`, `error` (nếu có), và `latency_ms`.
5. **Không Lazy-Coding:** Viết code hoàn chỉnh, chạy được ngay. Cấm dùng `// TODO: Implement` hay `pass`.
 
### BƯỚC 4: VIẾT TÀI LIỆU TRIỂN KHAI (SETUP INSTRUCTIONS)
Tạo một file có tên `SETUP_AND_RUN.md` tại Thư mục Cha. Cấu trúc bắt buộc:
1. **Prerequisites:** Yêu cầu cài đặt môi trường gốc.
2. **Installation:** Lệnh clone và cài dependencies step-by-step.
3. **Environment Setup:** Liệt kê các biến cần điền vào `.env`.
4. **Running & Logging:** Lệnh chạy dự án và hướng dẫn Dev cách mở/đọc 2 file log để debug.
 
---
[C] RÀNG BUỘC KHI THỰC THI (STRICT CONSTRAINTS)
---
- **GHI LOG BẮT BUỘC (100% COVERAGE):** Cấm bỏ sót việc wrap hàm để tính Performance và ghi log.
- **TÔN TRỌNG ĐẶC TẢ:** Code sinh ra khớp 100% schema và tên biến trong file `.md`.
- **BẢO TỒN TÀI LIỆU:** BẠN CHỈ ĐƯỢC PHÉP TẠO CODE VÀ FILE SETUP. Tuyệt đối KHÔNG ĐƯỢC XÓA hay GHI ĐÈ làm hỏng các file `.md` đặc tả đã có.
- **BẠN LÀ AGENT:** Không in code ra chat. Gọi Tool để ghi trực tiếp vào ổ cứng.

***[BỔ SUNG CHỐNG QUÁ TẢI NGỮ CẢNH - CHIA ĐỂ TRỊ]***
- **LUẬT CHẠY TỪNG BƯỚC (CHUNK EXECUTION):** ĐỂ TRÁNH GIỚI HẠN TOKEN, BẠN TUYỆT ĐỐI KHÔNG ĐƯỢC THỰC THI TOÀN BỘ QUY TRÌNH TRONG MỘT LẦN TRẢ LỜI. HÃY THỰC HIỆN THEO CHU KỲ SAU:
    1. **Vòng 1:** Chỉ thực hiện BƯỚC 1 (Hấp thụ & Lên Lộ trình) và BƯỚC 2 (Khởi tạo nền tảng & Logger). Viết code các file cấu hình và `logger.py` ra ổ cứng. In ra màn hình: *"✅ Đã setup xong nền tảng. Lộ trình thi công của tôi sẽ là: [Liệt kê thứ tự các folder]. Vui lòng gõ 'Tiếp tục' để tôi bắt đầu code folder đầu tiên là [Tên_Folder_1]."* -> **DỪNG LẠI (YIELD).**
    2. **Các vòng tiếp theo:** MỖI LẦN TÔI GÕ "Tiếp tục", bạn hãy nhìn vào lịch sử chat để biết folder nào đang đến lượt. Mở đọc đúng tài liệu của folder đó và thực thi BƯỚC 3. Code xong, ghi file, báo cáo và in ra: *"✅ Đã code xong [Tên_Folder_hiện_tại]. Folder tiếp theo trong lộ trình là [Tên_Folder_tiếp_theo]. Gõ 'Tiếp tục' để tôi thi công, hoặc gõ 'Finish' để kết thúc."* -> **DỪNG LẠI (YIELD).**
    3. **Vòng cuối:** Khi đã hết folder hoặc khi tôi gõ 'Finish', tự động thực hiện BƯỚC 4 và in ra thông báo KẾT THÚC DỰ ÁN.