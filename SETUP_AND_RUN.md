# 🕵️‍♂️ Hướng dẫn Cài đặt & Vận hành Chi tiết — C++ AI Code Detector

Chào mừng bạn đến với hệ thống phát hiện mã nguồn C++ do AI sinh ra. Tài liệu này hướng dẫn bạn từng bước một để thiết lập hệ thống chạy trên mô hình **Hybrid**: "Não bộ" AI chạy trên Cloud (Google Colab) và "Cơ thể" điều phối chạy tại máy Local của bạn.

---

## 📋 Tổng quan kiến trúc
Hệ thống gồm 2 thành phần chính giao tiếp với nhau qua Internet (ngrok):
1.  **GPU Worker (Google Colab):** Chạy các model AI nặng (RoBERTa, Qwen 2.5 Coder 7B, vLLM). Yêu cầu GPU T4/L4.
2.  **Local Brain (Máy của bạn):** Chạy Django (Giao diện người dùng), FastAPI (Điều phối), và PostgreSQL (Lưu trữ dữ liệu).

---

## 🛠️ Bước 1: Chuẩn bị GPU Worker (Trên Google Colab)

Vì mô hình AI yêu cầu hơn 14GB VRAM, chúng ta sẽ tận dụng GPU miễn phí của Google Colab.

1.  **Mở Colab:** Truy cập [Google Colab](https://colab.research.google.com/).
2.  **Cấu hình GPU:** Vào menu **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** (hoặc L4 nếu có).
3.  **Tải Code lên:** Upload thư mục `src/colab_runtime/` từ máy bạn lên không gian lưu trữ của Colab.
4.  **Khởi chạy:** Chạy lệnh sau trong một cell:
    ```python
    !python bootstrap.py
    ```
5.  **Lấy URL:** Đợi khoảng 3-5 phút để hệ thống cài đặt và load model. Khi hoàn tất, nó sẽ in ra một khung thông báo có chứa **ngrok URL** (ví dụ: `https://xxxx-xxxx.ngrok-free.dev`). 
    > ⚠️ **Hãy copy URL này để dùng cho Bước 2.**

---

## 💻 Bước 2: Thiết lập máy Local (Tại máy tính của bạn)

### 1. Cài đặt môi trường
Mở terminal tại thư mục dự án và thực hiện:
```bash
# Tạo môi trường ảo (khuyên dùng Conda)
conda create -n detection_ai python=3.11 -y
conda activate detection_ai

# Cài đặt công cụ quản lý gói uv và các dependencies
pip install uv
uv pip install -r requirements.txt
```

### 2. Khởi động các dịch vụ nền (Docker)
Hệ thống cần PostgreSQL (Database) và Redpanda (Message Broker).
```bash
# Hãy chắc chắn bạn đã bật Docker Desktop
sudo docker-compose up -d
```
*Kiểm tra bằng lệnh `docker ps`, bạn phải thấy 3 container (postgres, redis, redpanda) đang ở trạng thái `Up (healthy)`.*

### 3. Cấu hình biến môi trường (.env)
Tạo file `.env` tại thư mục gốc dự án (nếu chưa có) và cấu hình như sau:
```ini
# Dán URL bạn vừa lấy từ Bước 1 vào đây
FASTAPI_AI_URL=https://xxxx-xxxx.ngrok-free.dev
COLAB_API_KEY=colab-secret-key-123

# Cấu hình Database (Mặc định theo docker-compose)
DB_NAME=cpp_detector
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Key AI (Dùng cho các tính năng bổ trợ)
GEMINI_API_KEY=AIzaSy... (Key của bạn)
```

### 4. Khởi tạo Database
```bash
export DJANGO_SETTINGS_MODULE="src.django_web.settings"
python manage.py migrate
python manage.py createsuperuser  # Tạo tài khoản đăng nhập web
```

---

## 🚀 Bước 3: Chạy hệ thống (Vận hành 3 Terminal)

Bạn cần mở **3 tab Terminal** riêng biệt để chạy toàn bộ các thành phần:

### Terminal 1: FastAPI AI Orchestrator (Cổng 8001)
Đây là cầu nối giữa Web và Colab.
```bash
conda activate detection_ai
uvicorn src.fastapi_service.main:app --port 8001 --reload
```

### Terminal 2: Django Web UI (Cổng 8000)
Giao diện chính để bạn dán code và xem kết quả.
```bash
conda activate detection_ai
python manage.py runserver 8000
```

### Terminal 3: Celery Background Workers
Xử lý các tác vụ ghi log và đẩy dữ liệu lên Data Lake.
```bash
conda activate detection_ai
celery -A src.celery_workers.celery_app worker --loglevel=info
```

---

## 📊 Bước 4: Sử dụng và Giám sát

1.  **Trang chủ:** Truy cập `http://localhost:8000`. Đăng nhập và bắt đầu phân tích code.
2.  **Trang Admin:** `http://localhost:8000/admin` (Quản lý dữ liệu).
3.  **API Swagger:** `http://localhost:8001/docs` (Xem chi tiết các endpoint AI).
4.  **Hệ thống Log:** Xem file `logs/current_run.log.json` để theo dõi hiệu năng và lỗi theo thời gian thực.

---

## 🛠️ Xử lý sự cố thường gặp (Troubleshooting)

| Lỗi | Nguyên nhân & Cách khắc phục |
| :--- | :--- |
| **Address already in use (8001)** | Cổng 8001 bị chiếm. Chạy lệnh: `fuser -k 8001/tcp` rồi khởi động lại. |
| **Trạng thái PENDING mãi** | FastAPI không kết nối được Colab. Kiểm tra `FASTAPI_AI_URL` trong `.env` và đảm bảo Colab vẫn đang chạy. |
| **Docker command not found** | Chưa bật tích hợp WSL trong Docker Desktop Settings -> Resources -> WSL integration. |
| **Colab bị OOM (Out of Memory)** | Code bạn dán vào quá dài. Hãy thử chia nhỏ file code hoặc reset Runtime Colab. |
| **Lỗi DB Postgres** | Đảm bảo container postgres đang chạy (`docker ps`). Nếu không, chạy `docker-compose up -d` lại. |

---
*Tài liệu được tổng hợp và tối ưu hóa cho người dùng cuối. Chúc bạn có trải nghiệm tốt với dự án!*
