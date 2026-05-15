# 🗺️ PROJECT ARCHITECTURE MAP & DIRECTORY KNOWLEDGE BASE

Tài liệu này là "Bản đồ chỉ hướng" (Master Navigation Map) và là "Nguồn tri thức tổng thể" (Single Source of Truth) cho dự án C++ AI Code Detector. Nó cung cấp góc nhìn chi tiết từ cấp độ hệ thống đến từng thư mục và tệp tin cốt lõi, giúp Developer, DevOps và các AI Agent khác nhanh chóng nắm bắt và định vị kiến trúc dự án.

---

## 1. System Topology Overview (Tóm tắt nhanh)

Hệ thống được thiết kế theo 2 triết lý kiến trúc lõi:
1. **Hybrid Cloud-Local Architecture**: Tách biệt luồng xử lý nhẹ (I/O, Orchestration, Dashboard) và luồng xử lý nặng (AI Inference, VRAM-heavy).
    - **Local Machine**: Chạy Django Web, FastAPI Orchestrator, PostgreSQL, Celery, Redis/Redpanda.
    - **GPU Runtime**: Chạy Local GPU Server (hoặc có thể offload lên Google Colab) để thực thi các model nặng (RoBERTa, Qwen 2.5) nhằm tránh OOM cho máy chủ chính.
2. **Medallion Data Architecture**: Quản lý dữ liệu qua 3 tầng (Bronze 🥉, Silver 🥈, Gold 🥇) phục vụ cho MLOps và quá trình huấn luyện lại (Retraining Pipeline).

---

## 2. Global Directory & Component Mapping

Dưới đây là bản đồ quy chiếu chi tiết từ thư mục vật lý đến các vai trò kiến trúc trong hệ thống:

| Kiến trúc / Dịch vụ | Thư mục & File đảm nhiệm | Mô tả chức năng cốt lõi |
|---|---|---|
| **Django Web App & Dashboard** | `src/django_web/` | Nơi chứa mã nguồn Web Frontend và Admin Dashboard. |
| ↳ Quản lý User (Auth) | `src/django_web/apps/accounts/` | Đăng nhập, JWT, User profile. |
| ↳ Quản lý Bronze Data | `src/django_web/apps/submissions/` | Tiếp nhận mã nguồn thô từ user, lưu DB và gửi message cho broker. |
| ↳ MLOps Hub & Dashboard | `src/django_web/apps/dashboard/` | Giao diện giám sát hệ thống, Infra, VRAM, Models và Data Metrics. |
| **FastAPI Orchestrator** | `src/fastapi_service/` | Trái tim điều phối AI Pipeline (Port 8001). |
| ↳ API & SSE Streaming | `src/fastapi_service/routers/` | Cung cấp endpoints nhận code và trả kết quả realtime về cho client. |
| ↳ AI Flow (LangGraph) | `src/fastapi_service/services/agent_service.py` | Định nghĩa luồng xử lý Agentic 4-node (Router → Analyzer → Judge → Critique). |
| ↳ AI Node Modules | `src/fastapi_service/engine/` | Chứa các module hỗ trợ: `llm_handler.py`, `fingerprint_engine.py`, `explainer.py`. |
| **GPU Inference Server** | `src/local_gpu_server/` | Server riêng biệt quản lý VRAM & Model Inference (Port 8002). |
| ↳ Core Model Execution | `src/local_gpu_server/engine.py` | Nơi nạp tạ (weights) model, cấu hình pipeline suy luận. |
| ↳ Hybrid Execution | `src/local_gpu_server/hybrid_evaluator.py` | Kế hợp DL & ML inferences. |
| **Background Workers** | `src/celery_workers/` | Xử lý các tác vụ bất đồng bộ. |
| ↳ Pipeline Tasks | `src/celery_workers/tasks/` | Chứa `bronze_tasks.py`, `silver_tasks.py`, `gold_tasks.py`, `retraining_tasks.py`. |
| **Shared Infrastructure** | `src/shared/` | Các module tái sử dụng chung cho toàn bộ ứng dụng. |
| ↳ Shared Config & Data | `src/shared/config.py`, `data_contracts.py` | Các schema cấu hình dùng chung giữa FastAPI, Django và GPU Server. |
| ↳ Infrastructure Clients | `src/shared/message_broker.py`, `database.py` | Cấu hình kết nối RabbitMQ/Redis và PostgreSQL. |

---

## 3. Component Interaction & Protocols

Các dịch vụ trong hệ thống tương tác với nhau thông qua các giao thức và cấu hình rõ ràng:

- **User → Django Web:** HTTP/HTTPS. Người dùng tương tác qua giao diện web, dữ liệu được xác thực và nạp vào hệ thống.
- **Django Web → Message Broker (Redis/Redpanda):** Giao tiếp bất đồng bộ qua message queue. Khi người dùng nộp code, Django ném sự kiện `code.submitted` vào Broker.
- **FastAPI Orchestrator → Local GPU Server:** Giao tiếp qua HTTP RESTful API (Port `8001` gọi Port `8002`). FastAPI gửi code thô và yêu cầu phân tích, GPU Server thực thi và trả về XAI JSON Payload. 
- **Message Broker → Celery Workers:** Celery lắng nghe các topic trên Broker để trigger các tác vụ xử lý dữ liệu ngầm (ETL).
- **Cấu hình kết nối (Connection Strings):** 
  - Toàn bộ thiết lập môi trường được cấu hình tại tệp `.env` (chứa Database URI, Broker URI, S3 Credentials).
  - Khởi tạo infrastructure service (PostgreSQL, Redis) bằng `docker-compose.yml`.
  - Python loader lấy thông tin từ `src/shared/config.py`.

---

## 4. Medallion Data Architecture Implementation

Hệ thống ETL 3 tầng được chia tách rõ ràng thành các logic riêng biệt:

- **🥉 Bronze Layer (Raw Data):**
  - **Logic:** Nhận và lưu trữ raw C++ Code (.jsonl) chưa qua xử lý.
  - **File thực thi:** `src/django_web/apps/submissions/` (Lưu PostgreSQL) và `src/celery_workers/tasks/bronze_tasks.py` (Lưu DagsHub S3).
  - **Thư mục local cache:** `data_lake/bronze/`.

- **🥈 Silver Layer (Feature Store):**
  - **Logic:** Extract đặc trưng (32 Machine Learning features) và Tokenize (512 Deep Learning tokens). Data được chuẩn hóa dạng Parquet để phục vụ ML Training hiệu suất cao.
  - **File thực thi:** `src/data_pipeline/bronze_to_silver/etl.py` (Chứa logic trích xuất).
  - **Worker kích hoạt:** `src/celery_workers/tasks/silver_tasks.py`.

- **🥇 Gold Layer (Predictions & Analytics):**
  - **Logic:** Lưu trữ kết quả AI inference (Predictions), XAI Fingerprints, Model Analytics phục vụ cho báo cáo Dashboard.
  - **File thực thi:** `src/data_pipeline/silver_to_gold/`.
  - **Worker kích hoạt:** `src/celery_workers/tasks/gold_tasks.py`.

---

## 5. MLOps & Model Artifacts Location

Đóng vai trò nền tảng cho việc vận hành AI, MLOps artifact được quy hoạch chặt chẽ như sau:

- **Nơi chứa Model Weights (Checkpoints, ONNX):** `local_models/` (Thư mục vật lý lưu trữ local các model phục vụ Dev/Test như các fold đã train `fold_1_basic`, `fold_1_oop`).
- **Nơi chứa Data Local (Data Lake Simulation):** `data_lake/` (Gồm bronze, silver, gold folder để test data pipeline).
- **Automation Scripts:**
  - Khởi tạo toàn bộ system (Tmux session 4 pane): `dev.sh` hoặc sử dụng `make dev` (định nghĩa tại `Makefile`).
- **Tài liệu MLOps (Knowledge Base Nội bộ):** Được lưu tập trung trong `System architecture/`
  - `00_system_overview.md`
  - `01_mlops_pipeline.md`
  - `02_admin_dashboard.md`
  - `03_shared_infrastructure.md`
  - `04_operations.md`

Bản đồ này là nguồn thông tin trung tâm. Nếu có sự thay đổi về thư mục và luồng kiến trúc, kỹ sư vui lòng update trực tiếp vào tài liệu này!
