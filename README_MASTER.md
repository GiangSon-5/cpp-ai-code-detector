# 🕵️‍♂️ C++ AI Code Detector — Enterprise Backend & MLOps Platform

## System Overview

**Mục tiêu:** Xây dựng hệ thống Backend/MLOps cấp doanh nghiệp để phát hiện mã nguồn C++ do AI sinh ra. Hệ thống nhận đầu vào là đoạn mã C++ thô, trả về kết quả phân tích dạng JSON với điểm AI Score, nhãn phân loại, điểm Perplexity, và giải thích chi tiết (XAI).

### Input/Output Cốt lõi

| | Mô tả | Ví dụ |
|---|---|---|
| **Input** | Code C++ thô (Base64-encoded qua API) | `#include <iostream>\nint main() { ... }` |
| **Output** | JSON chứa prediction + explainability | `{ "final_pred": "AI GENERATED", "final_score": 0.8742, ... }` |

### Kiến trúc Triển khai (Hybrid Local-Cloud)

Do mô hình AI (RoBERTa Ensemble + Qwen 7B) yêu cầu khoảng **14GB-16GB VRAM**, hệ thống được thiết kế chạy theo mô hình **Hybrid** để tối ưu hóa tài nguyên phần cứng hiện có (Laptop 4GB VRAM):

1. **Local (Laptop):** Chạy Web tầng giao diện, Cơ sở dữ liệu và Điều phối.
2. **Google Colab (Cloud GPU):** Chạy "Não bộ" AI và Inference.

```
┌────────────────────────────────────────────────────────────────────┐
│              🖥️  LOCAL MACHINE (Laptop / PC)                       │
│                                                                    │
│  ┌────────────────────────┐      ┌────────────┐  ┌────────────────┐ │
│  │  🌐 DJANGO WEB APP     │      │ Redpanda   │  │ Celery Workers │ │
│  │  (Auth, Dash, Bronze)  │      │ (Broker)   │  │ (ETL, S3 Push) │ │
│  └───────────┬────────────┘      └─────┬──────┘  └───────┬────────┘ │
│              │                         │                 │         │
│              └───────────────┬─────────┴─────────────────┘         │
│                              │                                     │
│      PostgreSQL (Gold) ◄─────┘        DagsHub S3 (Data Lake) ──┐   │
└──────────────────────────────┬─────────────────────────────────│───┘
                               │ HTTPS (ngrok tunnel)            │
                               ▼                                 │
┌────────────────────────────────────────────────────────────────│───┐
│              ☁️  GOOGLE COLAB (Primary GPU Host)               │   │
│                                                                │   │
│  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  ⚡ FASTAPI AI SERVICE (Phần não bộ AI)                  │  │   │
│  │  LangGraph Agent │ RoBERTa Ensemble │ Qwen 2.5 │ LIG/SHAP │  │   │
│  │  Model Weights (Load từ Google Drive)                    │  │   │
│  └──────────────────────────────────────────────────────────┘  │   │
│              ngrok tunnel expose port 8000                     │   │
└────────────────────────────────────────────────────────────────│───┘
                                                                 │
      DVC / MLflow Tracking ◄────────────────────────────────────┘
```

> **Lưu ý:** Việc tách rời này giúp Laptop không bị treo do tràn VRAM (OOM), đồng thời tận dụng được GPU T4/L4 miễn phí từ Colab cho các phép tính XAI (LIG) tốn kém tài nguyên.

---

## Tech Stack & Architecture Decisions

### Công nghệ đã chốt

| Layer | Công nghệ | Lý do chọn |
|-------|-----------|-------------|
| **Web & Admin** | Django 5.x + Django ORM | CRUD chuẩn ACID, Admin UI tự động, Session/Auth có sẵn |
| **AI Serving** | FastAPI (Async) + Pydantic v2 | Non-blocking I/O cho inference nặng, validation tự động |
| **AI Runtime** | ONNX Runtime + PyTorch | ONNX cho production (2-5x faster), PyTorch cho dev/train |
| **GPU Host** | **Google Colab (Primary)** | Chạy AI Engine (RoBERTa + Qwen) vì cần >14GB VRAM |
| **Local Host** | **Laptop/PC** | Chạy Django, DB, Redpanda (nhẹ, không cần GPU mạnh) |
| **LLM Router** | LangGraph + Qwen 2.5 Coder 7B | Agentic workflow: Router → Analyzer → Judge → Critique |
| **Database** | PostgreSQL 16 | ACID, JSON field cho Gold layer, full-text search |
| **Message Broker** | Redpanda | Kafka-compatible, nhẹ hơn 10x RAM, kiêm Cache layer |
| **Background Jobs** | Celery + Redpanda (as broker) | Async logging, ETL push to Data Lake |
| **Data Lake** | DagsHub S3-compatible | Offload Bronze/Silver data khỏi local RAM. Tích hợp sẵn DVC & MLflow |
| **Feature Store** | Parquet trên DagsHub S3 | Columnar format, tối ưu cho pandas/LightGBM training |
| **XAI** | SHAP + Layer Integrated Gradients (LIG) | Giải thích từng token/feature ảnh hưởng đến dự đoán |
| **Infra** | Ubuntu + MicroK8s (GPU, DNS) | Single-node K8s, nhẹ, hỗ trợ GPU passthrough |
| **Tunneling** | Cloudflare Tunnels | Public web (production). ngrok chỉ dùng cho Colab fallback |
| **IaC** | Terraform (K8s resources) + Ansible (OS/Driver) | Terraform quản lý Helm charts, Ansible cài đặt bare-metal |
| **Monitoring** | Prometheus + Grafana + Loki | Metrics (Prometheus), Dashboard (Grafana), Logs (Loki) |
| **CI/CD** | GitHub Actions | Auto test, build Docker image, deploy to MicroK8s |

### Quyết định quan trọng: Offload Data Lake lên DagsHub S3

```
❌ KHÔNG LÀM: Lưu Bronze/Silver raw data trên local disk
   → Tiêu tốn RAM/SSD, không version control, khó scale

✅ ĐÃ CHỌN: Push Bronze/Silver lên DagsHub S3 via Celery background task
   → Local chỉ giữ PostgreSQL (Gold) + Model weights
   → DagsHub cung cấp DVC (data versioning) + MLflow (experiment tracking) miễn phí
   → Tiết kiệm 60-80% storage local
```

---

## Project Structure

```
C:\Users\Admin\Desktop\New folder\
│
├── README_MASTER.md                          # 📋 Tài liệu tổng thể (file này)
├── metadata_implementation_plan.md           # 📊 Kế hoạch Medallion Data Architecture
│
├── src/                                      # 🏗️ SOURCE CODE CHÍNH
│   ├── django_web/                           # 🌐 Django Web Application
│   │   ├── apps/
│   │   │   ├── accounts/                     # User Authentication & Profile
│   │   │   ├── submissions/                  # Bronze Layer CRUD (Code submissions)
│   │   │   └── dashboard/                    # Gold Layer Analytics & Visualization
│   │   ├── templates/                        # Jinja2 HTML templates
│   │   └── static/                           # CSS/JS/Images
│   │
│   ├── fastapi_service/                      # ⚡ FastAPI AI Serving Microservice
│   │   ├── routers/                          # API endpoints (predict, health, stream)
│   │   ├── models/                           # SQLAlchemy Async ORM models (Gold)
│   │   ├── schemas/                          # Pydantic request/response schemas
│   │   ├── repositories/                     # Repository Pattern (CRUD abstraction)
│   │   ├── services/                         # Business logic orchestration
│   │   ├── core/                             # Config, DB session, dependencies
│   │   └── engine/                           # AI Engine (RoBERTa, LIG, Router logic)
│   │
│   ├── colab_runtime/                        # ☁️ GOOGLE COLAB GPU RUNTIME
│   │   ├── notebooks/                        # Jupyter notebooks (.ipynb)
│   │   │   ├── 01_inference_server.ipynb     # 🎯 Chạy FastAPI + ngrok trên Colab
│   │   │   ├── 02_retrain_pipeline.ipynb     # Tái huấn luyện trên Colab GPU
│   │   │   └── 03_export_onnx.ipynb          # Export ONNX cho production
│   │   └── scripts/                          # Python modules (import từ notebook)
│   │       ├── engine.py                     # RoBERTa, LIG, heuristic (từ extract_1.py)
│   │       ├── agent.py                      # LangGraph workflow (từ extract_1.py)
│   │       ├── server.py                     # FastAPI + SSE + ngrok (từ extract_1.py)
│   │       ├── llm_handler.py                # Qwen LLM + Perplexity (từ extract_1.py)
│   │       ├── feature_extractor.py          # CppFeatureExtractorV8 (từ hybrid notebook)
│   │       ├── hybrid_evaluator.py           # Fusion + SHAP (từ hybrid notebook)
│   │       └── config.py                     # API keys, paths, thresholds
│   │
│   ├── celery_workers/                       # 🔄 Celery Background Tasks
│   │   └── tasks/                            # Task definitions (log, ETL, push S3)
│   │
│   └── data_pipeline/                        # 📦 ETL Pipeline (Medallion Architecture)
│       ├── bronze_to_silver/                 # Raw → Features/Tokens transformation
│       ├── silver_to_gold/                   # Aggregation → PostgreSQL Gold tables
│       └── retraining/                       # Closed-loop retraining orchestration
│
├── infrastructure/                           # 🏛️ INFRASTRUCTURE AS CODE
│   ├── terraform/                            # Terraform configurations
│   │   ├── modules/
│   │   │   ├── k8s/                          # MicroK8s resource definitions
│   │   │   └── helm/                         # Helm chart releases (PostgreSQL, Redpanda, Grafana)
│   │   └── environments/
│   │       ├── dev/                           # Dev environment variables
│   │       └── prod/                          # Production environment variables
│   │
│   ├── ansible/                              # Ansible automation
│   │   ├── playbooks/                        # Main playbook files
│   │   ├── roles/
│   │   │   ├── nvidia_driver/                # NVIDIA Driver + CUDA installation
│   │   │   ├── microk8s/                     # MicroK8s setup + GPU addon
│   │   │   └── cloudflare_tunnel/            # Cloudflare Tunnel daemon
│   │   └── inventory/                        # Host definitions
│   │
│   └── k8s_manifests/                        # Raw Kubernetes manifests
│       ├── base/                             # Base manifests (Kustomize)
│       └── overlays/
│           ├── dev/                           # Dev overrides
│           └── prod/                          # Prod overrides
│
├── monitoring/                               # 📈 OBSERVABILITY STACK
│   ├── prometheus/                           # Prometheus config & rules
│   ├── grafana/
│   │   ├── dashboards/                       # Pre-built dashboard JSON
│   │   └── provisioning/                     # Auto-provisioning config
│   └── loki/                                 # Log aggregation config
│
├── .github/
│   └── workflows/                            # 🚀 GitHub Actions CI/CD pipelines
│
├── tests/                                    # 🧪 TEST SUITES
│   ├── unit/                                 # Unit tests (pytest)
│   ├── integration/                          # Integration tests (API + DB)
│   └── e2e/                                  # End-to-end tests
│
├── docs/                                     # 📚 ADDITIONAL DOCUMENTATION
│   ├── architecture/                         # Architecture diagrams & ADRs
│   └── api/                                  # OpenAPI/Swagger exports
│
├── data_lake/                                # 📦 LOCAL DATA LAKE (synced to DagsHub S3)
│   ├── bronze/                               # Raw code submissions (.jsonl)
│   ├── silver/
│   │   ├── ml/                               # LightGBM features (.parquet)
│   │   └── dl/                               # RoBERTa tokens (.parquet)
│   └── gold/                                 # Exported predictions (.parquet)
│
├── ml_models/                                # 🧠 MODEL ARTIFACTS
│   ├── onnx/                                 # Exported ONNX models
│   └── checkpoints/                          # PyTorch checkpoints (dev only)
│
└── legacy_code/                              # 📁 ORIGINAL CODE (Read-Only Archive)
    ├── app.py                                # Streamlit frontend (original)
    ├── extract_1.py                          # FastAPI + LangGraph backend (original)
    └── extracted_with_markdown.py            # Hybrid model evaluation (original)
```

---

## Pipeline Architecture — Luồng Dữ liệu Tổng thể

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          👤 USER INPUT                                   │
│                  Paste/Upload C++ code (.cpp/.c)                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  🌐 DJANGO WEB                                                          │
│                                                                          │
│  1. User Authentication (Session/JWT)                                    │
│  2. Validate Input → Base64 encode                                       │
│  3. BronzeRepository.save() → PostgreSQL (bronze_submissions)            │
│  4. Publish message to Redpanda Topic: `code.submitted`                  │
│                                                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
┌────────────────────┐ ┌─────────────┐ ┌─────────────────────────┐
│  ⚡ FASTAPI         │ │ 🔄 CELERY   │ │  📦 REDPANDA            │
│  AI SERVICE         │ │  WORKER     │ │  Message Broker         │
│                     │ │             │ │                         │
│  ┌───────────────┐  │ │ Task 1:     │ │  Topics:                │
│  │ LangGraph     │  │ │ Push raw    │ │  • code.submitted       │
│  │ Agent Flow:   │  │ │ code to     │ │  • prediction.completed │
│  │               │  │ │ DagsHub S3  │ │  • retrain.trigger      │
│  │ ① Router      │  │ │ (Bronze)    │ │                         │
│  │   ↓           │  │ │             │ │  Cache Layer:           │
│  │ ② Analyzer    │  │ │ Task 2:     │ │  • code_hash → result   │
│  │   (RoBERTa    │  │ │ Extract     │ │    (TTL: 24h)           │
│  │    Ensemble   │  │ │ features →  │ └─────────────────────────┘
│  │    + LIG)     │  │ │ Push to     │
│  │   ↓           │  │ │ DagsHub S3  │
│  │ ③ Judge       │  │ │ (Silver)    │
│  │   (Self-      │  │ │             │
│  │    Correct)   │  │ │ Task 3:     │
│  │   ↓           │  │ │ Write Gold  │
│  │ ④ Critique    │  │ │ prediction  │
│  │   (Map-Reduce │  │ │ to Postgres │
│  │    LLM)       │  │ └─────────────┘
│  └───────────────┘  │
│                     │
│  SSE Streaming      │
│  Response → User    │
└─────────┬───────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     📊 DATA FLOW (MEDALLION)                            │
│                                                                         │
│  🥉 BRONZE (DagsHub S3)          🥈 SILVER (DagsHub S3)                │
│  ┌─────────────────────┐         ┌─────────────────────────┐           │
│  │ raw_code.jsonl       │    ──►  │ Silver-ML: 32 features  │           │
│  │ + metadata           │    ──►  │ Silver-DL: 512 tokens   │           │
│  │ + code_hash          │         │ (.parquet format)        │           │
│  └─────────────────────┘         └─────────────────────────┘           │
│                                           │                             │
│                                           ▼                             │
│                              🥇 GOLD (PostgreSQL)                       │
│                              ┌──────────────────────┐                   │
│                              │ Predictions + SHAP    │                   │
│                              │ Model Performance     │                   │
│                              │ User Analytics        │                   │
│                              └──────────────────────┘                   │
│                                           │                             │
│                                           ▼                             │
│                              📈 DASHBOARD (Django)                      │
│                              ┌──────────────────────┐                   │
│                              │ Charts, Stats, Export │                   │
│                              └──────────────────────┘                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Closed-Loop Retraining Pipeline

```
DagsHub S3 (Silver Parquet)
        │
        ▼
┌─────────────────────────────┐
│  🔁 Retraining Pipeline     │
│                              │
│  1. DVC pull Silver data     │
│  2. Train LightGBM + BERT   │
│  3. Evaluate on holdout set  │
│  4. Log metrics to MLflow    │
│  5. If improved → Export     │
│     ONNX → Push to Registry  │
│  6. Redpanda: retrain.done   │
│  7. FastAPI hot-reload model │
└─────────────────────────────┘
```

---

## Output JSON Schema (Chiết xuất từ code cũ)

Trường dữ liệu output được chiết xuất trực tiếp từ `extract_1.py` (lines 465-470) và `app.py` (lines 177-186):

```json
{
  "final_pred": "AI GENERATED",
  "final_score": 0.8742,
  "model_used": "C++ OOP Model",
  "perplexity": 2.34,
  "is_ambiguous": false,
  "total_tokens": 1024,
  "total_chunks": 2,
  "global_critique": "The code exhibits consistent AI-generated patterns...",
  "global_html": "<html>...(LIG heatmap visualization)...</html>",
  "chunks": [
    {
      "index": 1,
      "score": 0.9123,
      "label": "AI",
      "top_ai": ["'iostream'", "'endl'", "'return'"],
      "top_hu": ["'ptr'", "'idx'"],
      "snippet": "#include <iostream>\nint main() { ... }",
      "html": "<html>...(chunk-level heatmap)...</html>",
      "critique": "This chunk shows typical AI-generated boilerplate patterns..."
    }
  ]
}
```

---

## Quick Start (Hybrid Mode)

### 1. Chuẩn bị Local (Laptop)
```bash
# Chạy các service hỗ trợ (DB, Broker)
docker-compose up -d
cd src/django_web && python manage.py migrate && python manage.py runserver
```

### 2. Chuẩn bị AI Cloud (Google Colab)
- Mở `src/colab_runtime/notebooks/01_inference_server.ipynb` trên Colab.
- Chọn Runtime: **GPU (T4 hoặc L4)**.
- Chạy các cell → Nhận URL ngrok (VD: `https://xyz.ngrok.io`).
- Cấu hình trong Django `.env`: `FASTAPI_AI_URL=https://xyz.ngrok.io`.

### 3. Kiểm tra
- Truy cập `localhost:8000` (Django).
- Paste code C++ và ấn Submit.
- Django sẽ gọi qua ngrok tới Colab để xử lý AI.

---

## Tài liệu tham chiếu

| File | Vị trí | Mô tả |
|------|--------|--------|
| `metadata_implementation_plan.md` | Root | Medallion Data Architecture chi tiết |
| `*_SPEC.md` | Mỗi module folder | Đặc tả kỹ thuật cho Developer |
| `*_SRS.md` | Mỗi module folder | Đặc tả nghiệp vụ / Use Case |
| `extract_1.py` | `legacy_code/` | Code gốc FastAPI + LangGraph |
| `app.py` | `legacy_code/` | Code gốc Streamlit frontend |
| `extracted_with_markdown.py` | `legacy_code/` | Code gốc Hybrid model evaluation |
