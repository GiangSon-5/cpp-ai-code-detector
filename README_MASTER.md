# 🕵️‍♂️ C++ AI Code Detector — Enterprise Backend & MLOps Platform

## System Overview

**Mục tiêu:** Xây dựng hệ thống Backend/MLOps cấp doanh nghiệp để phát hiện mã nguồn C++ do AI sinh ra. Hệ thống nhận đầu vào là đoạn mã C++ thô, trả về kết quả phân tích dạng JSON với điểm AI Score, nhãn phân loại, điểm Perplexity, và giải thích chi tiết (XAI).

### Input/Output Cốt lõi

| | Mô tả | Ví dụ |
|---|---|---|
| **Input** | Code C++ thô (Base64-encoded qua API) | `#include <iostream>\nint main() { ... }` |
| **Output** | JSON chứa prediction + explainability | `{ "final_pred": "AI GENERATED", "final_score": 0.8742, ... }` |

### Kiến trúc Triển khai (Hybrid Local-Cloud)

Hệ thống được thiết kế linh hoạt chạy theo mô hình **Hybrid** để tối ưu hóa tài nguyên phần cứng, chia thành 4 dịch vụ cốt lõi:

```text
┌────────────────────────────────────────────────────────────────────┐
│              🖥️  LOCAL MACHINE (Laptop / PC)                       │
│                                                                    │
│  ┌────────────────────────┐      ┌────────────┐  ┌────────────────┐ │
│  │  🌐 DJANGO WEB APP     │      │ Redis/     │  │ Celery Workers │ │
│  │  (Auth, Dash, Bronze)  │      │ Redpanda   │  │ (ETL, S3 Push) │ │
│  └───────────┬────────────┘      └─────┬──────┘  └───────┬────────┘ │
│              │                         │                 │         │
│              └───────────────┬─────────┴─────────────────┘         │
│                              │                                     │
│  ┌────────────────────────┐  │   PostgreSQL (Gold) ◄─────┘         │
│  │ ⚡ FastAPI Orchestrator│◄─┘                                     │
│  │    (Port 8001)         │           DagsHub S3 (Data Lake) ──┐   │
│  └───────────┬────────────┘                                    │   │
└──────────────┼─────────────────────────────────────────────────│───┘
               │ HTTPS (ngrok tunnel) hoặc localhost             │
               ▼                                                 │
┌────────────────────────────────────────────────────────────────│───┐
│              ☁️ GPU RUNTIME (Local GPU hoặc Google Colab)       │   │
│                                                                │   │
│  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  🔧 LOCAL GPU SERVER (Port 8002) HOẶC COLAB FASTAPI       │  │   │
│  │  LangGraph Agent │ RoBERTa Ensemble │ Qwen 2.5 │ SHAP/LIG│  │   │
│  │  Model Weights (~14-16GB VRAM)                           │  │   │
│  └──────────────────────────────────────────────────────────┘  │   │
└────────────────────────────────────────────────────────────────│───┘
                                                                 │
      DVC / MLflow Tracking ◄────────────────────────────────────┘
```

> **Lưu ý:** Việc tách rời Orchestrator và GPU Runtime giúp Laptop (VRAM thấp) không bị treo do tràn bộ nhớ (OOM). Có thể chạy Local GPU Server nếu có sẵn GPU mạnh, hoặc dùng Google Colab (GPU T4/L4 miễn phí) cho phần xử lý nặng.

---

## Tech Stack & Architecture Decisions

| Layer | Công nghệ | Lý do chọn |
|-------|-----------|-------------|
| **Web & Admin** | Django 5.x + Django ORM | CRUD chuẩn ACID, MLOps Dashboard (6 views) |
| **AI Orchestration**| FastAPI (Async) + Pydantic v2 | Điều phối pipeline AI, SSE streaming realtime |
| **AI Runtime** | ONNX Runtime + PyTorch | ONNX cho production (2-5x faster), PyTorch cho dev |
| **LLM Router** | LangGraph + Qwen 2.5 Coder 7B | Agentic workflow 4 node: Router → Analyzer → Judge → Critique |
| **Database** | PostgreSQL 16 | ACID, JSON field cho Gold layer prediction |
| **Broker** | Redis & Redpanda | Redis (Celery broker), Redpanda (Event streaming) |
| **Background Jobs** | Celery | Async ETL (Bronze → Silver), S3 Push |
| **Data Lake** | DagsHub S3-compatible | Offload dữ liệu, DVC versioning + MLflow tracking |
| **Feature Store** | Parquet trên DagsHub S3 | Lưu ML features & DL tokens, tối ưu cho training |
| **XAI** | SHAP + Layer Integrated Gradients | Giải thích Feature (SHAP) & Token (LIG) |
| **Container/K8s** | Docker Compose / MicroK8s | DB/Broker chạy Docker, Production K8s hỗ trợ GPU |

### Quyết định quan trọng: Medallion Data Architecture

Hệ thống áp dụng kiến trúc dữ liệu 3 tầng (Medallion) để xử lý lượng lớn dữ liệu huấn luyện:
- **🥉 Bronze (DagsHub S3 + PostgreSQL):** Lưu mã nguồn thô (JSONL).
- **🥈 Silver (DagsHub S3):** Lưu đặc trưng đã trích xuất (32 Features cho ML, 512 Tokens cho DL) dưới dạng `.parquet`.
- **🥇 Gold (PostgreSQL):** Lưu kết quả dự đoán cuối cùng (Predictions, XAI Fingerprints) phục vụ Dashboard thống kê.

---

## Project Structure

```text
LVTN-main/
│
├── README_MASTER.md                          # 📋 Tài liệu tổng thể (file này)
├── metadata_implementation_plan.md           # 📊 Kế hoạch Medallion Data Architecture
│
├── src/                                      # 🏗️ SOURCE CODE CHÍNH
│   ├── django_web/                           # 🌐 Django Web & Dashboard
│   │   └── apps/
│   │       ├── accounts/                     # Auth & Profile
│   │       ├── submissions/                  # Bronze Layer CRUD
│   │       └── dashboard/                    # Admin MLOps (overview, metrics, infra, db, users, models)
│   │
│   ├── fastapi_service/                      # ⚡ FastAPI Orchestrator
│   │   ├── routers/                          # API endpoints (SSE, health)
│   │   ├── engine/                           # AI Engine Orchestration
│   │   └── services/
│   │       └── agent_service.py              # LangGraph 4-node pipeline
│   │
│   ├── local_gpu_server/                     # 🔧 Chạy inference trên GPU (Port 8002)
│   │
│   ├── celery_workers/                       # 🔄 Background tasks (ETL, Push S3)
│   │
│   ├── data_pipeline/                        # 📦 Medallion ETL (Bronze→Silver→Gold)
│   │
│   └── shared/                               # 🛠️ Common modules
│       ├── config.py, database.py, logger.py, data_contracts.py
│
├── data_lake/                                # 📦 LOCAL DATA (Sync DagsHub S3)
│   ├── bronze/                               # Raw code (.jsonl)
│   ├── silver/                               # Features/Tokens (.parquet)
│   └── gold/                                 # Predictions
│
├── ml_models/                                # 🧠 MODEL ARTIFACTS
│   ├── onnx/                                 # Exported ONNX models
│   └── checkpoints/                          # PyTorch checkpoints (dev only)
│
├── System architecture/                      # 📚 TÀI LIỆU NỘI BỘ MỚI
│   ├── 00_system_overview.md
│   ├── 01_mlops_pipeline.md
│   ├── 02_admin_dashboard.md
│   ├── 03_shared_infrastructure.md
│   └── 04_operations.md
│
├── legacy_code/                              # 📁 ORIGINAL CODE (Read-Only Archive)
│
├── infrastructure/                           # 🏛️ Terraform, Ansible, K8s
├── monitoring/                               # 📈 Prometheus, Grafana, Loki
├── .github/workflows/                        # 🚀 CI/CD pipelines
├── tests/                                    # 🧪 Unit, Integration, E2E
├── docker-compose.yml                        # PostgreSQL, Redis, Redpanda
├── Makefile                                  # Lệnh `make dev`
└── dev.sh                                    # tmux 2x2 startup script
```

---

## Pipeline Architecture — Luồng Dữ liệu Tổng thể

```text
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
│  4. Publish message to Broker: `code.submitted`                          │
│                                                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
┌────────────────────┐ ┌─────────────┐ ┌─────────────────────────┐
│  ⚡ FASTAPI         │ │ 🔄 CELERY   │ │  📦 REDIS/REDPANDA      │
│  ORCHESTRATOR       │ │  WORKER     │ │  Message Broker         │
│                     │ │             │ │                         │
│  ┌───────────────┐  │ │ Task 1:     │ │  Topics:                │
│  │ LangGraph     │  │ │ Push raw    │ │  • code.submitted       │
│  │ Agent Flow:   │  │ │ code to     │ │  • prediction.completed │
│  │               │  │ │ DagsHub S3  │ │  • retrain.trigger      │
│  │ ① Router      │  │ │ (Bronze)    │ │                         │
│  │   ↓           │  │ │             │ │  Cache Layer:           │
│  │ ② Analyzer    │  │ │ Task 2:     │ │  • code_hash → result   │
│  │   (Local GPU  │  │ │ Extract     │ │    (TTL: 24h)           │
│  │    hoặc Colab)│  │ │ features →  │ └─────────────────────────┘
│  │   ↓           │  │ │ Push to     │
│  │ ③ Judge       │  │ │ DagsHub S3  │
│  │   (Self-      │  │ │ (Silver)    │
│  │    Correct)   │  │ │             │
│  │   ↓           │  │ │ Task 3:     │
│  │ ④ Critique    │  │ │ Write Gold  │
│  │   (Map-Reduce │  │ │ prediction  │
│  │    LLM)       │  │ │ to Postgres │
│  └───────────────┘  │ └─────────────┘
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

```text
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
│  7. API Server hot-reload    │
└─────────────────────────────┘
```

---

## Output JSON Schema

Dữ liệu trả về từ API (SSE Streaming) và được lưu vào Gold layer:

```json
{
  "final_pred": "AI GENERATED",
  "final_score": 0.8742,
  "model_used": "C++ OOP Model",
  "dl_score": 0.91,
  "ml_score": 0.72,
  "hybrid_score": 0.83,
  "perplexity": 2.34,
  "max_ppl": 15.42,
  "burstiness": 3.85,
  "is_ambiguous": false,
  "total_tokens": 1024,
  "total_chunks": 2,
  "global_critique": "The code exhibits consistent AI-generated patterns...",
  "chunks": [
    {
      "index": 1,
      "score": 0.9123,
      "label": "AI",
      "top_ai": ["iostream", "endl", "return"],
      "top_hu": ["ptr", "idx"],
      "snippet": "#include <iostream>...",
      "critique": "This chunk shows typical AI-generated boilerplate..."
    }
  ],
  "fingerprint": { 
      "features": [...],
      "shap_values": [...]
  }
}
```

---

## Quick Start (Local Development)

### Yêu cầu tiên quyết
- Python 3.10+, Conda/Mamba
- Docker & Docker Compose
- Tmux (cho `dev.sh`)

### Khởi động (Chỉ 2 bước)

1. **Khởi động DB & Brokers (Docker):**
   ```bash
   docker-compose up -d
   ```

2. **Khởi động toàn bộ dịch vụ (Tmux 2x2 pane):**
   ```bash
   make dev
   # Hoặc chạy script bash: ./dev.sh
   ```

`make dev` sẽ tự động tạo một phiên Tmux với 4 cửa sổ chạy song song:
- **Pane 0**: FastAPI Orchestrator (Port 8001)
- **Pane 1**: Django Web Dashboard (Port 8000)
- **Pane 2**: Celery Worker (Background tasks)
- **Pane 3**: Local GPU Server (Port 8002)

> Truy cập **http://localhost:8000** để xem giao diện web và **http://localhost:8000/dashboard/** cho MLOps Admin.

---

## Hệ thống Tài liệu Nội bộ

Vui lòng tham khảo các thư mục và file tài liệu sau để nắm bắt chi tiết:
1. `System architecture/00_system_overview.md` - Tổng quan kiến trúc Hybrid & Medallion.
2. `System architecture/01_mlops_pipeline.md` - Chi tiết LangGraph 4-node và các model (RoBERTa, Qwen, LightGBM).
3. `System architecture/02_admin_dashboard.md` - Hệ thống Dashboard (Metrics, Infra, VRAM Monitoring).
4. `System architecture/03_shared_infrastructure.md` - Core Utils (Logger, DB Session, Config, Broker).
5. `System architecture/04_operations.md` - Vận hành, Debugging và Deployment.
6. `metadata_implementation_plan.md` - (Gốc) Kế hoạch triển khai Medallion Data Architecture ban đầu.
