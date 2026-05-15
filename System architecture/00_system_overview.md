# 📋 Tài Liệu Nội Bộ — Hệ Thống C++ AI Code Detector

> **Trạng thái:** Bước 1/5 — Tổng quan hệ thống  
> **Phiên bản:** 1.0 · Cập nhật: 2026-05-11  
> **Dự án:** LVTN-main (`/home/nguyenvannhi242/LVTN-main`)

---

## 1. Mục Tiêu Hệ Thống

Hệ thống phát hiện mã nguồn C++ **do AI sinh ra** (AI-generated code detection). Đây là một hệ thống Backend/MLOps cấp enterprise với đầu vào/đầu ra như sau:

| | Mô tả | Ví dụ |
|---|---|---|
| **Input** | Code C++ thô (Base64-encoded) | `#include <iostream>\nint main() {...}` |
| **Output** | JSON: điểm AI Score + nhãn + XAI | `{ "final_pred": "AI GENERATED", "final_score": 0.8742, ... }` |

---

## 2. Kiến Trúc Hybrid Local-Cloud

Do mô hình AI (RoBERTa Ensemble + Qwen 7B) yêu cầu **14–16GB VRAM**, hệ thống chia thành 2 môi trường:

```
┌──────────────────────────────────────┐
│         LOCAL MACHINE (Laptop)       │
│                                      │
│  🌐 Django (Port 8000)               │  ← Giao diện, Auth, Admin
│  ⚡ FastAPI Orchestrator (Port 8001)  │  ← Điều phối pipeline AI
│  🔧 Local GPU Server (Port 8002)     │  ← Chạy model nếu có GPU local
│  🔄 Celery Worker                    │  ← Background ETL tasks
│  🐘 PostgreSQL (Port 5432)           │  ← Gold layer DB
│  ⚡ Redis (Port 6379)                │  ← Celery broker
│  📨 Redpanda (Port 9092)             │  ← Message streaming (optional)
└──────────────────────────────────────┘
                   │ HTTPS / ngrok tunnel
                   ▼
┌──────────────────────────────────────┐
│     GOOGLE COLAB (Cloud GPU)         │
│                                      │
│  ⚡ FastAPI AI Service               │  ← Chạy RoBERTa + Qwen 7B
│  🧠 Model Weights (từ Google Drive)  │  ← ~14GB VRAM (T4/L4)
└──────────────────────────────────────┘
```

> **Tại sao Hybrid?** Laptop chỉ có ~4GB VRAM → OOM nếu chạy trực tiếp.
> Colab cung cấp GPU T4/L4 miễn phí → tận dụng cho inference nặng.

---

## 3. Tech Stack Đầy Đủ

| Layer | Công nghệ | Version / Chi tiết | Lý do chọn |
|-------|-----------|-------------------|------------|
| **Web & Admin** | Django | 5.x | CRUD chuẩn ACID, Admin UI tự động, Session/Auth sẵn có |
| **AI Serving** | FastAPI + Pydantic v2 | Async | Non-blocking I/O cho inference nặng, validation tự động |
| **AI Runtime** | ONNX Runtime + PyTorch | — | ONNX cho production (2-5x faster), PyTorch cho dev/train |
| **LLM / Router** | LangGraph + Qwen 2.5 Coder 7B | — | Agentic workflow 4-node |
| **DL Model** | RoBERTa / GraphCodeBERT | Ensemble K-Fold | Phát hiện pattern AI trên token sequence |
| **ML Model** | LightGBM | — | 32 static features, nhanh, interpretable |
| **XAI** | SHAP + Layer Integrated Gradients (LIG) | — | Giải thích feature/token ảnh hưởng đến dự đoán |
| **Database** | PostgreSQL 16 | Port 5432 | ACID, JSON field cho Gold layer |
| **Broker** | Redis | Port 6379 | Celery broker + result backend (đang dùng) |
| **Streaming** | Redpanda | Port 9092 | Kafka-compatible, nhẹ hơn 10x RAM (optional) |
| **Background** | Celery | — | Async ETL, push to Data Lake |
| **Data Lake** | DagsHub S3-compatible | — | Offload Bronze/Silver, tích hợp DVC + MLflow |
| **Feature Store** | Parquet trên DagsHub S3 | — | Columnar format, tối ưu pandas/LightGBM |
| **Tunneling** | Cloudflare Tunnels (prod) / ngrok (Colab) | — | Public web production |
| **IaC** | Terraform + Ansible | — | Terraform cho K8s, Ansible cho bare-metal |
| **Monitoring** | Prometheus + Grafana + Loki | — | Metrics, Dashboard, Logs |
| **CI/CD** | GitHub Actions | — | Auto test + build + deploy |
| **Container** | Docker Compose (dev) / MicroK8s (prod) | — | K8s single-node với GPU addon |

---

## 4. Cấu Trúc Thư Mục Thực Tế

```
LVTN-main/
│
├── src/                              # Source code chính
│   ├── django_web/                   # Django Web Application
│   │   ├── apps/
│   │   │   ├── accounts/             # Authentication & User Profile
│   │   │   ├── submissions/          # Bronze Layer CRUD (BronzeSubmission model)
│   │   │   └── dashboard/            # Admin Dashboard + MLOps views
│   │   │       ├── views.py          # ← 6 views admin: overview, metrics, infra, db, users, models
│   │   │       └── urls.py           # ← URL routing
│   │   ├── templates/                # HTML templates (Jinja2)
│   │   ├── static/                   # CSS/JS
│   │   └── settings.py               # Django config (load từ .env)
│   │
│   ├── fastapi_service/              # FastAPI AI Serving Microservice
│   │   ├── main.py                   # Entry point, lifespan, middleware
│   │   ├── routers/
│   │   │   ├── health.py             # GET /health — runtime stats
│   │   │   └── predict.py            # POST /api/analyze_stream — SSE inference
│   │   ├── services/
│   │   │   └── agent_service.py      # ← LangGraph 4-node pipeline (QUAN TRỌNG)
│   │   ├── engine/
│   │   │   ├── roberta_engine.py     # RoBERTa Ensemble + LIG
│   │   │   ├── llm_handler.py        # Qwen 2.5 Coder (classify + perplexity + critique)
│   │   │   ├── fingerprint_engine.py # LightGBM + SHAP (32 features)
│   │   │   ├── heuristic_classifier.py # Fallback OOP/NORMAL detection
│   │   │   ├── explainer.py          # LIG attribution logic
│   │   │   └── model_manager.py      # Load/cache model weights
│   │   ├── schemas/
│   │   │   └── prediction_schema.py  # Pydantic: AnalyzeResponse, ChunkResult, SSEProgressEvent
│   │   └── core/                     # Config, DB session, dependencies
│   │
│   ├── local_gpu_server/             # Chạy model trên GPU local (Port 8002)
│   ├── celery_workers/               # Background tasks (ETL, S3 push)
│   │   ├── __init__.py               # Celery app config
│   │   └── tasks/                    # Task definitions
│   ├── data_pipeline/                # ETL Bronze→Silver→Gold
│   └── shared/                       # Shared utilities (dùng chung)
│       ├── config.py                 # Settings (threshold, paths, API keys)
│       ├── data_contracts.py         # GoldPredictionRecord dataclass
│       ├── database.py               # SQLAlchemy Async session
│       ├── logger.py                 # AppLogger (structured logging)
│       ├── hashing.py                # compute_code_hash (SHA-256)
│       ├── message_broker.py         # Redpanda/Redis messaging
│       └── s3_client.py              # DagsHub S3 client
│
├── data_lake/                        # Local Data Lake (sync lên DagsHub S3)
│   ├── bronze/                       # Raw code JSONL
│   ├── silver/
│   │   ├── ml/                       # LightGBM features (.parquet)
│   │   └── dl/                       # RoBERTa tokens (.parquet)
│   └── gold/                         # Exported predictions (.parquet)
│
├── infrastructure/                   # IaC
│   ├── ansible/                      # Ansible playbooks (NVIDIA driver, MicroK8s)
│   └── k8s_manifests/                # Kubernetes manifests (base + overlays)
│
├── monitoring/                       # Prometheus, Grafana, Loki configs
├── Saved_Models/                     # Model weights (local)
├── docker-compose.yml                # PostgreSQL + Redis + Redpanda (dev)
├── dev.sh                            # Script khởi động 1 lệnh (tmux 2×2)
├── Makefile                          # make dev shortcut
└── .env                              # Environment variables
```

---

## 5. Các Dịch Vụ & Cổng Port

| Service | Port | Khởi động | Mô tả |
|---------|------|-----------|-------|
| Django Web | **8000** | `python manage.py runserver 8000` | Frontend + Admin Dashboard |
| FastAPI Orchestrator | **8001** | `uvicorn src.fastapi_service.main:app --port 8001` | AI pipeline điều phối |
| Local GPU Server | **8002** | `uvicorn src.local_gpu_server.main:app --port 8002` | Chạy model trên GPU local |
| PostgreSQL | **5432** | `docker-compose up postgres` | Database chính |
| Redis | **6379** | `docker-compose up redis` | Celery broker |
| Redpanda | **9092** | `docker-compose up redpanda` | Message streaming |
| Celery Worker | — | `celery -A src.celery_workers.celery_app worker` | Background ETL |

### Khởi động nhanh (1 lệnh)

```bash
# Khởi động DB/Redis (Docker)
docker-compose up -d

# Khởi động tất cả service trong tmux 2×2
./dev.sh          # hoặc: make dev
```

`dev.sh` tự động tạo tmux session `lvtn_dev` với layout 4 pane:
- **Pane 0**: AI Orchestrator (8001)
- **Pane 1**: Django Web (8000)
- **Pane 2**: Celery Worker
- **Pane 3**: Local GPU Server (8002)

---

## 6. Kiến Trúc Dữ Liệu Medallion (Bronze → Silver → Gold)

### 🥉 Bronze — Dữ liệu thô

**Lưu ở:** PostgreSQL (`bronze_submissions` table) + DagsHub S3 (JSONL backup)

```python
class BronzeSubmission(models.Model):
    code_hash       # SHA-256 của raw_code (unique)
    user            # FK → auth_user
    raw_code        # Code C++ gốc (TEXT)
    language        # 'cpp'
    source          # 'pasted_code' | 'file_upload' | 'batch'
    file_size_bytes
    prediction      # '' (pending) | 'AI GENERATED' | 'HUMAN WRITTEN'
    confidence      # float 0-1 (nullable, sau khi có kết quả)
    timestamp
```

### 🥈 Silver — Dữ liệu đã xử lý

**Lưu ở:** DagsHub S3 (`.parquet`), managed by DVC

- **Silver-ML**: 32 features tĩnh từ `CppFeatureExtractorV8` (dùng cho LightGBM)
  - Style features: `avg_line_length`, `brace_style_consistency`, ...
  - Complexity: `avg_cyclomatic_complexity`, `halstead_volume`, ...
  - Entropy: `shannon_entropy`, `bigram_entropy`, `whitespace_entropy`
- **Silver-DL**: 512 token IDs (GraphCodeBERT tokenizer) + attention mask (dùng cho RoBERTa)

### 🥇 Gold — Kết quả dự đoán

**Lưu ở:** PostgreSQL (`gold_predictions` table) — ACID-compliant

```python
class GoldPrediction(Base):
    code_hash, user_id
    model_used      # 'C++ OOP Model' | 'C++ Normal Model'
    classification  # 'OOP' | 'NORMAL'
    prediction      # 'AI GENERATED' | 'HUMAN WRITTEN'
    confidence      # final_score (0-1)
    perplexity, max_ppl, burstiness   # Từ Qwen LLM
    is_ambiguous, retry_count
    global_critique                   # LLM Map-Reduce summary
    top_ai_signals, top_hu_signals    # JSON lists
    chunk_details                     # JSON breakdown per chunk
    inference_ms                      # Thời gian inference (ms)
```

---

## 7. Quyết Định Thiết Kế Quan Trọng

### ✅ Offload Data Lake lên DagsHub S3
- Bronze/Silver **KHÔNG** lưu trên local disk lâu dài
- Celery worker push lên DagsHub S3 sau mỗi submission
- Local chỉ giữ PostgreSQL (Gold) + Model weights
- **Lợi ích:** tiết kiệm 60-80% storage, có DVC data versioning + MLflow experiment tracking miễn phí

### ✅ Redis thay Redpanda làm Celery broker (thực tế hiện tại)
- Redpanda (Kafka-compatible) trong `docker-compose.yml` nhưng Celery đang dùng Redis
- Redpanda được giữ lại cho streaming events (`code.submitted`, `prediction.completed`)

### ✅ SSE Streaming thay vì Polling
- FastAPI dùng Server-Sent Events (SSE) để stream progress realtime về Django
- Progress: 5% → 20% → 65% → 75% → 95% → 100%

---

## 8. Output JSON Schema (từ API)

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
  "fingerprint": { ... }  // LightGBM + SHAP XAI result
}
```

---

## Bước Tiếp Theo

| Bước | Nội dung | Trạng thái |
|------|----------|------------|
| **Bước 1** | Tổng quan hệ thống, Tech Stack, Medallion Architecture | ✅ Hoàn thành |
| **Bước 2** | MLOps Pipeline chi tiết: LangGraph 4-node, RoBERTa, LightGBM, XAI | 🔲 Chờ review |
| **Bước 3** | Admin Dashboard: 6 views, metrics API, infra healthcheck | 🔲 Chờ review |
| **Bước 4** | Shared infrastructure: Logger, Config, Data Contracts, S3, Celery | 🔲 Chờ review |
| **Bước 5** | Vận hành: khởi động, debugging, monitoring, CI/CD | 🔲 Chờ review |
