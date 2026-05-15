# 🚀 Tài Liệu Nội Bộ — Bước 5: Vận Hành

> **Phạm vi:** Khởi động, cấu hình môi trường, monitoring, debugging, và CI/CD  
> **Files chính:** `dev.sh`, `Makefile`, `docker-compose.yml`, `.env`, `Run_And_Setup.md`

---

## 1. Hai Chế Độ Triển Khai

Hệ thống hỗ trợ 2 chế độ tùy theo phần cứng:

### Chế Độ A — Colab GPU Worker (Máy không GPU)

```
Laptop (CPU only)                    Google Colab (T4/L4 GPU)
─────────────────                    ────────────────────────
Django  :8000  ◄────────────────────── User
FastAPI :8001  ──── POST /api/predict/roberta ──►  Colab GPU Worker
               ◄── HTTP via ngrok tunnel ──────   • RoBERTa 5-fold
               ──── /api/proxy/vllm ──────────►  • vLLM Qwen 2.5 7B
Celery Worker                                    • LightGBM (on Colab)
PostgreSQL :5432
Redis :6379
```

**Cấu hình `.env`:**
```ini
FASTAPI_AI_URL=https://xxxx-xxxx-xxxx.ngrok-free.app   # URL ngrok Colab
```

---

### Chế Độ B — Local GPU Server (Máy có GPU Nvidia)

```
Laptop (CPU + GPU)
─────────────────────────────────────────────────────
Django      :8000   User interface
FastAPI     :8001   Orchestrator (LangGraph, LightGBM)
Local GPU   :8002   RoBERTa inference (torch, 4GB VRAM limit)

FastAPI :8001 ──── POST /api/predict/roberta ──► GPU Server :8002
              ──── GET  /gpu-stats           ──► GPU Server :8002
              ──── GET  /health              ──► GPU Server :8002
```

**Cấu hình `.env`:**
```ini
FASTAPI_AI_URL=http://localhost:8002          # Local GPU Server

LOCAL_MODEL_PATH_OOP=/path/to/fold_1_oop     # RoBERTa OOP weights
LOCAL_MODEL_PATH_NORMAL=/path/to/fold_1_basic # RoBERTa Normal weights
LOCAL_SAVED_MODELS_PATH=/path/to/saved_models # LightGBM PKL files
LOCAL_GPU_API_KEY=local-gpu-secret-key
LOCAL_GPU_PORT=8002
API_KEY=local-gpu-secret-key                  # Phải khớp với LOCAL_GPU_API_KEY
```

> **Lưu ý VRAM:** Local GPU Server tự động giới hạn PyTorch ở 4GB:  
> `torch.cuda.set_per_process_memory_fraction(4.0 / total_vram, 0)`

---

## 2. Cấu Hình `.env` — Checklist Trước Khi Chạy

File `.env` ở project root — được load bởi `shared/config.py` khi import.

```ini
# ==== BẮT BUỘC (Critical) =====================================

FASTAPI_AI_URL=http://localhost:8002          # ← Thay bằng ngrok nếu dùng Colab

DB_NAME=cpp_detector
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

SECRET_KEY=<random-secret-64-chars>           # ← ĐỔI trước khi dùng thực tế!

# ==== PHỤ THUỘC CHẾ ĐỘ ========================================

# Chế độ Local GPU:
LOCAL_MODEL_PATH_OOP=/mnt/c/.../fold_1_oop
LOCAL_MODEL_PATH_NORMAL=/mnt/c/.../fold_1_basic
LOCAL_SAVED_MODELS_PATH=./local_models/saved_models
LOCAL_GPU_API_KEY=local-gpu-secret-key
API_KEY=local-gpu-secret-key                  # Phải trùng với LOCAL_GPU_API_KEY

# ==== TÙY CHỌN (Optional) =====================================

CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# LLM Fallback (bỏ trống nếu không có):
GEMINI_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...

# DagsHub S3 (để trống → tự động dùng local filesystem):
AWS_ACCESS_KEY_ID=                            # ← Để trống = local mode
AWS_SECRET_ACCESS_KEY=

# Monitoring (optional):
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

---

## 3. Quy Trình Khởi Động Hệ Thống

### Bước 0 — Infrastructure (Docker)

```bash
# Khởi động DB, Redis, Redpanda
docker compose up -d

# Kiểm tra
docker compose ps
# postgres  → healthy
# redis     → healthy
# redpanda  → healthy (optional)
```

### Bước 1 — Django Migration (lần đầu)

```bash
cd /home/nguyenvannhi242/LVTN-main
conda activate detection_ai

python manage.py migrate
python manage.py createsuperuser   # username: admin, is_staff=True
```

### Bước 2 — Khởi Động Tất Cả Services (1 Lệnh)

```bash
make dev
# hoặc trực tiếp:
./dev.sh
```

Script `dev.sh` tạo tmux session `lvtn_dev` với layout 2×2:

```
┌─────────────────────────┬─────────────────────────┐
│  Pane 0                 │  Pane 1                 │
│  AI Orchestrator        │  Web Frontend           │
│  uvicorn :8001          │  manage.py :8000        │
├─────────────────────────┼─────────────────────────┤
│  Pane 2                 │  Pane 3                 │
│  Celery Worker          │  Local GPU Worker       │
│  celery worker          │  uvicorn :8002          │
└─────────────────────────┴─────────────────────────┘
```

Mỗi pane tự động:
1. `source ~/anaconda3/etc/profile.d/conda.sh`
2. `conda activate detection_ai`
3. Chạy lệnh service

### Bước 3 — (Chế Độ Colab) Cập Nhật ngrok URL

```bash
# Sau khi Colab in ra URL:
# https://xxxx-xxxx-xxxx.ngrok-free.app

# Cập nhật .env:
FASTAPI_AI_URL=https://xxxx-xxxx-xxxx.ngrok-free.app

# Restart FastAPI Orchestrator:
# Trong tmux Pane 0: Ctrl+C → Up arrow → Enter
```

### Bước 4 — Xác Nhận Hệ Thống Hoạt Động

```bash
# FastAPI Orchestrator
curl http://localhost:8001/health

# Local GPU Server
curl http://localhost:8002/health

# Django (trên browser)
open http://localhost:8000/accounts/login/
```

---

## 4. Makefile Commands

```bash
make dev     # Khởi động toàn bộ (dev.sh + tmux)
make stop    # Dừng toàn bộ services + kill tmux session
make logs    # Re-attach vào tmux session lvtn_dev
```

`make stop` thực hiện:
```bash
tmux kill-session -t lvtn_dev
pkill -f "uvicorn src.fastapi_service.main:app"
pkill -f "python manage.py runserver"
pkill -f "celery -A src.celery_workers.celery_app"
pkill -f "uvicorn src.local_gpu_server.main:app"
```

---

## 5. Local GPU Server — Chi Tiết

**File:** `src/local_gpu_server/main.py`  
**Port:** 8002 (mặc định, đổi qua `LOCAL_GPU_PORT`)

### Startup Sequence

```python
# 1. Giới hạn VRAM 4GB ngay khi import
if torch.cuda.is_available():
    fraction = 4.0 / total_vram
    torch.cuda.set_per_process_memory_fraction(fraction, 0)

# 2. Khởi tạo LocalModelManager (load RoBERTa weights)
_manager = LocalModelManager()

# 3. Tạo FastAPI app
app = create_app(_manager)
```

### Các Endpoints

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `GET /` | GET | Health check đơn giản |
| `GET /health` | GET | Status, model loaded, VRAM |
| `GET /gpu-stats` | GET | VRAM metrics chi tiết |
| `POST /api/predict/roberta` | POST | RoBERTa inference |

### VRAM-Tuned Hyperparameters (vs Colab)

| Parameter | Local GPU (4GB) | Colab (16GB) |
|-----------|----------------|-------------|
| `LIG_N_STEPS` | 10 | 20 |
| `LIG_BATCH_SIZE` | 1 | 4 |
| `ENGINE_BATCH_SIZE` | 1 | 4 |
| `MAX_CACHE_SIZE` | 50 | 200 |
| Folds | 1 (tiết kiệm VRAM) | 5 |

### Model Artifacts Layout

```
local_models/
  fold_1_oop/          ← RoBERTa OOP weights (HuggingFace format)
    config.json
    pytorch_model.bin  (hoặc model.safetensors)
    tokenizer_config.json
    vocab.json
    merges.txt
  fold_1_basic/        ← RoBERTa Normal weights
    (tương tự)
  saved_models/        ← LightGBM + SHAP artifacts
    LightGBM_Regulated.pkl
    scaler.pkl
    final_features.pkl
    baselines.json
```

---

## 6. FastAPI Orchestrator — Startup Sequence

**File:** `src/fastapi_service/main.py`  
Thực hiện theo thứ tự khi uvicorn khởi động:

```
1. logger.rotate_on_startup()
   └── current_run.log.json → previous_run.log.json
   └── Tạo current_run.log.json trống mới

2. await init_async_db()
   └── CREATE TABLE IF NOT EXISTS (SQLAlchemy models)
   └── Non-fatal nếu PostgreSQL chưa sẵn sàng

3. agent.ensure_models_loaded()
   └── Kiểm tra kết nối tới FASTAPI_AI_URL (GPU Backend)
   └── Non-fatal — service vẫn start nếu GPU backend offline

4. READY — nhận requests
```

**Middleware tự động log mọi HTTP request:**
```
POST /api/analyze_stream → 200  [latency: 4521ms]
GET  /health             → 200  [latency: 12ms]
```

**Global exception handler:** Bất kỳ unhandled exception nào → `500 Internal Server Error` + log đầy đủ stack trace.

---

## 7. Health Monitoring

### Endpoint `/health` (FastAPI Orchestrator :8001)

```json
{
  "status": "ok",
  "models_loaded": true,
  "gpu_available": true,
  "version": "1.0.0",
  "request_count": 47,
  "cache_size": 12,
  "last_latency_ms": 4320.5,
  "p95_latency_ms": 5100.2,
  "uptime_seconds": 3621.0,
  "gpu_vram_used_gb": 3.2,
  "gpu_vram_total_gb": 4.0,
  "gpu_vram_pct": 80.0
}
```

**Cách tính P95 latency:**
```python
# Giữ deque(maxlen=200) — 200 request gần nhất
# P95: lấy phần tử ở vị trí 95% sau khi sort
sorted_v = sorted(latency_history)
idx = max(0, int(len(sorted_v) * 0.95) - 1)
p95 = sorted_v[idx]
```

**VRAM Stats Priority:**
```
1. GET http://localhost:8002/gpu-stats   (Local GPU Server — chính xác nhất)
2. torch.cuda.memory_reserved()         (Process hiện tại — fallback)
3. None                                 (Nếu không có GPU)
```

### Kiểm Tra Thủ Công Các Services

```bash
# PostgreSQL
psql -U postgres -d cpp_detector -c "SELECT COUNT(*) FROM submissions_bronzesubmission;"

# Redis
redis-cli ping
# → PONG

# Celery
celery -A src.celery_workers.celery_app inspect active

# Redpanda
curl http://localhost:18081/topics

# VRAM Local GPU Server
curl http://localhost:8002/gpu-stats | python3 -m json.tool
```

---

## 8. Debug Với Log Files

Log files ở `logs/`:

```
logs/
  current_run.log.json    ← Phiên đang chạy
  previous_run.log.json   ← Phiên trước
  local_gpu_server/
    current.jsonl         ← GPU server log
    previous.jsonl
```

### Đọc Log Bằng `jq`

```bash
# Xem tất cả ERROR trong phiên hiện tại
cat logs/current_run.log.json | jq 'select(.level == "ERROR")'

# Xem latency của tất cả roberta_engine calls
cat logs/current_run.log.json | jq 'select(.module == "roberta_engine") | {function, latency_ms}'

# Xem các request chậm hơn 5 giây
cat logs/current_run.log.json | jq 'select(.latency_ms > 5000) | {module, function, latency_ms}'

# Theo dõi realtime
tail -f logs/current_run.log.json | jq '.'

# Xem 10 log gần nhất
tail -n 10 logs/current_run.log.json | jq '{level, module, function, message}'
```

### Debug Khi Submit Code Bị Treo

```bash
# 1. Kiểm tra FastAPI đang nhận request không
tail -f logs/current_run.log.json | jq 'select(.function == "log_requests")'

# 2. Kiểm tra node nào trong pipeline đang xử lý
tail -f logs/current_run.log.json | jq 'select(.module == "agent_service")'

# 3. Kiểm tra GPU server có respond không
curl -w "@-" -o /dev/null -s http://localhost:8002/health <<< "time_total: %{time_total}s\n"
```

---

## 9. Các Lỗi Thường Gặp & Cách Sửa

| Triệu chứng | Nguyên nhân | Cách sửa |
|-------------|------------|---------|
| `bert_score = 0.5`, `total_tokens = 0` | GPU backend không phản hồi | Kiểm tra `FASTAPI_AI_URL`, restart GPU server hoặc Colab |
| `perplexity = 0.0` | vLLM chưa load xong | Chờ 2-3 phút sau khi Colab start, kiểm tra `/content/vllm.log` |
| `Connection refused :8001` | FastAPI Orchestrator chưa start | `uvicorn src.fastapi_service.main:app --port 8001` |
| `Connection refused :8002` | Local GPU Server chưa start | Pane 3 trong tmux: kiểm tra log khởi động |
| `CUDA out of memory` | VRAM > 4GB | Giảm `LIG_N_STEPS` xuống 5, restart GPU server |
| `FingerprintEngine: Failed to load artifacts` | Thiếu `.pkl` files | Kiểm tra `local_models/saved_models/` có đủ 4 files |
| `django.db.OperationalError` | PostgreSQL chưa chạy | `docker compose up -d postgres` |
| `Redis connection refused` | Redis chưa chạy | `docker compose up -d redis` |
| Port đã bị dùng (8000/8001) | Tiến trình cũ chưa kill | `make stop` rồi `make dev` |
| Ngrok URL expired | Session Colab 12 giờ | Chạy lại `bootstrap.py`, copy URL mới → cập nhật `.env` |
| `403 Forbidden` trên `/dashboard/` | User không phải staff | `python manage.py shell` → `User.objects.filter(username="x").update(is_staff=True)` |
| Log file rất lớn | Không rotate | `rotate_on_startup()` chỉ được gọi khi restart service |

---

## 10. Cấu Trúc Thư Mục Toàn Dự Án

```
LVTN-main/
├── .env                        ← ⚙️ Cấu hình toàn hệ thống
├── docker-compose.yml          ← PostgreSQL + Redis + Redpanda
├── dev.sh                      ← 🚀 One-command startup (tmux)
├── Makefile                    ← make dev | stop | logs
├── manage.py                   ← Django CLI
├── requirements.txt            ← Python dependencies
│
├── src/
│   ├── django_web/             ← Web frontend + Admin dashboard
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── apps/
│   │   │   ├── accounts/       ← Auth: login, register, profile
│   │   │   ├── submissions/    ← Submit, result, history, batch
│   │   │   └── dashboard/      ← 6 admin views + metrics API
│   │   └── templates/          ← HTML templates (Tailwind)
│   │
│   ├── fastapi_service/        ← AI Orchestrator (port 8001)
│   │   ├── main.py             ← Entry point + lifespan
│   │   ├── services/
│   │   │   └── agent_service.py ← LangGraph 5-node pipeline
│   │   ├── engine/
│   │   │   ├── roberta_engine.py    ← HTTP client → GPU backend
│   │   │   ├── fingerprint_engine.py ← LightGBM + SHAP (local)
│   │   │   ├── llm_handler.py       ← vLLM→Gemini→OpenAI fallback
│   │   │   ├── heuristic_classifier.py ← OOP/Normal regex
│   │   │   ├── explainer.py         ← LIG attribution (on GPU)
│   │   │   └── feature_extractor/
│   │   │       └── extractor.py     ← CppFeatureExtractorV8 (32 features)
│   │   ├── routers/
│   │   │   ├── health.py       ← GET /health, GET /
│   │   │   └── predict.py      ← POST /api/analyze, /api/analyze_stream
│   │   └── schemas/
│   │       └── prediction_schema.py ← Pydantic request/response
│   │
│   ├── local_gpu_server/       ← Local GPU inference (port 8002)
│   │   ├── main.py             ← VRAM limit + app factory
│   │   ├── config.py           ← Paths, hyperparams, API key
│   │   ├── engine.py           ← LocalModelManager (load weights)
│   │   └── server.py           ← FastAPI routes
│   │
│   ├── celery_workers/         ← Background ETL tasks
│   │   ├── __init__.py         ← Celery app factory
│   │   └── tasks/
│   │       ├── bronze_tasks.py  ← push_bronze_to_s3
│   │       ├── silver_tasks.py  ← extract_and_push_silver
│   │       ├── gold_tasks.py    ← log_prediction_to_gold
│   │       └── retraining_tasks.py ← check_retrain_trigger (daily)
│   │
│   └── shared/                 ← Shared utilities (no external deps)
│       ├── config.py           ← Settings singleton
│       ├── logger.py           ← AppLogger + @log_function
│       ├── hashing.py          ← compute_code_hash (SHA-256)
│       ├── database.py         ← Async + Sync SQLAlchemy engines
│       ├── s3_client.py        ← DagsHub S3 + local fallback
│       ├── message_broker.py   ← Redpanda producer/consumer
│       └── data_contracts.py   ← Pydantic schemas (single source of truth)
│
├── local_models/               ← Model artifacts (không commit Git)
│   ├── fold_1_oop/             ← RoBERTa OOP weights
│   ├── fold_1_basic/           ← RoBERTa Normal weights
│   └── saved_models/           ← LightGBM PKL files
│       ├── LightGBM_Regulated.pkl
│       ├── scaler.pkl
│       ├── final_features.pkl
│       └── baselines.json
│
├── data_lake/                  ← Local S3 fallback (auto-created)
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── logs/                       ← JSONL logs (auto-created)
│   ├── current_run.log.json
│   └── previous_run.log.json
│
└── colab_runtime/              ← Google Colab GPU Worker code
    └── scripts/
        ├── bootstrap.py        ← One-command Colab setup
        ├── engine.py           ← ModelManager (RoBERTa 5-fold)
        └── server.py           ← FastAPI Colab Worker
```

---

## 11. Quick Reference — Lệnh Hay Dùng Nhất

```bash
# ══ KHỞI ĐỘNG ═══════════════════════════════════════════════════
docker compose up -d               # PostgreSQL + Redis
make dev                           # Tất cả 4 services (tmux)
make stop                          # Dừng tất cả

# ══ DJANGO ══════════════════════════════════════════════════════
python manage.py migrate           # Chạy migration
python manage.py createsuperuser   # Tạo admin
python manage.py shell             # Interactive shell

# ══ DEBUG NHANH ══════════════════════════════════════════════════
curl http://localhost:8001/health | python3 -m json.tool
curl http://localhost:8002/gpu-stats | python3 -m json.tool
tail -f logs/current_run.log.json | jq '{level,module,message,latency_ms}'

# ══ DATABASE ═════════════════════════════════════════════════════
psql -U postgres -d cpp_detector
  \dt                              # List tables
  SELECT COUNT(*) FROM submissions_bronzesubmission;
  SELECT prediction, COUNT(*) FROM submissions_bronzesubmission GROUP BY 1;

# ══ CELERY ═══════════════════════════════════════════════════════
celery -A src.celery_workers inspect active
celery -A src.celery_workers inspect stats

# ══ TMUX ═════════════════════════════════════════════════════════
tmux attach -t lvtn_dev            # Re-attach
Ctrl+B → số 0/1/2/3               # Chuyển pane
Ctrl+B → D                        # Detach (giữ services chạy)
```

---

## 12. Tổng Kết — 5 Bước Tài Liệu

| Bước | Tài liệu | Nội dung chính |
|------|---------|---------------|
| **1** | `00_system_overview.md` | Kiến trúc Hybrid, Tech Stack, Medallion Architecture |
| **2** | `01_mlops_pipeline.md` | LangGraph 5-node, RoBERTa, LightGBM+SHAP, Self-correction |
| **3** | `02_admin_dashboard.md` | 6 Admin views, SSE proxy, Submit flow, BronzeRepository |
| **4** | `03_shared_infrastructure.md` | Logger, Config, Hashing, DB, S3, Celery ETL |
| **5** | `04_operations.md` | Khởi động, .env, Local GPU, Monitoring, Debug, Troubleshoot |
