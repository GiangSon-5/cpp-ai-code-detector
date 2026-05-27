# 🔧 Tài Liệu Nội Bộ — Bước 4: Shared Infrastructure

> **Phạm vi:** `src/shared/` + `src/celery_workers/`  
> **Vai trò:** Lớp nền tảng dùng chung cho Django, FastAPI, và Celery

---

## 1. Sơ Đồ Phụ Thuộc Shared Modules

```
┌─────────────────────────────────────────────────┐
│                src/shared/                       │
│                                                  │
│  config.py ──────────────────────────────────┐  │
│     ↓ (settings singleton)                   │  │
│  logger.py  ← tất cả module khác dùng       │  │
│     ↓ (AppLogger singleton)                  │  │
│  ┌──────────────────────────────────────┐    │  │
│  │  database.py   hashing.py  s3_client.py │  │  │
│  │  message_broker.py  data_contracts.py  │  │  │
│  └──────────────────────────────────────┘    │  │
│                    ↑                         │  │
│        (import từ Django / FastAPI / Celery) │  │
└─────────────────────────────────────────────────┘
```

**Quy tắc vàng:** Mọi module trong `shared/` chỉ import lẫn nhau (không import từ `django_web`, `fastapi_service`, hay `celery_workers`).

---

## 2. `config.py` — Cấu Hình Trung Tâm

**File:** `src/shared/config.py`  
**Pattern:** Frozen dataclass singleton → `settings = Settings()`

### Cách hoạt động
```python
# Tìm project root (folder chứa .env)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env một lần duy nhất khi import
load_dotenv(_ENV_PATH, override=False)

# Frozen dataclass → immutable sau khi khởi tạo
@dataclass(frozen=True)
class Settings: ...

settings = Settings()  # Singleton, import từ mọi nơi
```

### Toàn bộ cấu hình

| Nhóm | Key | Mặc định | Ghi chú |
|------|-----|---------|---------|
| **Đường dẫn** | `PROJECT_ROOT` | Auto-detect | Từ vị trí `config.py` |
| | `LOG_DIR` | `PROJECT_ROOT/logs` | JSONL log files |
| **Django** | `DEBUG` | `True` | từ env `DEBUG` |
| | `SECRET_KEY` | `"django-insecure-..."` | **Phải đổi khi production** |
| | `ALLOWED_HOSTS` | `["localhost"]` | Comma-separated trong env |
| **AI Service** | `FASTAPI_AI_URL` | `http://localhost:8000` | URL Colab ngrok hoặc `:8002` |
| **PostgreSQL** | `DB_NAME` | `cpp_detector` | |
| | `DB_USER/PASSWORD` | `postgres/postgres` | |
| | `DB_HOST/PORT` | `localhost:5432` | |
| | `DATABASE_URL_ASYNC` | `postgresql+asyncpg://...` | Property cho FastAPI |
| | `DATABASE_URL_SYNC` | `postgresql+psycopg2://...` | Property cho Celery |
| **Broker** | `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Redpanda |
| | `REDPANDA_API_URL` | `http://localhost:18081` | REST API |
| | `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery qua Redis |
| | `CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | |
| **DagsHub S3** | `S3_ENDPOINT_URL` | `""` | DagsHub S3-compatible URL |
| | `AWS_ACCESS_KEY_ID` | `""` | DagsHub access key |
| | `AWS_SECRET_ACCESS_KEY` | `""` | DagsHub secret |
| | `DAGSHUB_TOKEN` | `""` | Cho DVC/MLflow |
| **AI APIs** | `GEMINI_API_KEY` | `""` | LLM fallback #2 |
| | `OPENAI_API_KEY` | `""` | LLM fallback #3 |
| | `NGROK_TOKEN` | `""` | Colab tunnel |
| **Monitoring** | `PROMETHEUS_PORT` | `9090` | |
| | `GRAFANA_PORT` | `3000` | |
| | `LOKI_URL` | `http://localhost:3100` | |
| **AI Thresholds** | `DEFAULT_THRESHOLD` | `0.5` | Ngưỡng AI vs Human |
| | `AMBIGUOUS_LOW` | `0.40` | Vùng nhập nhằng dưới |
| | `AMBIGUOUS_HIGH` | `0.60` | Vùng nhập nhằng trên |
| | `FUSION_ALPHA` | `0.48` | α·BERT + (1-α)·LightGBM |

> **Quan trọng:** `FASTAPI_AI_URL` trỏ đến GPU backend (Colab ngrok hoặc Local GPU Server `:8002`). FastAPI Orchestrator (:8001) đọc env này để biết nơi gọi inference.

---

## 3. `logger.py` — Deep Logging JSONL

**File:** `src/shared/logger.py`  
**Pattern:** Singleton + 2-session file rotation

### Thiết kế

```
logs/
  current_run.log.json     ← phiên đang chạy (append)
  previous_run.log.json    ← phiên trước (chỉ đọc)
```

**Khi khởi động app** (`rotate_on_startup()` được gọi trong FastAPI `lifespan`):
1. Xóa `previous_run.log.json` (nếu tồn tại)
2. Rename `current_run.log.json` → `previous_run.log.json`
3. Tạo file `current_run.log.json` trống mới

**Tại sao chỉ giữ 2 file?**
- Tránh log không giới hạn chiếm disk
- Luôn có log phiên trước để debug so sánh
- Phù hợp môi trường laptop phát triển

### Cấu Trúc Mỗi Log Entry (JSONL)

```json
{
  "timestamp": "2026-05-12T00:01:23.456789+00:00",
  "level": "INFO",
  "module": "roberta_engine",
  "function": "analyze",
  "message": "Calling GPU worker at http://localhost:8002/api/predict/roberta",
  "input": {"args": [], "kwargs": {"code": "...", "model_type": "OOP"}},
  "output": {"mean_score": 0.8742, "total_chunks": 3},
  "error": null,
  "latency_ms": 4521.234
}
```

### Public API

```python
logger = AppLogger()          # Singleton — tất cả module dùng chung 1 instance

logger.info(module="...", function="...", message="...", input_data={}, output_data={}, latency_ms=0.0)
logger.warning(...)
logger.error(module="...", function="...", error="stack trace...")
logger.debug(...)
```

### `@AppLogger.log_function()` Decorator

Decorator này tự động wrap **bất kỳ hàm nào** (sync hoặc async) để log input, output, error, latency:

```python
@AppLogger.log_function(module="fingerprint_engine")
def analyze(self, code: str) -> FingerprintResult | None:
    # Mọi exception sẽ được log và re-raise
    # Mọi return value sẽ được log
    # Latency được đo tự động
    ...
```

**Dùng ở đâu:**
- `RoBERTaEngine.analyze()`
- `FingerprintEngine.analyze()`
- `heuristic_classify()`
- `LLMHandler.classify_code()`, `generate_critique()`, v.v.
- Tất cả hàm trong `s3_client.py`
- Tất cả hàm trong `message_broker.py`

**Thư viện:** `orjson` (3-5x nhanh hơn `json` stdlib) để serialize JSON.

---

## 4. `hashing.py` — Khóa Liên Kết Toàn Hệ Thống

**File:** `src/shared/hashing.py`

```python
def compute_code_hash(raw_code: str) -> str:
    normalised = raw_code.strip().replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    # → 64-char hex string
```

### Tại sao `code_hash` là primary key?

`code_hash` là **dây liên kết duy nhất** xuyên suốt 3 tầng Medallion:

```
code_hash = SHA-256(normalize(raw_code))
     │
     ├── BronzeSubmission.code_hash          (PostgreSQL — Django ORM)
     ├── bronze/{YYYY}/{MM}/{DD}/{hash}.jsonl (DagsHub S3)
     ├── silver/{YYYY}/{MM}/{DD}/{hash}.json  (DagsHub S3)
     ├── gold/{YYYY}/{MM}/{DD}/{hash}.parquet (DagsHub S3)
     └── GoldPrediction.code_hash            (PostgreSQL — SQLAlchemy)
```

**Chuẩn hóa trước khi hash:**
- Trim whitespace thừa đầu/cuối
- Chuẩn hóa `\r\n` → `\n` (cross-platform)
- Đảm bảo cùng code nộp trên Windows hay Linux → cùng hash

---

## 5. `database.py` — Hai Engine Song Song

**File:** `src/shared/database.py`

Hệ thống dùng **2 engine PostgreSQL song song** cho 2 mục đích khác nhau:

| | Async Engine | Sync Engine |
|--|-------------|-------------|
| **Driver** | `asyncpg` (psycopg3 async) | `psycopg2` |
| **URL prefix** | `postgresql+asyncpg://` | `postgresql+psycopg2://` |
| **Dùng cho** | FastAPI (async/await) | Celery workers (blocking) |
| **Pool size** | `10` connections + `20` overflow | `5` + `10` overflow |
| **Session** | `AsyncSessionFactory` | `SyncSessionFactory` |
| **Factory** | `async_sessionmaker` | `sessionmaker` |

### FastAPI — Dependency injection

```python
# Trong FastAPI route:
async def my_route(session: AsyncSession = Depends(get_async_session)):
    # session tự commit khi route trả về
    # session tự rollback khi có exception
    ...
```

### Celery — Context manager

```python
# Trong Celery task:
session = get_sync_session()
try:
    session.add(row)
    session.commit()
finally:
    session.close()
```

> **Lưu ý quan trọng:** Django **không dùng** `database.py`. Django có ORM riêng với cấu hình trong `django_web/settings.py`. `database.py` chỉ dành cho **FastAPI** (SQLAlchemy models) và **Celery** (gold_tasks write vào GoldPrediction table).

---

## 6. `s3_client.py` — DagsHub S3 Data Lake

**File:** `src/shared/s3_client.py`  
**Thư viện:** `boto3` với endpoint S3-compatible của DagsHub

### Local Mode vs S3 Mode

```python
def _is_local_mode() -> bool:
    key = settings.AWS_ACCESS_KEY_ID
    return not key or key == "your_access_key" or "dummy" in key
```

| Mode | Điều kiện | Lưu tại |
|------|-----------|---------|
| **Local** | `AWS_ACCESS_KEY_ID` rỗng hoặc `"dummy"` | `PROJECT_ROOT/data_lake/{key}` |
| **S3** | Có credentials hợp lệ | DagsHub S3 `s3://data-lake/{key}` |

> **Mặc định khi dev:** Luôn chạy Local Mode → lưu file vào `data_lake/` trong project.  
> `DagsHubS3Client.upload_bytes()` **luôn lưu local TRƯỚC** rồi mới thử upload S3.

### S3 Key Conventions

```
data-lake/
  bronze/{YYYY}/{MM}/{DD}/{code_hash}.jsonl     ← raw code + metadata
  silver/{YYYY}/{MM}/{DD}/{code_hash}.json      ← features (ML + DL tokens)
  gold/{YYYY}/{MM}/{DD}/{code_hash}.parquet     ← prediction result
```

### Helper Functions

| Hàm | Mô tả | Trả về |
|-----|-------|-------|
| `upload_bronze_jsonl(code_hash, record)` | Upload Bronze record dạng JSONL | S3 key string |
| `upload_silver_ml_parquet(code_hash, df)` | Upload DataFrame 44 features | S3 key string |
| `upload_silver_dl_parquet(code_hash, df)` | Upload DataFrame token IDs | S3 key string |
| `upload_gold_parquet(code_hash, df)` | Upload GoldPrediction record | S3 key string |
| `download_parquet(prefix, code_hash)` | Download và parse Parquet → DataFrame | `pd.DataFrame` |
| `list_objects(prefix, max_keys=1000)` | List keys dưới prefix | `list[str]` |

---

## 7. `message_broker.py` — Redpanda/Kafka Event Bus

**File:** `src/shared/message_broker.py`  
**Thư viện:** `confluent-kafka`

### 3 Topics

| Topic | Khi nào publish | Ai publish | Ai consume |
|-------|----------------|------------|------------|
| `code.submitted` | Sau khi save Bronze | Django/FastAPI | Celery Silver task |
| `prediction.completed` | Sau khi inference xong | FastAPI | Celery Gold task |
| `retrain.trigger` | Khi đủ sample mới | Celery Beat | Celery retrain pipeline |

### EventProducer — Singleton

```python
producer = EventProducer()
producer.publish(
    topic=Topics.CODE_SUBMITTED,
    value={"code_hash": "abc123", "user_id": 1, "source": "web_upload"},
    key="abc123"  # Kafka partition key → same hash → same partition
)
```

### EventConsumer — Blocking loop

```python
consumer = EventConsumer(group_id="silver-worker", topics=[Topics.CODE_SUBMITTED])
consumer.consume_loop(handler=my_handler_fn)  # Blocking
consumer.stop()  # Từ thread khác
```

> **Trạng thái hiện tại:** Redpanda đang là **Optional** (status "warning" trong infra dashboard). Hệ thống không phụ thuộc broker để hoạt động — Celery task được gọi trực tiếp từ Django views thay vì qua event bus.

---

## 8. Celery Workers — Background ETL Pipeline

**App:** `src/celery_workers/__init__.py`

### Cấu hình

```python
celery_app = Celery("cpp_detector",
    broker=CELERY_BROKER_URL,      # redis://localhost:6379/0 (default)
    backend=CELERY_RESULT_BACKEND, # redis://localhost:6379/0
    include=[
        "src.celery_workers.tasks.bronze_tasks",
        "src.celery_workers.tasks.silver_tasks",
        "src.celery_workers.tasks.gold_tasks",
        "src.celery_workers.tasks.retraining_tasks",
    ]
)

# Timezone: Asia/Ho_Chi_Minh
# Retry delay mặc định: 10 giây
# Max retry: 3 lần
```

**Khởi động:**
```bash
celery -A src.celery_workers worker --loglevel=info
celery -A src.celery_workers beat --loglevel=info   # Cho periodic tasks
```

### 4 Task Types

#### Task 1: `push_bronze_to_s3`
```
Trigger: Ngay sau khi user submit code
Input:   code_hash, raw_code, metadata{user_id, source, ...}
Action:  Build JSON payload → DagsHubS3Client.upload_bytes()
S3 Key:  bronze/{YYYY}/{MM}/{DD}/{code_hash}.jsonl
Retry:   3 lần, delay 10s
```

#### Task 2: `extract_and_push_silver`
```
Trigger: Sau Bronze task thành công
Input:   code_hash, raw_code, label (optional)
Action:  CppFeatureExtractorV8.extract() → 44 ML features (gồm 33 features cơ bản và 11 features OOP/style nâng cao)
         AutoTokenizer(graphcodebert) → 512 DL token IDs
         Upload JSON → silver/{YYYY}/{MM}/{DD}/{code_hash}.json
Retry:   3 lần, delay 15s
Note:    DL tokenization là non-fatal — nếu lỗi vẫn tiếp tục với dl_token_ids=[]
```

#### Task 3: `log_prediction_to_gold`
```
Trigger: Sau inference hoàn thành (FastAPI gọi .delay())
Input:   prediction_data dict (match GoldPredictionRecord schema với ml_dl_conflict, ml_dl_gap, fusion_applied)
Action:  SyncSession → GoldPrediction ORM object → session.commit()
DB:      INSERT vào gold_predictions table (PostgreSQL)
Retry:   3 lần, delay 5s
```

#### Task 4: `check_retrain_trigger` (Celery Beat — Daily)
```
Schedule: Mỗi 24 giờ (86400 giây)
Action:   Đếm Silver files trong 7 ngày gần nhất trên S3
          Nếu count >= 500 → log trigger event (pipeline chưa implement)
Retry:    1 lần
Note:     Actual retraining pipeline (trigger_retraining_pipeline.delay())
          hiện đang được comment out — chỉ log
```

### Retry Policy (tất cả tasks)

```python
acks_late=True   # ACK chỉ sau khi task thành công (không mất message)
max_retries=3    # Tối đa 3 lần thử lại
default_retry_delay=10  # Đợi 10s giữa các lần retry
```

---

## 9. Luồng ETL Đầy Đủ (End-to-End)

```
User submit code
       │
       ▼
Django submit_view()
       │ save BronzeSubmission
       │ push_bronze_to_s3.delay(code_hash, raw_code, metadata)  ──→ Celery
       │ [AJAX] return {hash, status:"pending"}
       │
       ▼
sse_proxy_view() → POST /api/analyze_stream (FastAPI)
       │
       ▼
FastAPI AgentService.analyze_code_stream()
   [Router → Analyzer → Fingerprint XAI → Judge → Critique]
       │
       │ On step "complete":
       ├── BronzeRepository.update_result(result_json)           ← Django
       ├── log_prediction_to_gold.delay(prediction_data)         ──→ Celery
       └── [optional] extract_and_push_silver.delay(code_hash)   ──→ Celery
       │
       ▼
Browser nhận SSE 100% → redirect result_view()

═══════ Background (Celery Workers) ═══════════════════════════════

push_bronze_to_s3:
  data_lake/bronze/{date}/{hash}.jsonl  (local + S3)

extract_and_push_silver:
  44 features + 512 tokens
  data_lake/silver/{date}/{hash}.json   (local + S3)

log_prediction_to_gold:
  INSERT INTO gold_predictions (PostgreSQL)

check_retrain_trigger (daily):
  Count silver files 7 days → if ≥500 → trigger retrain
```

---

## 10. Tóm Tắt — Shared Module Cheat Sheet

| Module | Singleton? | Dùng ở đâu | Quan trọng nhất |
|--------|-----------|-----------|----------------|
| `config.py` → `settings` | ✅ | Khắp nơi | `FASTAPI_AI_URL`, `AMBIGUOUS_LOW/HIGH` |
| `logger.py` → `AppLogger` | ✅ | Khắp nơi | `rotate_on_startup()`, `@log_function()` |
| `hashing.py` | Hàm thuần | Mọi tầng | `compute_code_hash()` → primary key |
| `database.py` | Engine (1 async, 1 sync) | FastAPI + Celery | `get_async_session()`, `get_sync_session()` |
| `s3_client.py` | `DagsHubS3Client` | Celery tasks | Local fallback tự động |
| `message_broker.py` | `EventProducer` | Optional | Topics: 3 topics |
| `data_contracts.py` | Pydantic schemas | Django + FastAPI + Celery | Single source of truth |

---

## Bước Tiếp Theo

| Bước | Nội dung | Trạng thái |
|------|----------|------------|
| **Bước 1** | Tổng quan hệ thống, Tech Stack, Medallion Architecture | ✅ Hoàn thành |
| **Bước 2** | MLOps Pipeline: LangGraph 5-node, RoBERTa, LightGBM, XAI | ✅ Hoàn thành |
| **Bước 3** | Admin Dashboard: 6 views, metrics API, infra healthcheck | ✅ Hoàn thành |
| **Bước 4** | Shared infrastructure: Logger, Config, S3, Celery, DB | ✅ Hoàn thành |
| **Bước 5** | Vận hành: khởi động, debugging, monitoring, CI/CD | 🔲 Chờ review |
