# 🌐 Tài Liệu Nội Bộ — Bước 3: Admin Dashboard & Django Web Layer

> **Phạm vi:** Django Web App — toàn bộ URL, Views, Templates, Repository  
> **File chính:** `src/django_web/apps/dashboard/views.py` (650 dòng)

---

## 1. URL Routing Toàn Hệ Thống

```
/ (root)
  └── root_redirect()         ← is_staff? → /dashboard/ : /submit/

/accounts/
  ├── login/                  → login_view()
  ├── logout/                 → logout_view()
  ├── register/               → register_view()
  └── profile/                → profile_view()

/submit/                      → submit_view()         [login_required]
  ├── result/<code_hash>/     → result_view()
  ├── history/                → history_view()
  ├── batch/<batch_id>/       → batch_result_view()
  ├── api/status/<code_hash>/ → status_api_view()     ← AJAX polling
  └── api/sse/<code_hash>/    → sse_proxy_view()      ← SSE streaming

/dashboard/                   → dashboard_view()      [login_required + is_staff]
  ├── api/stats/              → stats_api_view()      ← AJAX charts
  ├── api/metrics/            → metrics_api_view()    ← JS polling (30s)
  ├── metrics/                → metrics_view()        [staff_member_required]
  ├── infra/                  → infra_view()          [staff_member_required]
  ├── database/               → database_view()       [staff_member_required]
  ├── users/                  → users_view()          [staff_member_required]
  └── models/                 → models_view()         [staff_member_required]

/admin/                       → Django built-in Admin
```

---

## 2. Phân Quyền Người Dùng

| Loại | Điều kiện | Trang mặc định | Có vào Dashboard? |
|------|-----------|----------------|-------------------|
| **Admin / Staff** | `user.is_staff = True` | `/dashboard/` | ✅ Có |
| **User thường** | `user.is_staff = False` | `/submit/` | ❌ Bị chặn → redirect `/submit/` |
| **Chưa đăng nhập** | — | `/accounts/login/` | ❌ redirect login |

**Logic redirect trong `login_view()`:**
```python
if user.is_staff:
    return redirect("/dashboard/")    # Admin → Command Center
else:
    return redirect("/submit/")       # User → Submit page
```

**Logic trong `dashboard_view()`:**
```python
if not user.is_staff:
    return redirect("/submit/")       # Hard redirect nếu user thường vào nhầm
```

### UserProfile Model (`accounts/models.py`)
```python
class UserProfile(models.Model):
    user         = OneToOneField(User)
    organization = CharField(max_length=200)
    role         = CharField(max_length=50, default="student")
    # role: "student" | "teacher" | "admin"
```

---

## 3. BronzeSubmission Model — Nguồn Dữ Liệu Chính

**File:** `src/django_web/apps/submissions/models.py`

```python
class BronzeSubmission(models.Model):
    # Identity
    code_hash        # SHA-256(raw_code) — UNIQUE INDEX — cơ chế dedup
    user             # FK → auth_user (nullable)
    
    # Content
    raw_code         # TEXT — code C++ gốc
    language         # "cpp" (mặc định)
    source           # "pasted_code" | "file_upload" | "batch"
    filename         # tên file gốc (nếu upload)
    file_size_bytes  # kích thước bytes
    
    # Result (populated sau khi FastAPI trả về)
    prediction       # "" (pending) | "AI GENERATED" | "HUMAN WRITTEN"
    confidence       # float 0-1 (nullable)
    result_json      # JSON đầy đủ từ FastAPI (AnalyzeResponse)
    
    # Batch support
    batch            # FK → BatchSession (nullable)
    
    timestamp        # auto_now_add=True
```

> **`result_json`** là field quan trọng nhất: lưu toàn bộ `AnalyzeResponse` từ FastAPI,  
> bao gồm: `dl_score`, `ml_score`, `hybrid_score`, `perplexity`, `chunks`, `fingerprint`, v.v.

---

## 4. BronzeRepository — Repository Pattern

**File:** `src/django_web/apps/submissions/repositories/__init__.py`

Mọi thao tác DB đều đi qua class này (không query ORM trực tiếp trong views):

| Method | Mô tả |
|--------|-------|
| `save(user, raw_code, source, ...)` | `get_or_create` theo `code_hash` — tự động dedup |
| `get_by_hash(code_hash)` | Lookup một submission |
| `get_user_history(user, limit=50)` | Lịch sử của một user |
| `update_result(code_hash, prediction, confidence, result_json)` | Cập nhật kết quả sau inference |
| `count_all()` | Đếm tổng |
| `list_recent(limit=50)` | Lấy N submission mới nhất |

**Cơ chế dedup:** `code_hash = SHA-256(raw_code.strip())` → cùng code submit nhiều lần chỉ tính 1 record.

---

## 5. Luồng Submit Code (Submit Flow)

```
User paste code hoặc upload file
         │
         ▼
submit_view() [POST]
         │
         ├── BATCH upload (nhiều file)?
         │    └── Tạo BatchSession, spawn background thread, redirect /submit/batch/<id>/
         │
         ├── Single file hoặc paste?
         │    ├── Validate size (max 500KB)
         │    ├── BronzeRepository.save()
         │    │    └── get_or_create theo code_hash
         │    │
         │    ├── Duplicate (đã có result_json)?
         │    │    └── redirect thẳng → result_view()  [cache hit]
         │    │
         │    ├── AJAX request? (X-Requested-With: XMLHttpRequest)
         │    │    └── return JsonResponse({hash, status:"pending"})
         │    │         → JS frontend kết nối SSE
         │    │
         │    └── Fallback sync (non-AJAX):
         │         └── POST http://127.0.0.1:8001/api/analyze (timeout 120s)
         │              → update_result() → redirect result_view()
         ▼
result_view(code_hash)
    └── Render result.html với submission.result_json
```

### SSE Proxy Flow (AJAX path)

```
Browser JS
  │ 1. POST /submit/ → { hash: "abc123", status: "pending" }
  │
  │ 2. EventSource("/submit/api/sse/abc123/")
  │
  ▼
sse_proxy_view(code_hash)
  │ stream POST http://127.0.0.1:8001/api/analyze_stream
  │ → mỗi line "data: {...}" → yield → EventSource nhận
  │
  │ Khi step == "complete":
  │   BronzeRepository.update_result(code_hash, prediction, confidence, result_json)
  │
  ▼
Browser nhận progress events → cập nhật UI progress bar
Khi 100% → redirect /submit/result/<hash>/
```

---

## 6. Sáu Admin Views — Chi Tiết

Tất cả views trong `dashboard/views.py`, đều yêu cầu `@staff_member_required` (trừ 2 views đầu).

### 6.1 `dashboard_view` — `/dashboard/`
**Template:** `dashboard/dashboard.html`  
**`active_nav`:** `"overview"`

Dữ liệu trả về context:

| Key | Nội dung | Nguồn |
|-----|---------|-------|
| `user_stats` | total, ai_count, human_count, pending, avg_confidence, ai_ratio | Query BronzeSubmission filter(user=user) |
| `global_stats` | total, ai_count, human_count, avg_confidence | Query tất cả BronzeSubmission |
| `recent_submissions` | 10 bài nộp gần nhất của user | BronzeSubmission.order_by("-timestamp")[:10] |
| `metrics` | `{}` (rỗng) | Được populate bởi JS polling `/dashboard/api/metrics/` |

> **Pattern:** Số liệu realtime (FastAPI health, VRAM, ...) **không** render server-side. Chúng được load bằng JS polling `metrics_api_view` mỗi 30 giây.

---

### 6.2 `metrics_view` — `/dashboard/metrics/`
**Template:** `dashboard/metrics.html` (Chart.js)  
**`active_nav`:** `"metrics"`

Đây là trang MLOps analytics chính. Dữ liệu được tính toàn bộ server-side:

| Nhóm | Metrics | Cách tính |
|------|---------|-----------|
| **KPI Cards** | total, ai_count, pending, avg_confidence | Aggregate query |
| **Stale Alert** | `stale_pending` — pending > 10 phút | `prediction="" AND timestamp < now-10min` |
| **Cache Rate** | `cache_hit_rate` — tỷ lệ hash trùng | code_hash có count > 1 / total |
| **Score Cards** | `avg_dl`, `avg_ml`, `avg_hybrid` | Từ `result_json` của từng submission |
| **Histogram** | 10 bucket: 0–10%, 10–20%, ..., 90–100% | Đếm submission theo khoảng confidence |
| **Daily Trend** | 7 ngày: AI vs Human per day | Loop 7 ngày, filter timestamp range |
| **Model Router** | OOP vs NORMAL count | Từ `result_json.model_used` |
| **Source** | pasted / file_upload / batch | Filter `source` field |
| **FastAPI Health** | p95 latency, request count | GET http://localhost:8001/health |

**Charts.js trên trang Metrics:**

| Chart ID | Loại | Dữ liệu |
|----------|------|---------|
| `confHistChart` | Bar (histogram) | 10 bucket confidence — màu xanh (<50%) / đỏ (≥50%) |
| `dailyTrendChart` | Bar grouped | AI (đỏ) + Human (xanh) per day, 7 ngày |
| `modelUsageChart` | Doughnut | OOP (tím) vs NORMAL (cyan) |
| `sourceChart` | Doughnut | Pasted (xanh) / File (vàng) / Batch (tím) |

---

### 6.3 `infra_view` — `/dashboard/infra/`
**Template:** `dashboard/infra.html`  
**`active_nav`:** `"infra"`

Thực hiện **real healthcheck** cho từng service khi page load:

```python
# TCP check (socket.create_connection)
redis_ok    = _check_tcp("localhost", 6379)
redpanda_ok = _check_tcp("localhost", 9092)

# HTTP healthcheck
fastapi_health = GET http://localhost:8001/health
fastapi_latency = round(latency_ms, 1)

# Django DB check
connection.ensure_connection() → pg_latency

# Celery inspect
celery_app.control.inspect(timeout=1.0).ping() → worker count
```

**Services list** render dưới dạng card với màu/status động:

| Service | Status | Màu |
|---------|--------|-----|
| FastAPI AI Service | HTTP 200 từ /health | `#22d3ee` (cyan) |
| Django Web Server | Luôn "online" (đang serve page này) | `#60a5fa` (blue) |
| PostgreSQL | TCP 5432 + `ensure_connection()` | `#4ade80` (green) |
| Redis | TCP 6379 | `#fb923c` (orange) |
| Celery Worker | `inspect().ping()` > 0 workers | `#c084fc` (purple) |
| Redpanda | TCP 9092 | `#fbbf24` (yellow) |

**Observability tools** (quick links):
- FastAPI Swagger: `http://localhost:8001/docs`
- FastAPI ReDoc: `http://localhost:8001/redoc`
- Django Admin: `/admin/`

**Data Persistence section** (static display):
- PostgreSQL: `localhost:5432/cpp_detector`
- DagsHub S3: `s3://dagshub/silver/*.parquet`
- DVC: `dvc push/pull`
- Redis: `redis://localhost:6379`

---

### 6.4 `database_view` — `/dashboard/database/`
**Template:** `dashboard/database.html`  
**`active_nav`:** `"database"`

Database explorer — hiển thị 200 submission gần nhất dưới dạng bảng:

```python
all_submissions = BronzeSubmission.objects.select_related("user").order_by("-timestamp")[:200]
```

Cũng render schema của Bronze table (hardcoded trong view):

| Column | Type |
|--------|------|
| id | BIGINT PK |
| code_hash | VARCHAR(64) UNIQUE |
| user_id | INT FK → auth_user |
| raw_code | TEXT |
| language | VARCHAR(10) |
| source | VARCHAR(50) |
| prediction | VARCHAR(50) |
| confidence | FLOAT NULL |
| file_size_bytes | INT |
| timestamp | TIMESTAMPTZ |

---

### 6.5 `users_view` — `/dashboard/users/`
**Template:** `dashboard/users.html`  
**`active_nav`:** `"users"`

Đơn giản: list tất cả `User.objects.all().order_by("-date_joined")`.  
Không có edit, delete — chỉ xem danh sách.

---

### 6.6 `models_view` — `/dashboard/models/`
**Template:** `dashboard/models.html`  
**`active_nav`:** `"models"`

Model Registry — thông tin về model đang chạy:

```python
context = {
    "active_model": {
        "alpha": 0.48,           # Fusion weight (hardcoded)
        "threshold": 0.5,        # Decision threshold
        "val_accuracy": "92.4%", # Hardcoded (update sau mỗi retrain)
    },
    "model_history": GoldModelPerformance.objects.order_by("-timestamp")[:20],
    "fastapi_url": os.environ.get("FASTAPI_AI_URL", "http://localhost:8001"),
}
```

> **Lưu ý:** `GoldModelPerformance` là model Django (không phải SQLAlchemy) nếu đã migrate. Hiện tại `val_accuracy` được hardcode.

---

## 7. Metrics API — `/dashboard/api/metrics/`

**Hàm:** `metrics_api_view()` — được gọi bởi JS polling mỗi 30 giây trên tất cả admin pages.

Dữ liệu JSON trả về:

```json
{
  "total": 150,
  "ai_count": 92,
  "human_count": 58,
  "avg_confidence": 0.7431,
  "submissions_today": 12,
  "celery_backlog": 0,

  "inference_p95_ms": 4200,
  "last_latency_ms": 3800,
  "request_count": 89,
  "cache_size": 23,

  "gpu_vram_used_gb": 3.2,
  "gpu_vram_total_gb": 4.0,
  "gpu_vram_pct": 80.0,
  "models_loaded": true,
  "gpu_available": true,
  "fastapi_uptime_s": 3600,

  "redis_ok": true,
  "api_latency_ms": 45.2
}
```

**Nguồn dữ liệu từng field:**

| Field | Nguồn | Fallback |
|-------|-------|---------|
| `total`, `ai_count`, ... | Django ORM (BronzeSubmission) | 0 |
| `celery_backlog` | `celery_app.control.inspect().reserved()` | 0 |
| `inference_p95_ms`, `request_count`, `cache_size` | GET http://localhost:8001/health | None/0 |
| `gpu_vram_*` | GET `FASTAPI_AI_URL/gpu-stats` (Port 8002) | Fallback sang health của 8001 |
| `redis_ok` | `redis.Redis().ping()` | False |
| `api_latency_ms` | Đo thời gian render chính metrics_api_view | — |

> **VRAM priority:** Ưu tiên lấy từ **Local GPU Server** (port 8002) vì model thực sự chạy ở đó. FastAPI Orchestrator (8001) chỉ là fallback.

---

## 8. Base Template — Layout Admin

**File:** `src/django_web/templates/base_admin.html`

- **Framework:** TailwindCSS (CDN) + Font Outfit (Google Fonts)
- **Layout:** `flex` ngang — Sidebar trái cố định (288px) + Main content cuộn phải
- **Theme:** Dark glassmorphism (`background:#0b0f1a`, `.glass` class)

**Sidebar navigation** (6 items):

| Item | URL | `active_nav` | Màu active |
|------|-----|--------------|-----------|
| Live Monitoring | `/dashboard/` | `"overview"` | cyan |
| Metrics & Grafana | `/dashboard/metrics/` | `"metrics"` | orange |
| Infrastructure | `/dashboard/infra/` | `"infra"` | orange |
| System Database | `/dashboard/database/` | `"database"` | indigo |
| User Management | `/dashboard/users/` | `"users"` | cyan |
| Model Registry | `/dashboard/models/` | `"models"` | purple |

**Bottom status indicators** (hardcoded):
- `Broker: Redpanda` — green dot
- `S3: DagsHub` — green dot
- Signed in as `{{ user.username }}` + Logout link

**JS dependencies** (load ở cuối body):
- Chart.js 4.4.0 từ CDN
- `{% static 'js/dashboard.js' %}` — custom polling và metrics update

---

## 9. Template Kế Thừa

```
base_admin.html  (Sidebar + main layout)
    ├── dashboard/dashboard.html   → {% extends "base_admin.html" %}
    ├── dashboard/metrics.html     → {% extends "base_admin.html" %}
    ├── dashboard/infra.html       → {% extends "base_admin.html" %}
    ├── dashboard/database.html    → {% extends "base_admin.html" %}
    ├── dashboard/users.html       → {% extends "base_admin.html" %}
    └── dashboard/models.html      → {% extends "base_admin.html" %}

base.html  (User-facing layout)
    ├── submissions/submit.html
    ├── submissions/result.html    → Hiển thị XAI (fingerprint + SHAP + LIG)
    ├── submissions/history.html
    ├── submissions/batch_result.html
    ├── accounts/login.html
    ├── accounts/register.html
    └── accounts/profile.html
```

---

## 10. Tóm Tắt — Cheat Sheet Dashboard

```
Muốn biết...                         → Xem ở đâu
─────────────────────────────────────────────────────────────────
Tổng quan stats + recent submissions → /dashboard/
Biểu đồ phân tích ML metrics        → /dashboard/metrics/
Trạng thái service/port live         → /dashboard/infra/
Danh sách tất cả submissions         → /dashboard/database/
Danh sách users                      → /dashboard/users/
Model đang chạy, alpha, accuracy     → /dashboard/models/
JSON metrics realtime (API)          → /dashboard/api/metrics/
FastAPI docs                         → http://localhost:8001/docs
Django admin built-in                → /admin/
```

---

## Bước Tiếp Theo

| Bước | Nội dung | Trạng thái |
|------|----------|------------|
| **Bước 1** | Tổng quan hệ thống, Tech Stack, Medallion Architecture | ✅ Hoàn thành |
| **Bước 2** | MLOps Pipeline: LangGraph 5-node, RoBERTa, LightGBM, XAI | ✅ Hoàn thành |
| **Bước 3** | Admin Dashboard: 6 views, metrics API, infra healthcheck | ✅ Hoàn thành |
| **Bước 4** | Shared infrastructure: Logger, Config, S3, Celery, DB | 🔲 Chờ review |
| **Bước 5** | Vận hành: khởi động, debugging, monitoring, CI/CD | 🔲 |
