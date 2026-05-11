# Local Agent — Đặc tả Kỹ thuật (SPEC)

> **Cập nhật lần cuối:** 2026-04-29

## 1. Module Overview

Local Agent là **Smart Orchestrator (Bộ não)** trong kiến trúc microservices phân tán. Nó chạy trên máy cá nhân (CPU only), chịu trách nhiệm:

- **Điều phối** toàn bộ pipeline AI qua LangGraph (Router → Analyzer → Judge → Critique)
- **Gọi API** đến Colab GPU Worker để lấy dữ liệu thô (scores, tokens, attrs)
- **Xử lý logic** nhẹ: Fusion score, Render HTML heatmap, Phân tích tín hiệu AI/Human
- **Cache kết quả** (Lớp 1) để trả về ngay lập tức nếu cùng code

> **Triết lý:** Máy Local không cần GPU. Mọi tính toán ML/DL nặng đều được gửi lên Colab. Local chỉ giữ logic, cache, và rendering.

### So sánh vai trò

| | Local Brain (Module này) | Colab Worker |
|---|---|---|
| **Vai trò** | Smart Orchestrator — điều phối | Dumb Worker — tính toán |
| **Chạy trên** | Máy cá nhân (CPU) | Google Colab (T4/L4 GPU) |
| **Port** | `localhost:8080` | ngrok tunnel → port 8000 |
| **Yêu cầu phần cứng** | CPU + RAM 4GB đủ | GPU 16GB VRAM |

## 2. Cấu trúc thư mục

```
src/local_agent/
├── local_agent_SPEC.md        # File này — Đặc tả kỹ thuật
├── local_agent_SRS.md         # Đặc tả nghiệp vụ
├── requirements.txt           # Dependencies: fastapi, uvicorn, httpx, langgraph, numpy
├── main.py                    # FastAPI entry point + Cache Lớp 1 + Auto port cleanup
├── local_orchestrator.py      # LangGraph workflow + HTML Renderer + ExpertExplainer
├── remote_client.py           # Async HTTP client gọi API lên Colab Worker
└── test_colab_logic.py        # Script test end-to-end (gửi code → nhận Gold Metadata)
```

## 3. Chi tiết từng Module

### 3.1. `main.py` — FastAPI Entry Point

**Chức năng chính:**
- Khởi động FastAPI server tại `localhost:8080`
- Tự động kill process cũ đang chiếm cổng 8080 (dùng `lsof + os.kill`)
- Quản lý Cache Lớp 1 (`RESULT_CACHE`)
- Route endpoint `/api/analyze` → LangGraph pipeline

**Auto Port Cleanup:**
```python
# Khi chạy python -m src.local_agent.main:
# 1. Tìm PID chiếm cổng 8080 (lsof -t -i:8080)
# 2. Kill bằng SIGTERM
# 3. Chờ 1 giây → Khởi động server mới
```

**Cache Lớp 1:**

| Thuộc tính | Giá trị |
|-----------|---------|
| Biến | `RESULT_CACHE` (dict) |
| Key | `md5(decoded_code)` |
| Max size | 100 entries (FIFO eviction) |
| Scope | Kết quả cuối cùng (Gold Metadata hoàn chỉnh) |
| Khi hit | Trả ngay, không gọi Colab |

**Endpoints:**

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/` | Health check |
| `POST` | `/api/analyze` | Phân tích code C++ → Gold Metadata JSON |

**Request/Response `/api/analyze`:**
```
Request:  { "code_base64": "I2luY2x1ZGUg..." }
Response: { "result": { ...Gold Metadata... }, "cached": true/false }
```

### 3.2. `local_orchestrator.py` — LangGraph Workflow

Đây là **trái tim logic** của hệ thống. Sử dụng LangGraph `StateGraph` với 4 node:

```
START → Router → Analyzer → Judge ─┬→ Critique → END
                                    │
                                    └→ (Retry) → Analyzer (nếu ambiguous)
```

#### AgentState (TypedDict)

| Field | Type | Mô tả |
|-------|------|-------|
| `code_input` | `str` | Code C++ đã decode |
| `classification` | `Optional[str]` | `"OOP"` hoặc `"NORMAL"` (từ Router) |
| `is_ambiguous` | `bool` | Judge quyết định có retry không |
| `retry_count` | `int` | Số lần retry (max 1) |
| `final_output` | `Optional[Dict]` | Kết quả Gold Metadata cuối cùng |

#### Node 1: `router_node`
- **Input:** `code_input`
- **Output:** `classification` ("OOP" | "NORMAL")
- **Logic:** Gọi vLLM proxy (Qwen 2.5 Coder 7B) để phân loại code
- **Fallback:** Nếu lỗi → mặc định `"NORMAL"`

#### Node 2: `analyzer_node`
- **Input:** `code_input`, `classification`
- **Output:** `final_output` (Gold Metadata dict)
- **Logic:**
  1. Gọi **RoBERTa API** (`/api/predict/roberta`) → `bert_score`, tokens, attrs
  2. Gọi **LightGBM API** (`/api/predict/lightgbm`) → `lgbm_score`
  3. **Fusion:** `final_score = 0.48 × bert_score + 0.52 × lgbm_score`
  4. **Render HTML** (tại Local): XAI heatmap từ tokens + attrs
  5. **ExpertExplainer** (tại Local): Trích xuất top AI/Human signals
  6. Gọi **Perplexity API** (`/api/proxy/vllm`) → Dictionary chứa 3 chỉ số chuyên sâu:
     - `mean_ppl`: Perplexity trung bình của cả đoạn code
     - `max_ppl`: Perplexity cao nhất (dòng code làm AI bất ngờ nhất)
     - `burstiness`: Độ biến thiên (lệch chuẩn) của Perplexity

#### Node 3: `judge_node`
- **Input:** `final_output` (score + perplexity)
- **Output:** `is_ambiguous` (bool)
- **Quy tắc phát hiện nhập nhằng:**
  - Score nằm trong vùng "lửng": `0.45 ≤ score ≤ 0.55`
  - Score cao nhưng perplexity cũng cao: `score > 0.6 AND ppl > 3.0`
  - Score thấp nhưng perplexity cũng thấp: `score < 0.4 AND ppl < 1.0`
- **Retry:** Nếu ambiguous và chưa retry → quay lại `analyzer_node`

#### Node 4: `critique_node`
- **Input:** `code_input`, `final_score`
- **Output:** `global_critique` (string)
- **Logic:** Gọi vLLM proxy để sinh nhận xét bảo mật/code quality

#### Logic chuyển đã tại Local (Không nằm trên Colab)

| Logic | Ở đâu trước đây | Hiện tại |
|-------|-----------------|----------|
| LangGraph StateGraph | `colab_runtime/scripts/agent.py` | `local_orchestrator.py` |
| ExpertExplainer | `colab_runtime/scripts/engine.py` | `local_orchestrator.py` |
| `render_html_view` | `colab_runtime/scripts/engine.py` | `local_orchestrator.py` |
| Fusion score | `colab_runtime/scripts/hybrid_evaluator.py` | `local_orchestrator.py` |
| Perplexity call | `colab_runtime/scripts/llm_handler.py` | `remote_client.py` |

### 3.3. `remote_client.py` — Async HTTP Client

**Class `ColabAPIClient` (Singleton `client`):**

Gọi API bất đồng bộ lên Colab Worker qua ngrok tunnel.

| Method | Target Endpoint | Mô tả |
|--------|----------------|-------|
| `get_roberta_data(code, model_type)` | `POST /api/predict/roberta` | Lấy score + tokens + attrs phẳng |
| `get_lightgbm_score(code)` | `POST /api/predict/lightgbm` | Lấy LightGBM probability |
| `get_perplexity(code)` | `POST /api/proxy/vllm` | Lấy dict 3 chỉ số (mean_ppl, max_ppl, burstiness) |
| `get_critique(code, score)` | `POST /api/proxy/vllm` | Sinh nhận xét code quality |

**Config đọc từ biến môi trường:**

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `NGROK_URL` | `https://cedric-unstony-fulsomely.ngrok-free.dev` | URL Colab Worker |
| `API_KEY` | `colab-secret-key-123` | API key xác thực |

> **Lưu ý quan trọng:** `get_roberta_data()` trả về **toàn bộ response** phẳng (`return data`), không tìm key `details`. Điều này khớp với Colab trả về `{**result_data}`.

### 3.4. `test_colab_logic.py` — Script Test End-to-End

- Gửi một đoạn code C++ mẫu (tính tổng 1→n) lên `localhost:8080/api/analyze`
- In ra Gold Metadata JSON (không kèm HTML dài)
- In ra Global Critique
- Dùng để kiểm tra toàn bộ pipeline hoạt động đúng

## 4. Gold Metadata Schema (Output)

Kết quả trả về từ pipeline tuân theo Gold Schema cho PostgreSQL:

```json
{
  "model_used": "C++ Normal Model",
  "classification": "NORMAL",
  "final_score": 0.3053,
  "confidence": 0.3053,
  "bert_score": 0.6168,
  "lgbm_score": 0.0177,
  "total_tokens": 110,
  "total_chunks": 1,
  "top_ai_signals": ["'++'", "'in'", "'<<'", ...],
  "top_hu_signals": ["'return'", "'()'", "'namespace'", ...],
  "retry_count": 0,
  "final_pred": "HUMAN WRITTEN",
  "perplexity": 6.27,
  "max_ppl": 15.42,
  "burstiness": 3.85,
  "global_critique": "The provided C++ code is a simple program...",
  "is_ambiguous": false,
  "global_html": "<html>...</html>",
  "chunks": [{ "index": 1, "score": 0.617, "label": "AI", "html": "...", ... }]
}
```

## 5. Hệ thống Cache 2 lớp (Phối hợp với Colab)

```
User gửi code → [Lớp 1: Local RESULT_CACHE]
                    ├── HIT  → Trả ngay (⚡ 0.01s) — không gọi Colab
                    └── MISS → Gọi Colab APIs
                                    ├── [Lớp 2: COLAB_CACHE]
                                    │       ├── HIT  → Trả ngay (⚡ không chạy GPU)
                                    │       └── MISS → Chạy GPU inference (🚀 15-25s)
                                    └── Kết quả → Lưu vào Lớp 1
```

**Đồng bộ Hash:** Cả hai lớp dùng MD5 trên nội dung code đã `decode() → replace('\\r\\n', '\\n') → strip()`.

## 6. Khởi động và Vận hành

```bash
# Khởi động (tự động kill process cũ)
python -m src.local_agent.main

# Test
python src/local_agent/test_colab_logic.py
```

**Yêu cầu trước khi chạy:**
1. Colab Worker phải đang chạy (có ngrok URL)
2. `NGROK_URL` trong `remote_client.py` hoặc biến môi trường phải đúng
3. Cài dependencies: `pip install -r src/local_agent/requirements.txt`
