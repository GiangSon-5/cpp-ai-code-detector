# Colab Runtime — Đặc tả Nghiệp vụ (SRS)

> **Cập nhật lần cuối:** 2026-04-29 — Refactor sang kiến trúc Dumb Worker

### UC: Chạy GPU Worker trên Google Colab — Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Colab Runtime hoạt động như một **GPU Worker** (Dumb Worker) trong kiến trúc microservices. Nó chỉ chịu trách nhiệm chạy inference (RoBERTa, LightGBM, vLLM proxy) và trả về kết quả thô. Toàn bộ logic điều phối (LangGraph Agent, Fusion, Render HTML) nằm ở Local Brain (`src/local_agent/`).

User chỉ cần mở Colab, chạy `!python bootstrap.py`, hệ thống tự khởi động FastAPI server + ngrok tunnel + vLLM server, sẵn sàng nhận request từ Local Brain.

| Primary Actor: | DevOps / Researcher | Secondary Actor: | Google Colab / ngrok |
|----------------|---------------------|-------------------|----------------------|
| **Description:** | Triển khai GPU Worker inference server trên Colab miễn phí |
| **Trigger:** | User mở Colab notebook và chạy `!python bootstrap.py` |
| **Preconditions:** | PRE1: Google account với Colab access. PRE2: Model weights trên Google Drive. PRE3: ngrok token trong Colab Secrets. |
| **Post-conditions:** | POST1: FastAPI server (port 8000) + vLLM server (port 8001) running. POST2: ngrok URL available để Local Brain kết nối. |

---

## Business Scenario Walkthrough

- **Đầu vào:** Researcher mở Colab, chạy `!python bootstrap.py`
- **Hệ thống xử lý:** Install deps → Mount Drive → Start vLLM (2-3 min) → Load RoBERTa + LightGBM → Start FastAPI → Start ngrok
- **Đầu ra:** URL ngrok (VD: `https://xyz.ngrok-free.dev`) — copy vào file `.env` của Local Brain

---

## Normal Flow

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Mở Colab, chọn GPU runtime (T4/L4) | Colab allocate GPU instance |
| 2 | Chạy `!python bootstrap.py` | Cài đặt thư viện qua `uv` (~30 giây) |
| 3 | (Tự động) Mount Google Drive | Google Drive mount tại `/content/drive` |
| 4 | (Tự động) Khởi động vLLM | Qwen 2.5 Coder 7B FP16 load vào GPU (~2-3 phút) |
| 5 | (Tự động) Load AI models | RoBERTa 5-fold + LightGBM + Scaler (~1 phút) |
| 6 | (Tự động) Start FastAPI + ngrok | Server listen :8000, ngrok tunnel → in ra URL |
| 7 | Copy ngrok URL | Paste vào file `.env` của Local Brain (`NGROK_URL=...`) |
| 8 | (Background) Server xử lý requests | Nhận Base64 code → Inference → Trả JSON phẳng |

---

## Exception

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | GPU runtime không khả dụng | In "Chọn GPU runtime". Fallback CPU mode (chậm ~10x) |
| 2 | Google Drive không mount | Error message + hướng dẫn re-mount |
| 3 | Model folder không tồn tại | In path expected, hướng dẫn upload model lên Drive |
| 4 | ngrok token hết hạn | In link đăng ký ngrok mới |
| 5 | Session timeout (90 phút idle) | User re-run `bootstrap.py`. Local Brain thấy lỗi kết nối |
| 6 | VRAM OOM | Giảm `ENGINE_BATCH_SIZE`, `LIG_N_STEPS` trong `config.py` |
| 7 | vLLM không start được | In lỗi + đường dẫn log `/content/vllm.log` |
| 8 | RoBERTa endpoint trả 404 | Kiểm tra cấu trúc (indentation) trong `server.py` |

---

## Business Rules

| No | Rule |
|----|------|
| 1 | Colab **chỉ** là Dumb Worker — **không** chứa logic điều phối (LangGraph, Judge, Critique) |
| 2 | Response từ Colab phải trả **JSON phẳng** (không bọc trong `details`) để Local Brain đọc trực tiếp |
| 3 | Notebook phải chạy trên GPU runtime (T4 tối thiểu, L4 recommended) |
| 4 | Model weights lưu trên Google Drive: `PATH_OOP`, `PATH_NORMAL`, `PATH_SAVED_MODELS` |
| 5 | API authentication bằng header `X-API-Key` (mặc định: `colab-secret-key-123`) |
| 6 | Hash cache giữa Local và Colab phải **đồng bộ** — cùng dùng `md5(f"prefix_{decoded_code}")` |
| 7 | ngrok URL thay đổi mỗi session → phải cập nhật `NGROK_URL` trong `.env` ở Local |
| 8 | Colab free tier: max ~12h continuous, throttle sau 90 phút idle |
| 9 | Scripts trong `scripts/` portable — chạy được cả Colab lẫn MicroK8s |

---

## API Contract (Giao diện giữa Colab Worker và Local Brain)

### POST `/api/predict/roberta`
**Request:**
```json
{
  "code_base64": "I2luY2x1ZGUg...",
  "model_type": "NORMAL"
}
```
**Response (phẳng):**
```json
{
  "success": true,
  "score": 0.6168,
  "model_used": "C++ Normal Model",
  "final_pred": "AI GENERATED",
  "final_score": 0.6168,
  "total_tokens": 110,
  "total_chunks": 1,
  "tokens": ["<s>", "#", "include", ...],
  "attrs": [0.034, 0.087, ...],
  "chunks": [{"index": 1, "score": 0.6168, "label": "AI", "tokens": [...], "attrs": [...], "snippet": "..."}]
}
```

### POST `/api/predict/lightgbm`
**Request:**
```json
{
  "code_base64": "I2luY2x1ZGUg..."
}
```
**Response:**
```json
{
  "success": true,
  "score": 0.0177
}
```

### POST `/api/proxy/vllm`
**Request:**
```json
{
  "payload": {
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 300
  }
}
```
**Response:** (proxy trực tiếp từ vLLM server port 8001)

---

## Data Mapping

| Tên trường | Mô tả | Kiểu | Mặc định | Bắt buộc | Lưu trữ tại |
|------------|-------|------|----------|----------|-------------|
| `NGROK_TOKEN` | ngrok auth token | string (secret) | — | Y | Colab Secrets |
| `API_KEY` | API key cho FastAPI | string | `colab-secret-key-123` | Y | Env var / Colab Secrets |
| `PATH_OOP` | OOP model directory | string | `/content/drive/.../C++_OOP_detection` | Y | Google Drive |
| `PATH_NORMAL` | Normal model directory | string | `/content/drive/.../C++_detection` | Y | Google Drive |
| `PATH_SAVED_MODELS` | LightGBM models | string | `/content/drive/.../Saved_Models/` | Y | Google Drive |
| `NGROK_URL` | Public URL output | string | — | auto-generated | Copy vào `.env` ở Local |
| `GPU_TYPE` | Colab GPU tier | string | T4 | auto-detected | Runtime |
| `FUSION_ALPHA` | Trọng số RoBERTa | float | `0.48` | Y | `config.py` |
| `DEFAULT_THRESHOLD` | Ngưỡng AI/Human | float | `0.5` | Y | `config.py` |

---

## Luồng dữ liệu tổng quan

```
┌──────────────────────────────────────────────────────────────────┐
│  LOCAL BRAIN (Smart Orchestrator) — localhost:8080               │
│  ┌────────┐  ┌──────────┐  ┌───────┐  ┌──────────┐             │
│  │ Router │→ │ Analyzer │→ │ Judge │→ │ Critique │→ Response    │
│  └────────┘  └────┬─────┘  └───────┘  └────┬─────┘             │
│                   │ (gọi API)               │ (gọi API)         │
└───────────────────┼─────────────────────────┼───────────────────┘
                    │                         │
         ┌──────────▼─────────────────────────▼──────────┐
         │  COLAB WORKER (Dumb Worker) — ngrok:8000      │
         │  ┌──────────────┐  ┌───────────┐  ┌────────┐ │
         │  │ /predict/    │  │ /predict/ │  │ /proxy/│ │
         │  │ roberta      │  │ lightgbm  │  │ vllm   │ │
         │  └──────────────┘  └───────────┘  └────────┘ │
         │       ↕                  ↕             ↕      │
         │  [RoBERTa GPU]   [LightGBM CPU]  [vLLM:8001] │
         └───────────────────────────────────────────────┘
```
