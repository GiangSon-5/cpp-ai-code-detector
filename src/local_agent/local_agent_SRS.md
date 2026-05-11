# Local Agent — Đặc tả Nghiệp vụ (SRS)

> **Cập nhật lần cuối:** 2026-04-29

### UC: Điều phối AI Inference Pipeline — Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Local Agent đóng vai trò **Smart Orchestrator** — bộ não điều phối toàn bộ pipeline phát hiện code C++ do AI sinh ra. Nó nhận code từ user, phân luồng qua LangGraph (Router → Analyzer → Judge → Critique), gọi API lên Colab GPU Worker để inference, rồi tổng hợp kết quả thành Gold Metadata JSON hoàn chỉnh.

Module này chạy trên máy cá nhân (không cần GPU), sử dụng CPU để xử lý logic nhẹ: fusion score, render HTML heatmap, phân tích tín hiệu AI/Human, và caching kết quả.

| Primary Actor: | End User / Frontend | Secondary Actor: | Colab GPU Worker |
|----------------|---------------------|-------------------|------------------|
| **Description:** | Phân tích code C++ để phát hiện AI-generated code |
| **Trigger:** | User gửi POST request chứa code C++ (base64) đến `/api/analyze` |
| **Preconditions:** | PRE1: Local Brain server đang chạy (port 8080). PRE2: Colab Worker đang online (ngrok URL hợp lệ). PRE3: vLLM server đang chạy trên Colab (port 8001). |
| **Post-conditions:** | POST1: Gold Metadata JSON trả về cho user. POST2: Kết quả được cache trong `RESULT_CACHE`. |

---

## Business Scenario Walkthrough

- **Đầu vào:** User gửi đoạn code C++ (đã encode base64) đến endpoint `/api/analyze`
- **Hệ thống xử lý:** Decode → Check cache → LangGraph pipeline (Router → Analyzer → Judge → Critique) → Tổng hợp kết quả
- **Đầu ra:** JSON chứa: điểm AI/Human, top signals, HTML heatmap, 3 chỉ số perplexity (mean, max, burstiness), critique nhận xét

---

## Normal Flow

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | User gửi `POST /api/analyze` với `code_base64` | Decode base64 → Tính MD5 hash |
| 2 | (Tự động) Kiểm tra Cache | Nếu **HIT**: trả kết quả ngay (⚡ 0.01s) → Kết thúc |
| 3 | (Tự động) **Router Node** | Gọi vLLM proxy → phân loại "OOP" hoặc "NORMAL" (~1s) |
| 4 | (Tự động) **Analyzer Node** | Gọi 3 API song song lên Colab: RoBERTa, LightGBM, Perplexity (Mean, Max, Burstiness) |
| 5 | (Tự động) Fusion + Render | Tính `fused = 0.48×BERT + 0.52×LGBM`. Render HTML heatmap locally |
| 6 | (Tự động) **Judge Node** | Kiểm tra confidence. Nếu nhập nhằng → retry (quay lại Step 4) |
| 7 | (Tự động) **Critique Node** | Gọi vLLM proxy → sinh nhận xét code quality (~3-5s) |
| 8 | (Tự động) Trả kết quả | Gold Metadata JSON → User. Lưu cache. |

---

## Exception

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Colab Worker offline (ngrok URL hết hạn) | `❌ Error (RoBERTa Data)`, `bert_score` fallback = 0.5 |
| 2 | vLLM server chưa sẵn sàng | Router fallback = "NORMAL", Critique = "API error" |
| 3 | Code rỗng | Trả `{"error": "Empty code"}` |
| 4 | Port 8080 đang bị chiếm | Tự động kill process cũ → Khởi động lại |
| 5 | Score nhập nhằng (0.45-0.55) | Judge đánh dấu `is_ambiguous=true` → Retry 1 lần |
| 6 | Perplexity cao bất thường | Judge kiểm tra tương quan score↔perplexity → có thể retry |
| 7 | LightGBM fail | Fallback: `lgbm_score = bert_score` |
| 8 | Network timeout (>45s) | httpx timeout → trả error |

---

## Business Rules

| No | Rule |
|----|------|
| 1 | Local Brain **không** chạy model ML/DL — mọi inference đều gửi lên Colab |
| 2 | Fusion score = `α × BERT + (1-α) × LightGBM` với `α = 0.48` |
| 3 | Ngưỡng phân loại: `score ≥ 0.5` → AI Generated, `score < 0.5` → Human Written |
| 4 | Cache FIFO: tối đa 100 entries, xóa entry cũ nhất khi đầy |
| 5 | Retry tối đa 1 lần khi Judge phát hiện nhập nhằng |
| 6 | HTML heatmap được render **tại Local** từ raw tokens + attribution scores |
| 7 | Perplexity được tính bằng cách gọi vLLM proxy qua Colab ngrok |
| 8 | Auto port cleanup: mỗi lần chạy sẽ kill process cũ trên cổng 8080 |

---

## API Contract

### POST `/api/analyze`

**Request:**
```json
{
  "code_base64": "I2luY2x1ZGUgPGlvc3RyZWFtPg0KdXNpbmcg..."
}
```

**Response (thành công):**
```json
{
  "result": {
    "model_used": "C++ Normal Model",
    "classification": "NORMAL",
    "final_score": 0.3053,
    "confidence": 0.3053,
    "bert_score": 0.6168,
    "lgbm_score": 0.0177,
    "total_tokens": 110,
    "total_chunks": 1,
    "top_ai_signals": ["'++'", "'in'", "'<<'"],
    "top_hu_signals": ["'return'", "'()'", "'namespace'"],
    "retry_count": 0,
    "final_pred": "HUMAN WRITTEN",
    "perplexity": 6.27,
    "max_ppl": 15.42,
    "burstiness": 3.85,
    "global_critique": "The provided C++ code is a simple program...",
    "is_ambiguous": false,
    "global_html": "<html>...(XAI heatmap)...</html>",
    "chunks": [
      {
        "index": 1,
        "score": 0.6168,
        "label": "AI",
        "tokens": ["<s>", "#", "include", ...],
        "attrs": [1.668, 0.034, 0.087, ...],
        "snippet": "#include <iostream>\nusing namespace std;...",
        "html": "<html>...(chunk heatmap)...</html>",
        "top_ai": ["'++'", "'include'"],
        "top_hu": ["'return'", "'main'"]
      }
    ]
  },
  "cached": false
}
```

**Response (cache hit):**
```json
{
  "result": { "...same Gold Metadata..." },
  "cached": true
}
```

**Response (lỗi):**
```json
{
  "error": "Connection refused"
}
```

---

## Data Mapping (Biến môi trường)

| Tên trường | Mô tả | Kiểu | Mặc định | Bắt buộc | Lưu trữ tại |
|------------|-------|------|----------|----------|-------------|
| `NGROK_URL` | URL Colab Worker | string | `https://cedric-unstony-fulsomely.ngrok-free.dev` | Y | `remote_client.py` / env var |
| `API_KEY` | API key xác thực Colab | string | `colab-secret-key-123` | Y | `remote_client.py` / env var |
| `FUSION_ALPHA` | Trọng số BERT trong Hybrid | float | `0.48` | Y | `local_orchestrator.py` |
| `MAX_CACHE_SIZE` | Số entry cache tối đa | int | `100` | Y | `main.py` |

---

## Luồng dữ liệu chi tiết

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LOCAL BRAIN (Smart Orchestrator) — localhost:8080                      │
│                                                                         │
│  POST /api/analyze                                                      │
│    │                                                                    │
│    ├─ Decode Base64 → MD5 Hash → Check RESULT_CACHE                    │
│    │     ├─ [HIT] ⚡ Return immediately                                │
│    │     └─ [MISS] ↓                                                   │
│    │                                                                    │
│    ├─ LangGraph Pipeline:                                               │
│    │   ┌──────────┐    ┌────────────┐    ┌─────────┐    ┌───────────┐  │
│    │   │  Router  │ →  │  Analyzer  │ →  │  Judge  │ →  │ Critique  │  │
│    │   │(vLLM)    │    │(3 API calls│    │(Retry?) │    │(vLLM)     │  │
│    │   └──────────┘    │ + Render   │    └────┬────┘    └───────────┘  │
│    │                   │ + Fusion)  │         │ Retry?                  │
│    │                   └────────────┘    ←────┘                         │
│    │                        │                                           │
│    │                   Gọi Colab APIs:                                   │
│    │                   ├── /api/predict/roberta  (BERT score + XAI)     │
│    │                   ├── /api/predict/lightgbm (LightGBM score)       │
│    │                   └── /api/proxy/vllm      (Perplexity + Critique)│
│    │                                                                    │
│    └─ Save to RESULT_CACHE → Return Gold Metadata JSON                 │
└─────────────────────────────────────────────────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────────┐
         │  COLAB WORKER (Dumb Worker) — ngrok:8000    │
         │  RoBERTa GPU │ LightGBM CPU │ vLLM:8001    │
         └─────────────────────────────────────────────┘
```

---

## Hướng dẫn vận hành

### Khởi động
```bash
# Cài dependencies (lần đầu)
pip install -r src/local_agent/requirements.txt

# Chạy server (tự động kill process cũ)
python -m src.local_agent.main
```

### Test
```bash
# Test end-to-end (gửi code mẫu → nhận Gold Metadata)
python src/local_agent/test_colab_logic.py
```

### Cập nhật ngrok URL
Khi Colab session mới tạo ra URL ngrok mới, cập nhật biến `NGROK_URL`:
- **Cách 1:** Sửa trực tiếp trong `remote_client.py` dòng 8
- **Cách 2:** Đặt biến môi trường `export NGROK_URL=https://new-url.ngrok-free.dev`
