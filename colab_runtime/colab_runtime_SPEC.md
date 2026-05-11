# Colab Runtime — Đặc tả Kỹ thuật (SPEC)

> **Cập nhật lần cuối:** 2026-04-29 — Refactor sang kiến trúc **"Smart Orchestrator / Dumb Worker"**

## 1. Module Overview

Colab Runtime là **GPU Worker** (Dumb Worker) trong kiến trúc microservices phân tán. Nó **chỉ** chịu trách nhiệm chạy các tác vụ tính toán nặng trên GPU (RoBERTa Ensemble, LIG Attribution, LightGBM, vLLM). **Toàn bộ logic điều phối** (LangGraph Agent, Render HTML, Heuristic Analysis) đã được chuyển sang **Local Brain** (`src/local_agent/`).

### Triết lý thiết kế

| | Colab (GPU Worker) | Local Brain (Orchestrator) |
|---|---|---|
| **Vai trò** | Dumb Worker — chỉ tính toán | Smart Orchestrator — điều phối logic |
| **Chạy trên** | Google Colab (T4/L4 GPU) | Máy cá nhân (CPU) |
| **Expose qua** | ngrok tunnel (port 8000) | localhost:8080 |
| **Chứa** | Model inference, Feature extraction | LangGraph Agent, HTML Render, Cache, Fusion |

> **Tại sao tách riêng?** Laptop 4GB VRAM sẽ bị OOM khi load model. Colab cung cấp T4 (16GB) miễn phí. Bằng cách tách Worker ra Colab, máy Local chỉ cần chạy logic nhẹ nhàng, không cần GPU.

## 2. Cấu trúc thư mục (Hiện tại)

```
src/colab_runtime/
├── colab_runtime_SPEC.md          # File này — Đặc tả kỹ thuật
├── colab_runtime_SRS.md           # Đặc tả nghiệp vụ
├── requirements.txt               # Danh sách thư viện cần cài trên Colab
├── bootstrap.py                   # One-command setup (cài đặt + khởi động server)
│
└── scripts/
    ├── __init__.py                # Package init
    ├── config.py                  # Cấu hình tập trung (device, paths, thresholds)
    ├── engine.py                  # Core engine (ModelManager, RoBERTa Ensemble, LIG)
    ├── server.py                  # FastAPI endpoints + ngrok + Cache (Dumb Worker)
    ├── feature_extractor.py       # CppFeatureExtractorV8 (32 đặc trưng tĩnh)
    └── hybrid_evaluator.py        # LightGBM prediction + Fusion score
```

### Các file đã bị XÓA so với phiên bản cũ

| File cũ | Lý do xóa | Chuyển sang đâu |
|---------|-----------|-----------------|
| `scripts/agent.py` | Logic LangGraph Agent | `src/local_agent/local_orchestrator.py` |
| `scripts/llm_handler.py` | Gọi vLLM/Qwen trực tiếp | `src/local_agent/remote_client.py` (gọi qua proxy) |
| `notebooks/` (thư mục) | Không còn cần notebook riêng | `bootstrap.py` thay thế |

## 3. Chi tiết từng Module

### 3.1. `config.py` — Cấu hình tập trung

| Hằng số | Giá trị | Mô tả |
|---------|---------|-------|
| `DEVICE` | `cuda` hoặc `cpu` | Tự động phát hiện GPU |
| `PATH_OOP` | `/content/drive/.../C++_OOP_detection` | Thư mục chứa 5 fold RoBERTa OOP |
| `PATH_NORMAL` | `/content/drive/.../C++_detection` | Thư mục chứa 5 fold RoBERTa Normal |
| `PATH_SAVED_MODELS` | `/content/drive/.../Saved_Models/` | Chứa LightGBM `.pkl` files |
| `DEFAULT_THRESHOLD` | `0.5` | Ngưỡng phân loại AI/Human |
| `FUSION_ALPHA` | `0.48` | Trọng số RoBERTa trong Hybrid Fusion |
| `MAX_LEN` | `510` | Max tokens per chunk (trừ CLS/SEP) |
| `STRIDE` | `256` | Sliding window stride |
| `LIG_N_STEPS` | `20` | Số bước Layer Integrated Gradients |
| `MAX_CACHE_SIZE` | `100` | Giới hạn số entry trong COLAB_CACHE |

### 3.2. `engine.py` — Core AI Engine

**Class `ModelManager`:**
- Load tokenizer (`RobertaTokenizer`) và 5-fold models từ Google Drive
- Load ngưỡng (`threshold.json`) nếu có
- Cung cấp `get_models(type_name)` để trả về danh sách model theo loại
- `cleanup()` để giải phóng VRAM khi shutdown

**Hàm `run_roberta_engine(code_text, model_type_name, manager)`:**
1. Tokenize code → sliding window chunks (MAX_LEN=510, STRIDE=256)
2. Với mỗi fold (5 fold):
   - Inference batch → xác suất AI/Human per chunk
   - Layer Integrated Gradients (LIG) → attribution scores per token
3. Aggregate kết quả: weighted average score, global attribution
4. **Trả về dict phẳng** (không bọc trong `details`):
   ```python
   {
       "model_used": "C++ Normal Model",
       "final_pred": "AI GENERATED" | "HUMAN WRITTEN",
       "final_score": 0.6168,      # Weighted average
       "total_tokens": 110,
       "total_chunks": 1,
       "tokens": ["<s>", "#", "include", ...],   # Global tokens
       "attrs": [0.034, 0.087, ...],              # Global attribution scores
       "chunks": [{                                # Per-chunk details
           "index": 1, "score": 0.6168, "label": "AI",
           "tokens": [...], "attrs": [...], "snippet": "..."
       }]
   }
   ```

> **Lưu ý quan trọng:** HTML rendering (`render_html_view`) và ExpertExplainer (`analyze`) đã được chuyển sang Local Brain. Colab chỉ trả về raw `tokens` và `attrs` (attribution scores).

### 3.3. `server.py` — FastAPI Endpoints (Dumb Worker)

#### Authentication
- API Key: Header `X-API-Key` (mặc định: `colab-secret-key-123`)
- Có thể thay đổi qua biến môi trường `API_KEY`

#### Caching (COLAB_CACHE)
- In-memory dictionary cache trên RAM Colab
- Hash key: `md5(f"bert_{model_type}_{decoded_code}")` hoặc `md5(f"lgbm_{decoded_code}")`
- **Đồng bộ với Local Brain:** Cả hai dùng cùng thuật toán hash dựa trên nội dung code đã decode và strip
- Mục đích: Tránh chạy lại GPU inference cho cùng một đoạn code

#### Endpoints

| Method | Path | Input | Output | Mô tả |
|--------|------|-------|--------|-------|
| `GET` | `/` | — | `{"status", "mode", "lightgbm_ready", "gpu"}` | Health check |
| `POST` | `/api/predict/roberta` | `{code_base64, model_type}` | `{success, score, final_score, tokens, attrs, chunks, ...}` | RoBERTa Ensemble + LIG |
| `POST` | `/api/predict/lightgbm` | `{code_base64}` | `{success, score}` | LightGBM prediction |
| `POST` | `/api/proxy/vllm` | `{payload}` | vLLM response | Proxy tới vLLM server nội bộ (port 8001) |

#### Luồng xử lý endpoint `/api/predict/roberta`:
```
Request → Decode Base64 → Tính MD5 Hash → Kiểm tra COLAB_CACHE
    ├── [CACHE HIT] → Trả về kết quả ngay (⚡)
    └── [CACHE MISS] → run_roberta_engine() → Lưu cache → Trả về (🚀)
```

#### Response Format (Phẳng)
Kết quả trả về được **trải phẳng** (`**result_data`), không bọc trong `details`, để Local Brain có thể đọc trực tiếp các trường `final_score`, `tokens`, `attrs`, `total_tokens`, v.v.

### 3.4. `feature_extractor.py` — Trích xuất đặc trưng C++

**Class `CppFeatureExtractorV8`:**
- Trích xuất **32 đặc trưng tĩnh** từ mã nguồn C++ bằng regex + thư viện `lizard`
- Các nhóm đặc trưng:
  - Style (empty_line_ratio, avg_line_length, tab_vs_space_ratio, ...)
  - Naming (avg_identifier_length, single_char_var_ratio, ...)
  - Complexity (avg_cyclomatic_complexity, num_functions, ...)
  - Structure (comment_density, template_usage, ...)
- **Chọn 20 đặc trưng** quan trọng nhất (theo `final_features.pkl` từ LightGBM feature selection)

### 3.5. `hybrid_evaluator.py` — LightGBM + Fusion

**Class `HybridEvaluator` (Singleton):**
- Load model LightGBM (`LightGBM_Regulated.pkl`), scaler (`scaler.pkl`), feature names (`final_features.pkl`)
- `predict_lgbm(code_text)`:
  1. Trích xuất 32 features → Scale → Chọn 20 features → Predict probability
- `fuse_scores(bert_score, lgbm_score)`:
  - Công thức: `fused = α × bert_score + (1-α) × lgbm_score` (α=0.48)
  - Fallback: Nếu LightGBM fail → trả về bert_score

### 3.6. `bootstrap.py` — One-Command Setup

Chạy `!python bootstrap.py` trên Colab sẽ tự động:
1. Cài `uv` (package manager nhanh) → Cài dependencies từ `requirements.txt`
2. Mount Google Drive (chứa model weights)
3. Kill processes cũ (ngrok, port 8000, 8001)
4. Khởi động **vLLM Server** (Qwen 2.5 Coder 7B FP16) trên port 8001
5. Đợi vLLM sẵn sàng (2-3 phút)
6. Khởi tạo `ModelManager` → Load RoBERTa + LightGBM
7. Tạo FastAPI app → Bật ngrok tunnel → In ra URL

## 4. End-to-End Data Flow

```
1. [Local] User gửi code C++ → Local Brain (localhost:8080)
2. [Local] LangGraph Router: Qwen classify "OOP" / "NORMAL"
3. [Local → Colab] Gọi API song song:
   ├── POST /api/predict/roberta  →  RoBERTa 5-fold + LIG (8-15s)
   ├── POST /api/predict/lightgbm →  LightGBM (0.5s)
   └── POST /api/proxy/vllm       →  Perplexity calculation (2-3s)
4. [Colab → Local] Trả về JSON phẳng (scores, tokens, attrs)
5. [Local] Orchestrator:
   ├── Fusion: α×BERT + (1-α)×LGBM = final_score
   ├── Render HTML heatmap (XAI visualization)
   ├── Judge: Kiểm tra ambiguity → retry nếu cần
   └── Critique: Gọi vLLM proxy để sinh nhận xét
6. [Local] Trả về Gold Metadata JSON cho user
```

**Tổng thời gian:** ~15-25s (lần đầu) | ~0.01s (cache hit)

## 5. Hệ thống Cache 2 lớp

| Lớp | Vị trí | Biến | Mục đích |
|-----|--------|------|----------|
| **Lớp 1** | Local Brain | `RESULT_CACHE` (main.py) | Tránh gọi lên Colab nếu đã có kết quả cuối |
| **Lớp 2** | Colab Worker | `COLAB_CACHE` (server.py) | Tránh chạy lại GPU nếu Local restart nhưng Colab vẫn sống |

**Đồng bộ Hash:** Cả hai lớp dùng cùng thuật toán `md5(f"bert_{model_type}_{decoded_code}")` trên nội dung code đã `decode() → replace('\\r\\n', '\\n') → strip()`.

## 6. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|-------|
| 1 | Colab session timeout (90 phút idle) | User re-run `bootstrap.py`. Local Brain hiển thị error khi gọi API |
| 2 | Colab GPU không khả dụng | Fallback CPU mode (chậm ~10x) |
| 3 | ngrok tunnel bị đổi URL | Cập nhật `NGROK_URL` trong file `.env` ở Local |
| 4 | Google Drive disconnect | Re-mount drive, reload model weights |
| 5 | VRAM OOM (>16GB T4) | Giảm `ENGINE_BATCH_SIZE`, `LIG_N_STEPS` trong config.py |
| 6 | RoBERTa trả về 404 | Kiểm tra thụt lề (indentation) trong server.py |
| 7 | Local Brain restart | Lớp 1 cache mất, nhưng Lớp 2 (Colab) vẫn giữ → không tốn GPU |
| 8 | Cùng code gửi lại | Cache hit ở Lớp 1 (instant) hoặc Lớp 2 (không chạy lại GPU) |

## 7. Migration Path: Colab → Production

```
Phase 1 (Hiện tại):  Colab GPU (free) + ngrok → Local Brain (localhost)
Phase 2 (Scale):      MicroK8s local GPU + Cloudflare Tunnel
                      → scripts/ giữ nguyên, chỉ thay server.py
                        (bỏ ngrok, thêm K8s health checks)
                      → Dùng ONNX Runtime thay PyTorch (2-5x faster)
```

Scripts trong `scripts/` được thiết kế **portable** — chạy được cả trên Colab lẫn MicroK8s mà không cần sửa logic AI.
