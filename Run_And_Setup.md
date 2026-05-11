# 🚀 Run & Setup Guide — C++ AI Code Detector

> **Cập nhật:** 2026-04-29 | **Kiến trúc:** Smart Orchestrator (Local) + Dumb Worker (Colab)

---

## 📋 Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [BƯỚC 1 — Setup Colab GPU Worker](#3-bước-1--setup-colab-gpu-worker)
4. [BƯỚC 2 — Setup Local Brain](#4-bước-2--setup-local-brain)
5. [BƯỚC 3 — Kết nối Local ↔ Colab](#5-bước-3--kết-nối-local--colab)
6. [BƯỚC 4 — Chạy và Test](#6-bước-4--chạy-và-test)
7. [Troubleshooting](#7-troubleshooting)
8. [Tham chiếu tài liệu](#8-tham-chiếu-tài-liệu)

---

## 1. Tổng quan kiến trúc

Hệ thống gồm **2 node** chạy độc lập, giao tiếp qua HTTP/ngrok:

```
┌───────────────────────────────┐         ┌──────────────────────────────────┐
│  LOCAL BRAIN (Máy cá nhân)    │  HTTP   │  COLAB WORKER (Google Colab)     │
│  Smart Orchestrator           │ ──────► │  Dumb Worker (GPU)               │
│                               │  ngrok  │                                  │
│  • LangGraph Agent            │         │  • RoBERTa 5-fold Ensemble       │
│  • Cache Lớp 1                │         │  • LightGBM                      │
│  • HTML Heatmap Render        │         │  • vLLM (Qwen 2.5 Coder 7B)     │
│  • Fusion Score               │         │  • Cache Lớp 2                   │
│                               │         │                                  │
│  Port: localhost:8080         │         │  Port: ngrok → 8000              │
│  CPU only ✅                  │         │  GPU T4/L4 16GB ✅               │
└───────────────────────────────┘         └──────────────────────────────────┘
```

### Kiến trúc 2: Hybrid Local GPU + Colab Qwen (MỚI)
Dành cho máy có sẵn GPU Nvidia. Giải quyết hoàn toàn vấn đề rớt mạng và giảm tải cho Colab.

```
┌───────────────────────────────┐         ┌──────────────────────────────────┐
│  LOCAL GPU AGENT (Máy cá nhân)│  HTTP   │  COLAB QWEN WORKER (Chỉ tính PPL)│
│  Port: localhost:8081         │ ──────► │  Port: ngrok → 8000              │
│                               │  ngrok  │                                  │
│  • RoBERTa 10-fold (GPU Local)│         │  • vLLM (Qwen 2.5 Coder)         │
│  • LightGBM (CPU Local)       │         │  • Tốc độ tính PPL siêu nhanh    │
│  • LangGraph + HTML Render    │         │  • Ít tốn RAM, không rớt mạng    │
│  • Gemini API (Critique)      │         │                                  │
└───────────────────────────────┘         └──────────────────────────────────┘
```

---

## 2. Yêu cầu hệ thống

### Máy Local (chạy Local Brain)

| Thành phần | Yêu cầu |
|-----------|---------|
| OS | Linux / WSL2 (Windows) / macOS |
| Python | 3.10+ (đã test trên 3.13) |
| RAM | ≥ 4GB |
| GPU | **Không cần** |
| Mạng | Có internet (để gọi API lên Colab) |

### Google Colab (chạy GPU Worker)

| Thành phần | Yêu cầu |
|-----------|---------|
| Runtime | GPU (T4 tối thiểu, L4 recommended) |
| Google Drive | Chứa model weights (~2GB) |
| ngrok account | Free tier đủ dùng |

### Model weights cần có trên Google Drive

```
/content/drive/MyDrive/LVTN: AI code detection/
├── My_AI_Models/
│   ├── C++_OOP_detection/       # 5 thư mục fold_0 → fold_4 (RoBERTa OOP)
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   └── threshold.json
│   └── C++_detection/           # 5 thư mục fold_0 → fold_4 (RoBERTa Normal)
│       ├── fold_0/ ... fold_4/
│       └── threshold.json
└── Saved_Models/
    ├── LightGBM_Regulated.pkl   # Model LightGBM
    ├── scaler.pkl               # StandardScaler
    └── final_features.pkl       # 20 features đã chọn
```

---

## 3. BƯỚC 1 — Setup Colab GPU Worker

### Cách 1: One-command (Khuyên dùng)

1. Mở Google Colab → **Runtime → Change runtime type → GPU (T4)**
2. Upload thư mục `src/colab_runtime/` lên Colab (hoặc clone repo)
3. Chạy 1 cell duy nhất:

```python
%cd /content
# Clone hoặc upload repo, sau đó:
%cd src/colab_runtime
!python bootstrap.py
```

Script `bootstrap.py` sẽ tự động:
- ✅ Cài đặt thư viện (qua `uv`, nhanh gấp 10x pip)
- ✅ Mount Google Drive
- ✅ Kill processes cũ
- ✅ Khởi động vLLM Server (Qwen 2.5 Coder 7B) — **mất ~2-3 phút**
- ✅ Load RoBERTa 5-fold + LightGBM
- ✅ Bật ngrok tunnel → **In ra URL**

### Cách 2: Thủ công (từng bước)

```python
# Cell 1: Cài đặt
!pip install -q uv
!uv pip install -q torch transformers captum fastapi uvicorn pyngrok nest_asyncio vllm lizard lightgbm joblib pandas scikit-learn numpy httpx --system

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Start vLLM (background)
import subprocess
subprocess.Popen("python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-7B-Instruct --dtype half --max-model-len 2048 --gpu-memory-utilization 0.80 --port 8001 > /content/vllm.log 2>&1", shell=True)
# ⏳ Đợi 2-3 phút cho vLLM load xong

# Cell 4: Start GPU Worker
import sys
sys.path.insert(0, '/content')
from scripts.engine import ModelManager
from scripts.server import create_app, start_ngrok, run_server

manager = ModelManager()
app = create_app(manager)
public_url = start_ngrok(port=8000)
run_server(app, port=8000)

print(f"🔗 URL: {public_url}")
# → Copy URL này cho bước tiếp theo
```

### Kết quả mong đợi
```
🚀 ═══════════════════════════════════════════════════
   TÊN MIỀN NGROK CHO GPU WORKER:
   https://xxxx-xxxx-xxxx.ngrok-free.dev     ← COPY CÁI NÀY
   (API Key: colab-secret-key-123)
═══════════════════════════════════════════════════════
```

---

## 3b. BƯỚC 1 (Cách 2) — Setup Colab Qwen Worker (Cho Kiến trúc Hybrid)

Nếu máy bạn có GPU Nvidia và muốn chạy Kiến trúc 2:
1. Mở Colab (GPU T4).
2. Tải/upload thư mục `src/colab_qwen_worker/` lên.
3. Chạy `bootstrap_qwen.py`:
```python
%cd /content/src/colab_qwen_worker
!python bootstrap_qwen.py
```
> Phiên bản này **KHÔNG** tải RoBERTa hay LightGBM, nên Colab chạy siêu mượt và không bị đầy RAM. Nó sẽ in ra 1 cái `NGROK_URL`.

---

## 4. BƯỚC 2 — Setup Local Brain

### Cài đặt dependencies

```bash
# Tạo virtual environment (khuyên dùng)
conda create -n detection_ai python=3.13 -y
conda activate detection_ai

# Cài đặt thư viện
pip install -r src/local_agent/requirements.txt
```

### Dependencies cần thiết

```
fastapi        # Web framework
uvicorn        # ASGI server
httpx          # Async HTTP client
langgraph      # Agent workflow
langchain      # LangGraph dependency
langchain-core # LangGraph dependency
numpy          # Numerical computing
```

---

## 5. BƯỚC 3 — Kết nối Local ↔ Colab

### Cập nhật ngrok URL

Lấy URL ngrok từ **Bước 1**, rồi cập nhật vào file `src/local_agent/remote_client.py` dòng 8:

```python
# Thay URL này bằng URL ngrok mới từ Colab
NGROK_URL = os.getenv("NGROK_URL", "https://xxxx-xxxx-xxxx.ngrok-free.dev").rstrip('/')
```

**Hoặc** dùng biến môi trường (không cần sửa code):

```bash
export NGROK_URL="https://xxxx-xxxx-xxxx.ngrok-free.dev"
```

### Kiểm tra kết nối

```bash
# Kiểm tra Colab Worker có online không
curl https://xxxx-xxxx-xxxx.ngrok-free.dev/
```

Kết quả mong đợi:
```json
{"status": "Online", "mode": "Microservices GPU Worker", "lightgbm_ready": true, "gpu": "Tesla T4"}
```

---

## 6. BƯỚC 4 — Chạy và Test

### Khởi động Local Brain

```bash
cd ~/detection\ AI\ code\ c
python -m src.local_agent.main
```

Kết quả mong đợi:
```
🧠 Khởi động Local Brain tại port 8080...
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
```

> **Lưu ý:** Nếu port 8080 đang bị chiếm, server sẽ **tự động kill** process cũ và khởi động lại. Bạn không cần gõ `pkill` thủ công.

### Chạy Test

Mở terminal mới:

```bash
cd ~/detection\ AI\ code\ c
python src/local_agent/test_colab_logic.py
```

Kết quả mong đợi:
```
🚀 Đang gửi mã nguồn lên http://127.0.0.1:8080/api/analyze ...

✅ PHÂN TÍCH THÀNH CÔNG!
--------------------------------------------------
📦 FULL GOLD METADATA JSON:
{
  "model_used": "C++ Normal Model",
  "classification": "NORMAL",
  "final_score": 0.3053,
  "bert_score": 0.6168,          ← Phải KHÁC 0.5
  "lgbm_score": 0.0177,
  "total_tokens": 110,           ← Phải > 0
  "total_chunks": 1,             ← Phải > 0
  "top_ai_signals": ["'++'", "'in'", ...],
  "perplexity": 6.27,
  "final_pred": "HUMAN WRITTEN",
  ...
}
```

### Kiểm tra nhanh (Checklist) ✅

| Mục | Đúng | Sai (cần fix) |
|-----|------|---------------|
| `bert_score` | `≠ 0.5` (VD: 0.6168) | `= 0.5` → Colab trả 404 hoặc remote_client parse sai |
| `total_tokens` | `> 0` (VD: 110) | `= 0` → Colab không trả data phẳng |
| `lgbm_score` | `≠ bert_score` | `= bert_score` → LightGBM API lỗi |
| `top_ai_signals` | Có danh sách token | `[]` → Colab không trả attrs |
| `perplexity` | `> 0` | `= 0` → vLLM server chưa sẵn sàng |

### Chạy Kiến trúc Hybrid (Local GPU Agent)

Nếu máy bạn có GPU Nvidia, hãy chạy bản `local_gpu_agent` thay vì `local_agent`:

1. Download các thư mục Model (OOP, Normal, Saved_Models) về máy tính cá nhân.
2. Sửa đường dẫn `PATH_OOP`, `PATH_NORMAL` trong `src/local_gpu_agent/config.py` trỏ tới thư mục vừa tải.
3. Mở file `.env` (hoặc cấu hình biến môi trường) thêm `GEMINI_API_KEY=AIzaSy...`.
4. Bật server:
```bash
python -m src.local_gpu_agent.main
```
5. Chạy test:
```bash
python src/local_gpu_agent/test_hybrid_logic.py
```

---

## 7. Troubleshooting

### ❌ `ERROR: [Errno 98] address already in use`

**Nguyên nhân:** Port 8080 bị chiếm bởi phiên cũ.

**Cách fix:** Server đã tự động xử lý. Nếu vẫn lỗi:
```bash
fuser -k 8080/tcp
python -m src.local_agent.main
```

---

### ❌ `bert_score = 0.5` và `total_tokens = 0`

**Nguyên nhân:** Local Brain không nhận được dữ liệu từ Colab.

**Kiểm tra:**
1. Colab Worker có đang chạy? → Kiểm tra cell Colab
2. ngrok URL đúng chưa? → `curl <ngrok_url>/`
3. Endpoint RoBERTa có trả 404? → Kiểm tra thụt lề trong `server.py`
4. `remote_client.py` có đang tìm key `details`? → Phải dùng `return data` (không phải `data.get("details", {})`)

---

### ❌ `Error (RoBERTa Data): 404 Not Found`

**Nguyên nhân:** Endpoint `/api/predict/roberta` không được đăng ký trên Colab.

**Cách fix:** Kiểm tra file `src/colab_runtime/scripts/server.py`:
- Đảm bảo `@app.post("/api/predict/roberta")` thụt lề đúng bên trong hàm `create_app()`
- Đảm bảo không có duplicate code blocks

---

### ❌ `Perplexity = 0.0`

**Nguyên nhân:** vLLM server chưa sẵn sàng.

**Cách fix:**
```bash
# Trên Colab, kiểm tra log:
!cat /content/vllm.log | tail -20

# Hoặc kiểm tra vLLM đã sẵn sàng:
!curl http://localhost:8001/v1/models
```

---

### ❌ `Connection refused` hoặc `Network Error`

**Nguyên nhân:** Colab session đã hết hạn hoặc ngrok tunnel bị ngắt.

**Cách fix:**
1. Quay lại Colab
2. Chạy lại `!python bootstrap.py`
3. Copy ngrok URL mới → Cập nhật vào Local

---

### ❌ `LightGBM prediction failed`

**Nguyên nhân:** Thiếu file `.pkl` trên Google Drive.

**Cách fix:** Kiểm tra 3 file sau có tồn tại:
```
/content/drive/MyDrive/LVTN: AI code detection/Saved_Models/
├── LightGBM_Regulated.pkl    ← Model
├── scaler.pkl                ← Scaler
└── final_features.pkl        ← Feature names
```

---

## 8. Tham chiếu tài liệu

| Tài liệu | Đường dẫn | Nội dung |
|-----------|----------|----------|
| **Local Agent SPEC** | `src/local_agent/local_agent_SPEC.md` | Đặc tả kỹ thuật Local Brain |
| **Local Agent SRS** | `src/local_agent/local_agent_SRS.md` | Đặc tả nghiệp vụ Local Brain |
| **Colab Runtime SPEC** | `src/colab_runtime/colab_runtime_SPEC.md` | Đặc tả kỹ thuật GPU Worker |
| **Colab Runtime SRS** | `src/colab_runtime/colab_runtime_SRS.md` | Đặc tả nghiệp vụ GPU Worker |
| **Legacy Code** | `extract_1.py`, `extracted_with_markdown.py` | Code gốc (tham khảo logic) |
| **Metadata Plan** | `metadata_implementation_plan.md` | Gold Metadata Schema |

---

## Tóm tắt quy trình (Quick Start)

```
1. Colab:   Mở notebook → Chạy !python bootstrap.py → Copy ngrok URL
2. Local:   Cập nhật NGROK_URL → python -m src.local_agent.main
3. Test:    python src/local_agent/test_colab_logic.py
4. Verify:  bert_score ≠ 0.5, total_tokens > 0
```

**Thời gian setup:** ~5-8 phút (Colab) + ~1 phút (Local)
**Thời gian inference:** ~15-25s (lần đầu) | ~0.01s (cache hit)
