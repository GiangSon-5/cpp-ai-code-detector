# Local GPU Server — Hướng dẫn Sử dụng

> Thay thế Colab Worker cho môi trường thử nghiệm với GPU local 4GB VRAM.  
> Chạy trên port **8002**, cùng API contract với Colab Worker.

## 📋 So sánh: Colab Worker vs Local GPU Server

| Tính năng | Colab Worker | Local GPU Server |
|---|---|---|
| **RoBERTa folds** | 5 folds | **1 fold** (tiết kiệm VRAM) |
| **LIG n_steps** | 20 | **10** (nhanh hơn 2x) |
| **Engine batch size** | 4 | **1** (an toàn 4GB) |
| **vLLM / Qwen** | ✅ (cần 14GB VRAM) | ❌ **Disabled** (501 response) |
| **ngrok** | ✅ | ❌ Không cần |
| **Google Drive** | ✅ | ❌ Không cần |
| **Perplexity** | ✅ | ❌ → 0.0 (bỏ qua) |
| **Critique (Gemini)** | ✅ | ✅ (nếu GEMINI_API_KEY set) |
| **LightGBM** | ✅ | ✅ (nếu có pkl files) |
| **Port** | 8000 (qua ngrok) | **8002** |

---

## 🗂️ Cấu trúc thư mục model

Model RoBERTa bạn đã tải về nằm tại:
```
C:\Users\LAPTOP\Desktop\Roberta model\
├── fold_1_oop\        ← OOP model (khoảng 500MB)
│   ├── config.json
│   ├── pytorch_model.bin  (hoặc model.safetensors)
│   ├── tokenizer_config.json
│   └── ...
└── fold_1_basic\      ← Normal/Basic model (khoảng 500MB)
    ├── config.json
    └── ...
```

---

## 🚀 Hướng dẫn Khởi động

### Bước 1: Cấu hình đường dẫn model trong `.env`

```bash
# Nếu dùng WSL2 (Linux), Windows drive C: được mount tại /mnt/c
LOCAL_MODEL_PATH_OOP=/mnt/c/Users/LAPTOP/Desktop/Roberta model/fold_1_oop
LOCAL_MODEL_PATH_NORMAL=/mnt/c/Users/LAPTOP/Desktop/Roberta model/fold_1_basic

# Nếu bạn copy model sang máy Linux:
LOCAL_MODEL_PATH_OOP=/home/nguyenvannhi242/LVTN-main/local_models/fold_1_oop
LOCAL_MODEL_PATH_NORMAL=/home/nguyenvannhi242/LVTN-main/local_models/fold_1_basic

# Chuyển FASTAPI_AI_URL sang Local GPU Server
FASTAPI_AI_URL=http://localhost:8002

# API Key (phải khớp với cài đặt server)
LOCAL_GPU_API_KEY=local-gpu-secret-key
LOCAL_GPU_PORT=8002
```

### Bước 2: Kiểm tra cấu hình

```bash
conda activate detection_ai
python check_local_gpu.py
```

Output mong đợi:
```
✅  .env loaded
✅  [OOP] /path/to/fold_1_oop
    ✅ Đây là 1 fold trực tiếp (có config.json)
✅  [NORMAL] /path/to/fold_1_basic
    ✅ Đây là 1 fold trực tiếp (có config.json)
✅  CUDA available: NVIDIA GeForce RTX ...  (4.0 GB)
✅  Tất cả kiểm tra OK!
```

### Bước 3: Khởi động Local GPU Server (Terminal 4)

```bash
conda activate detection_ai
cd /home/nguyenvannhi242/LVTN-main
uvicorn src.local_gpu_server.main:app --host 0.0.0.0 --port 8002
```

Hoặc dùng Python module:
```bash
python -m src.local_gpu_server.main
```

### Bước 4: Khởi động các service còn lại (như bình thường)

```bash
# Terminal 1 — AI Orchestrator (FastAPI :8001)
uvicorn src.fastapi_service.main:app --port 8001

# Terminal 2 — Web Frontend (Django :8000)  
python manage.py runserver 8000

# Terminal 3 — Background ETL (Celery)
celery -A src.celery_workers.celery_app worker

# Terminal 4 — Local GPU Worker (mới) :8002
uvicorn src.local_gpu_server.main:app --host 0.0.0.0 --port 8002
```

---

## 🔄 Luồng xử lý với Local GPU Server

```
Browser (:8000)
    │
    ▼
Django → FastAPI Orchestrator (:8001)
    │
    │  POST /api/predict/roberta
    ▼
Local GPU Server (:8002)        ← THAY THẾ COLAB
    │  Load 1 fold RoBERTa
    │  LIG (n_steps=10)
    │  Engine batch size = 1
    │  Trả về: tokens, attrs, chunks, final_score
    │
    │  POST /api/proxy/vllm → 501 (disabled)
    │  → Perplexity = 0.0 (bỏ qua)
    │
    │  Critique → Gemini API trực tiếp (nếu có key)
    ▼
FastAPI Orchestrator
    │  Tổng hợp kết quả
    │  Render HTML heatmap
    │  SSE stream về Browser
    ▼
result.html
```

---

## ⚡ Tối ưu VRAM 4GB

| Cơ chế | Mô tả |
|---|---|
| **1 fold only** | Chỉ load 1 model thay vì 5 → tiết kiệm ~80% VRAM |
| **CPU offload** | Model được move về CPU ngay sau mỗi batch |
| **Batch size = 1** | 1 chunk tại 1 thời điểm, không pad nhiều sequences |
| **LIG steps = 10** | Giảm từ 20 xuống 10 → nhanh hơn 2x, ít VRAM hơn |
| **Cache FIFO** | 50 results được cache để tránh inference lại |

---

## 🐛 Troubleshooting

### VRAM Out of Memory
```bash
# Giảm thêm trong src/local_gpu_server/config.py:
LIG_N_STEPS = 5     # (default: 10)
```

### Model path không tìm thấy
```bash
# Kiểm tra path trong WSL2:
ls "/mnt/c/Users/LAPTOP/Desktop/Roberta model/fold_1_oop"
# Phải thấy: config.json, pytorch_model.bin, tokenizer_config.json

# Nếu cần copy sang Linux:
mkdir -p ~/LVTN-main/local_models
cp -r "/mnt/c/Users/LAPTOP/Desktop/Roberta model/fold_1_oop" ~/LVTN-main/local_models/
cp -r "/mnt/c/Users/LAPTOP/Desktop/Roberta model/fold_1_basic" ~/LVTN-main/local_models/
```

### Lỗi 403 Forbidden
```bash
# Đảm bảo API key khớp nhau:
# Trong .env: LOCAL_GPU_API_KEY=local-gpu-secret-key
# Server dùng X-API-Key header với cùng value
```

### vLLM / Perplexity trả về 0.0
Đây là hành vi bình thường trong Local GPU Server. Perplexity được set = 0.0 và không ảnh hưởng đến kết quả phát hiện (chỉ ảnh hưởng đến Judge node's PPL conflict check).

---

## 📊 Kỳ vọng hiệu năng

| Code size | Tokens | Thời gian (4GB GPU) |
|---|---|---|
| Nhỏ (< 100 lines) | ~300 | 8-15s |
| Trung bình (100-300 lines) | ~800-2500 | 15-35s |
| Lớn (> 300 lines) | >2500 | 35-90s |

> **Lưu ý:** Lần đầu chạy chậm hơn (load model ~30s). Từ lần thứ 2 với cùng code sẽ instant (cache hit).
