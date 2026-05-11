Ran command: `conda activate detection_ai
python check_local_gpu.py`
Ran command: `conda activate detection_ai`
Ran command: `conda activate detection_ai`
Ran command: `conda activate detection_ai`
Viewed .env:6-26

Đây là toàn bộ lệnh đầy đủ cho cả 4 terminal:

---

### 🖥️ Terminal 1 — AI Orchestrator (FastAPI :8001)
```bash
conda activate detection_ai && cd /home/nguyenvannhi242/LVTN-main
uvicorn src.fastapi_service.main:app --port 8001
```

---

### 🌐 Terminal 2 — Web Frontend (Django :8000)
```bash
conda activate detection_ai && cd /home/nguyenvannhi242/LVTN-main
python manage.py runserver 8000
```

---

### ⚙️ Terminal 3 — Background ETL (Celery)
```bash
conda activate detection_ai && cd /home/nguyenvannhi242/LVTN-main
celery -A src.celery_workers.celery_app worker --loglevel=info
```

---

### 🚀 Terminal 4 — Local GPU Worker (**MỚI** :8002)
```bash
conda activate detection_ai && cd /home/nguyenvannhi242/LVTN-main
uvicorn src.local_gpu_server.main:app --host 0.0.0.0 --port 8002
```

---

> **Lưu ý:** Pre-flight check đã ✅ OK, `.env` đã đúng cấu hình. Khi Terminal 4 khởi động sẽ mất **~30-60s** để nạp model RoBERTa vào GPU — bình thường, chờ xuất hiện dòng `✅ [LOCAL GPU SERVER] READY` là xong.