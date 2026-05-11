"""
main.py — Entry point cho Local GPU Server

Chạy bằng:
    python -m src.local_gpu_server.main
    hoặc:
    uvicorn src.local_gpu_server.main:app --port 8002

Biến môi trường cần thiết (trong .env hoặc export):
    LOCAL_MODEL_PATH_OOP     — Đường dẫn đến thư mục fold_1_oop
    LOCAL_MODEL_PATH_NORMAL  — Đường dẫn đến thư mục fold_1_basic
    LOCAL_GPU_API_KEY        — API key (default: local-gpu-secret-key)
    LOCAL_GPU_PORT           — Port (default: 8002)
    LOCAL_SAVED_MODELS_PATH  — (optional) Đường dẫn LightGBM .pkl files
"""

from .engine import LocalModelManager
from .server import create_app

# --- Tạo app tại module-level để uvicorn import được ---
_manager = LocalModelManager()
app = create_app(_manager)

if __name__ == "__main__":
    from .server import run_local_gpu_server
    run_local_gpu_server()
