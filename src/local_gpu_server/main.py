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

import torch

# --- Limit VRAM to 4GB as requested ---
if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    fraction = 4.0 / total_vram
    if fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        print(f"⚠️  [LOCAL GPU] VRAM limited to 4GB (fraction {fraction:.4f})")

from .engine import LocalModelManager
from .server import create_app

# --- Tạo app tại module-level để uvicorn import được ---
_manager = LocalModelManager()
app = create_app(_manager)

if __name__ == "__main__":
    from .server import run_local_gpu_server
    run_local_gpu_server()
