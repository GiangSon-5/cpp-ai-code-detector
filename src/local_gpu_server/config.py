"""
config.py — Cấu hình tập trung cho Local GPU Server (chế độ thử nghiệm 4GB VRAM)

Khác với Colab Worker:
  - MODEL_PATH đọc từ biến môi trường LOCAL_MODEL_PATH_OOP / LOCAL_MODEL_PATH_NORMAL
  - Chỉ load 1 fold (tiết kiệm VRAM)
  - Không có ngrok / Google Drive
  - ENGINE_BATCH_SIZE = 1 (an toàn cho VRAM 4GB)
  - LIG_N_STEPS giảm xuống 10 (nhanh hơn, ít tốn VRAM hơn)
"""

import os
import json
import warnings
import logging
from datetime import datetime

import torch

warnings.filterwarnings("ignore")

# ==================================================================================
# LOGGING
# ==================================================================================
LOG_DIR = os.environ.get("LOCAL_GPU_LOG_DIR", "logs/local_gpu_server")
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_CURRENT  = os.path.join(LOG_DIR, "current.jsonl")
_LOG_PREVIOUS = os.path.join(LOG_DIR, "previous.jsonl")

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("local_gpu_server")


def _rotate_logs():
    """Rotate current → previous on startup."""
    if os.path.exists(_LOG_CURRENT):
        try:
            if os.path.exists(_LOG_PREVIOUS):
                os.remove(_LOG_PREVIOUS)
            os.rename(_LOG_CURRENT, _LOG_PREVIOUS)
        except OSError:
            pass


def gpu_log(level: str, module: str, function: str = "",
            message: str = "", **kwargs):
    """Write a JSONL log entry."""
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "module": module,
        "function": function,
        "message": message,
    }
    entry.update(kwargs)
    try:
        with open(_LOG_CURRENT, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass
    # Mirror ra stdout
    getattr(_logger, level.lower(), _logger.info)(
        f"[{module}:{function}] {message}"
        + (f" | {kwargs}" if kwargs else "")
    )


_rotate_logs()

# ==================================================================================
# DEVICE
# ==================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU only"
print(f"🖥️  [LOCAL GPU] Device: {DEVICE} ({_gpu_name})")
gpu_log("info", "config", "init", f"Device: {DEVICE} ({_gpu_name})")

# ==================================================================================
# MODEL PATHS — đọc từ .env (LOCAL_MODEL_PATH_OOP / LOCAL_MODEL_PATH_NORMAL)
#
# Trên Windows model nằm ở:
#   C:\Users\LAPTOP\Desktop\Roberta model\fold_1_oop     → mount / copy sang Linux
#   C:\Users\LAPTOP\Desktop\Roberta model\fold_1_basic
#
# Trên Linux (máy này) đặt đường dẫn thực trong .env, ví dụ:
#   LOCAL_MODEL_PATH_OOP=/mnt/windows/Roberta model/fold_1_oop
#   LOCAL_MODEL_PATH_NORMAL=/mnt/windows/Roberta model/fold_1_basic
#
# Nếu không set, mặc định fallback sang thư mục local_models/ trong project
# ==================================================================================
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH_OOP = os.environ.get(
    "LOCAL_MODEL_PATH_OOP",
    os.path.join(_project_root, "local_models", "fold_1_oop"),
)
PATH_NORMAL = os.environ.get(
    "LOCAL_MODEL_PATH_NORMAL",
    os.path.join(_project_root, "local_models", "fold_1_basic"),
)
PATH_SAVED_MODELS = os.environ.get(
    "LOCAL_SAVED_MODELS_PATH",
    os.path.join(_project_root, "local_models", "saved_models"),
)

print(f"📂 [LOCAL GPU] OOP    path: {PATH_OOP}")
print(f"📂 [LOCAL GPU] NORMAL path: {PATH_NORMAL}")
gpu_log("info", "config", "init",
        "Model paths configured",
        oop_path=PATH_OOP, normal_path=PATH_NORMAL)

# ==================================================================================
# HYPERPARAMETERS — tuned cho 4GB VRAM + single-fold
# ==================================================================================
DEFAULT_THRESHOLD  = 0.5
FUSION_ALPHA       = 0.48     # Trọng số RoBERTa trong Hybrid Fusion
MAX_LEN            = 510      # Max token length per chunk (trừ CLS/SEP)
STRIDE             = 256      # Sliding window stride

# ⚠️ Giảm để fit 4GB VRAM
LIG_N_STEPS        = 10       # (Colab = 20) — giảm 50% thời gian + VRAM
LIG_BATCH_SIZE     = 1        # (Colab = 4)  — tuyệt đối an toàn cho 4GB
ENGINE_BATCH_SIZE  = 1        # (Colab = 4)  — 1 chunk tại một thời điểm

MAX_CACHE_SIZE     = 50       # Ít hơn Colab do RAM ít hơn

# API Key để Local Orchestrator gọi vào Local GPU Server
# Giữ đồng bộ với giá trị trong .env (API_KEY)
LOCAL_GPU_API_KEY  = os.environ.get("LOCAL_GPU_API_KEY", "local-gpu-secret-key")
LOCAL_GPU_PORT     = int(os.environ.get("LOCAL_GPU_PORT", "8002"))
