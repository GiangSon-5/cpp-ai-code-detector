"""
config.py — Cấu hình tập trung cho Colab Runtime (GPU Worker)

Chứa device detection, API keys, model paths, hyperparameters.
Wrap từ extract_1.py §1 (lines 31-110)
"""

import os
import sys
import json
import warnings
import logging
from datetime import datetime

import torch

warnings.filterwarnings("ignore")

# ==================================================================================
# LOGGING — Deep Logging 2-Session Rotation (Colab version)
# ==================================================================================
# On Colab, we use a simplified logger that writes to /content/logs/
# and rotates between current.jsonl and previous.jsonl

LOG_DIR = os.environ.get("COLAB_LOG_DIR", "/content/logs")
os.makedirs(LOG_DIR, exist_ok=True)

_LOG_CURRENT = os.path.join(LOG_DIR, "colab_current.jsonl")
_LOG_PREVIOUS = os.path.join(LOG_DIR, "colab_previous.jsonl")


def _rotate_logs():
    """Rotate current → previous on startup."""
    if os.path.exists(_LOG_CURRENT):
        try:
            if os.path.exists(_LOG_PREVIOUS):
                os.remove(_LOG_PREVIOUS)
            os.rename(_LOG_CURRENT, _LOG_PREVIOUS)
        except OSError:
            pass


def colab_log(level: str, module: str, function: str = "",
              message: str = "", **kwargs):
    """Write a JSONL log entry (Colab deep logging)."""
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
        pass  # Logging should never crash the worker


# Rotate on import
_rotate_logs()

# ==================================================================================
# DEVICE
# ==================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
print(f"🖥️ Device: {DEVICE} ({_gpu_name})")
colab_log("info", "config", "init", f"Device: {DEVICE} ({_gpu_name})")

# ==================================================================================
# API KEYS — Đọc từ Colab Secrets (userdata) hoặc env vars
# ==================================================================================
def _get_key(key_name, default=""):
    """Lấy key từ Colab Secrets → OS env → default"""
    try:
        from google.colab import userdata
        val = userdata.get(key_name)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key_name, default)

NGROK_TOKEN = _get_key("NGROK_TOKEN", "")

# ==================================================================================
# MODEL PATHS — Google Drive
# ==================================================================================
PATH_OOP = "/content/drive/MyDrive/LVTN: AI code detection/My_AI_Models/C++_OOP_detection"
PATH_NORMAL = "/content/drive/MyDrive/LVTN: AI code detection/My_AI_Models/C++_detection"
PATH_SAVED_MODELS = "/content/drive/MyDrive/LVTN: AI code detection/Saved_Models/"

# ==================================================================================
# HYPERPARAMETERS
# ==================================================================================
DEFAULT_THRESHOLD = 0.5
FUSION_ALPHA      = 0.48      # Trọng số cho RoBERTa (LightGBM sẽ là 1 - alpha)
MAX_LEN           = 510       # Max token length per chunk (trừ CLS/SEP)
STRIDE            = 256       # Sliding window stride
LIG_N_STEPS       = 20        # Layer Integrated Gradients steps
LIG_BATCH_SIZE    = 4         # Internal batch size cho LIG
ENGINE_BATCH_SIZE = 4         # Batch size khi xử lý chunks
MAX_CACHE_SIZE    = 100       # Số lượng kết quả cache tối đa
