"""
config.py — Cấu hình tập trung cho Colab Runtime
Chứa API keys, device, paths, constants.
Wrap từ extract_1.py §1 (lines 31-110)
"""
import os
import warnings
import torch

warnings.filterwarnings("ignore")

# ==================================================================================
# DEVICE
# ==================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

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

NGROK_TOKEN     = _get_key("NGROK_TOKEN", "")

# ==================================================================================
# MODEL PATHS — Google Drive
# ==================================================================================
PATH_OOP    = "/content/drive/MyDrive/LVTN: AI code detection/My_AI_Models/C++_OOP_detection"
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
