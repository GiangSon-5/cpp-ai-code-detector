"""
local_gpu_server — Local GPU Worker (thay thế Colab trong môi trường thử nghiệm)

Chạy cùng API contract với Colab Worker (server.py) nhưng:
  - Không cần ngrok / Google Drive
  - Chỉ load 1 fold (tiết kiệm VRAM 4GB)
  - Bỏ qua vLLM / Qwen (không đủ VRAM)
  - Paths đọc từ biến môi trường LOCAL_MODEL_PATH_OOP / LOCAL_MODEL_PATH_NORMAL
"""
