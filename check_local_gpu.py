#!/usr/bin/env python3
"""
check_local_gpu.py — Kiểm tra nhanh cấu hình Local GPU Server trước khi chạy.

Chạy:
    conda activate detection_ai
    python check_local_gpu.py
"""

import os
import sys

print("=" * 65)
print("🔍  LOCAL GPU SERVER — Pre-flight Check")
print("=" * 65)

# 1. Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅  .env loaded")
except ImportError:
    print("⚠️   python-dotenv chưa cài — đọc trực tiếp từ env vars")

# 2. Kiểm tra biến môi trường
PATH_OOP    = os.environ.get("LOCAL_MODEL_PATH_OOP",    "")
PATH_NORMAL = os.environ.get("LOCAL_MODEL_PATH_NORMAL", "")
PATH_SAVED  = os.environ.get("LOCAL_SAVED_MODELS_PATH", "")
API_KEY     = os.environ.get("LOCAL_GPU_API_KEY",       "local-gpu-secret-key")
PORT        = os.environ.get("LOCAL_GPU_PORT",          "8002")

print(f"\n{'─' * 65}")
print("📂  Model Paths:")
print(f"   LOCAL_MODEL_PATH_OOP    = {PATH_OOP!r}")
print(f"   LOCAL_MODEL_PATH_NORMAL = {PATH_NORMAL!r}")
print(f"   LOCAL_SAVED_MODELS_PATH = {PATH_SAVED!r}")
print(f"   LOCAL_GPU_API_KEY       = {API_KEY!r}")
print(f"   LOCAL_GPU_PORT          = {PORT}")

# 3. Kiểm tra paths tồn tại
errors = []
for label, path in [("OOP", PATH_OOP), ("NORMAL", PATH_NORMAL)]:
    if not path:
        errors.append(f"LOCAL_MODEL_PATH_{label} chưa được set trong .env")
        print(f"\n❌  [{label}] Path chưa được cấu hình!")
        continue

    exists = os.path.isdir(path)
    print(f"\n{'✅' if exists else '❌'}  [{label}] {path}")

    if exists:
        contents = os.listdir(path)
        print(f"    Contents: {contents[:10]}")

        # Kiểm tra xem có config.json (là 1 fold) hay có fold_* subdirs
        has_config = os.path.exists(os.path.join(path, "config.json"))
        fold_dirs  = [d for d in contents if d.startswith("fold_")]

        if has_config:
            print(f"    ✅ Đây là 1 fold trực tiếp (có config.json)")
        elif fold_dirs:
            print(f"    ✅ Tìm thấy {len(fold_dirs)} fold(s): {fold_dirs}")
        else:
            errors.append(f"[{label}] Không tìm thấy config.json hay fold_* trong {path}")
            print(f"    ❌ Không tìm thấy config.json hay thư mục fold_*!")
    else:
        errors.append(f"[{label}] Path không tồn tại: {path}")

# 4. LightGBM (optional)
print(f"\n{'─' * 65}")
print("📦  LightGBM (optional):")
if PATH_SAVED and os.path.isdir(PATH_SAVED):
    for fname in ["LightGBM_Regulated.pkl", "scaler.pkl", "final_features.pkl"]:
        fpath = os.path.join(PATH_SAVED, fname)
        exists = os.path.exists(fpath)
        print(f"   {'✅' if exists else '❌'}  {fname}")
else:
    print(f"   ⚠️  PATH_SAVED_MODELS không được set hoặc không tồn tại → LightGBM bị bỏ qua")

# 5. GPU
print(f"\n{'─' * 65}")
print("🖥️   GPU:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅  CUDA available: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ℹ️   VRAM: {total_mem:.1f} GB")
        if total_mem < 4.0:
            print(f"   ⚠️  Cảnh báo: VRAM < 4GB — có thể bị OOM khi load model!")
    else:
        print("   ⚠️  CUDA không khả dụng — sẽ chạy trên CPU (rất chậm)")
except ImportError:
    print("   ❌  PyTorch chưa cài!")
    errors.append("PyTorch chưa được cài đặt")

# 6. Dependencies
print(f"\n{'─' * 65}")
print("📦  Dependencies:")
deps = [
    ("fastapi",       "fastapi"),
    ("uvicorn",       "uvicorn"),
    ("transformers",  "transformers"),
    ("captum",        "captum"),
    ("torch",         "torch"),
    ("lightgbm",      "lightgbm"),
    ("joblib",        "joblib"),
    ("pandas",        "pandas"),
    ("lizard",        "lizard"),
]
for pkg_label, pkg_import in deps:
    try:
        __import__(pkg_import)
        print(f"   ✅  {pkg_label}")
    except ImportError:
        print(f"   ❌  {pkg_label} — chưa cài!")
        errors.append(f"{pkg_label} chưa được cài đặt")

# 7. Kết quả
print(f"\n{'=' * 65}")
if errors:
    print(f"❌  Phát hiện {len(errors)} lỗi:")
    for i, e in enumerate(errors, 1):
        print(f"   {i}. {e}")
    print("\n⚠️  Hãy sửa các lỗi trên trước khi chạy server!")
    sys.exit(1)
else:
    print("✅  Tất cả kiểm tra OK! Server sẵn sàng để khởi động.")
    print(f"\n🚀  Chạy lệnh sau để khởi động Local GPU Server:")
    print(f"    uvicorn src.local_gpu_server.main:app --host 0.0.0.0 --port {PORT}")
    print(f"\n    Hoặc:")
    print(f"    python -m src.local_gpu_server.main")
    print("=" * 65)
