"""
bootstrap.py — One-command setup for C++ AI Code Detector on Google Colab

Usage:
    !python bootstrap.py

Sequence:
    1. Install uv → install requirements.txt
    2. Mount Google Drive
    3. Kill old processes (ngrok, port 8000/8001)
    4. Start vLLM Server (Qwen 2.5 Coder 7B FP16) on port 8001
    5. Wait for vLLM ready (~2-3 minutes)
    6. Load RoBERTa + LightGBM models
    7. Start FastAPI server + ngrok tunnel
"""

import os
import sys
import time
import subprocess
import urllib.request


def run_command(command):
    """Execute a shell command with output streaming."""
    print(f"Executing: {command}")
    process = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"⚠️ Command returned non-zero: {command}")


def is_vllm_ready(port=8001):
    """Check if vLLM server is responding."""
    try:
        code = urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2).getcode()
        return code == 200
    except Exception:
        return False


def setup_all():
    """Main setup orchestrator."""
    t0 = time.time()
    print("🚀 Starting full setup for C++ AI Code Detector...")
    print(f"   Python: {sys.version}")
    print(f"   CWD: {os.getcwd()}")

    # 1. Install uv (ultra-fast package installer)
    print("\n⚡ Installing uv package manager...")
    run_command("pip install -q uv")

    # 2. Install dependencies using uv (10x faster than pip)
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_file):
        print(f"\n📦 Installing libraries from {req_file} (via uv)...")
        run_command(f"uv pip install -q -r {req_file} --system")
    else:
        print(f"⚠️ requirements.txt not found at {req_file}")

    # 3. Mount Google Drive
    print("\n📂 Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted.")
    except ImportError:
        print("⚠️ Not running on Colab — skipping Drive mount.")
        print("   Models must be available at configured paths.")
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
        print("   Please ensure you are running this on Google Colab.")
        return

    # 4. Kill old processes
    print("\n🛑 Cleaning up old processes...")
    run_command("killall ngrok 2>/dev/null")
    run_command("fuser -k 8000/tcp 2>/dev/null")
    run_command("fuser -k 8001/tcp 2>/dev/null")
    time.sleep(1)

    # 5. Start vLLM Server
    print("\n🧠 Starting vLLM Server (Qwen 2.5 Coder 7B FP16) on port 8001 in background...")
    vllm_cmd = (
        "python -m vllm.entrypoints.openai.api_server "
        "--model Qwen/Qwen2.5-Coder-7B-Instruct "
        "--dtype half "
        "--max-model-len 2048 "
        "--gpu-memory-utilization 0.80 "
        "--port 8001 > /content/vllm.log 2>&1"
    )
    subprocess.Popen(vllm_cmd, shell=True)

    print("⏳ Waiting for vLLM server to be ready (~2-3 minutes for 14GB model)...")
    ready = False
    for i in range(60):  # max 60 × 5s = 300s = 5 phút
        time.sleep(5)
        if is_vllm_ready():
            ready = True
            break
        if i % 6 == 0:  # Every 30s
            elapsed = i * 5
            print(f"  ... still loading ({elapsed}s elapsed). Check /content/vllm.log for details.")

    if ready:
        print("✅ vLLM Server is UP and running!")
    else:
        print("❌ vLLM Server failed to start! Check /content/vllm.log for errors.")
        print("   Continuing anyway — vLLM proxy endpoint will return errors.")

    # 6. Verify model paths
    from scripts.config import PATH_OOP, PATH_NORMAL, colab_log
    print(f"\n🔍 Verifying model paths...")
    print(f"   OOP path exists: {os.path.exists(PATH_OOP)} → {PATH_OOP}")
    print(f"   Normal path exists: {os.path.exists(PATH_NORMAL)} → {PATH_NORMAL}")

    if not os.path.exists(PATH_OOP) and not os.path.exists(PATH_NORMAL):
        print("⚠️ Warning: No model paths found. Please check your Google Drive structure.")
        print("   Expected: /content/drive/MyDrive/LVTN: AI code detection/My_AI_Models/")

    # 7. Initialize and Start
    print("\n⚙️ Initializing Agent and Server...")
    sys.path.insert(0, '/content')

    try:
        from scripts.engine import ModelManager
        from scripts.server import create_app, start_ngrok, run_server

        manager = ModelManager()
        app = create_app(manager)

        # Start ngrok
        public_url = start_ngrok(port=8000)

        # Start server
        run_server(app, port=8000)

        elapsed = time.time() - t0
        print(f"\n✅ SERVER IS UP AND RUNNING! (Total setup: {elapsed:.0f}s)")
        print(f"🔗 Public URL: {public_url}")
        print(f"\n📋 Hướng dẫn:")
        print(f"   1. Copy URL trên vào file .env trên máy local:")
        print(f"      NGROK_URL={public_url}")
        print(f"   2. Giữ cell này chạy để duy trì server")
        print(f"   3. Nếu bị timeout, chạy lại bootstrap.py")
        colab_log("info", "bootstrap", "setup_all",
                  f"Setup complete in {elapsed:.0f}s",
                  public_url=public_url, total_time_s=round(elapsed, 1))

        # Keep alive loop
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user.")
            colab_log("info", "bootstrap", "setup_all", "Server stopped by user")

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        colab_log("error", "bootstrap", "setup_all", f"Startup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    setup_all()
