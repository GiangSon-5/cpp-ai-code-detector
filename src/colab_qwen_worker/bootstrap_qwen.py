import os
import subprocess
import time
import sys

def run_cmd(cmd):
    print(f"⚙️ Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error running command: {cmd}")
        sys.exit(1)

def install_dependencies():
    print("🚀 [1/3] Cài đặt dependencies siêu nhẹ (vLLM & ngrok)...")
    run_cmd("pip install -q uv")
    run_cmd("uv pip install -q fastapi uvicorn pyngrok vllm nest_asyncio python-multipart pydantic --system")

def start_vllm():
    print("\n🚀 [2/3] Khởi động vLLM Server (Qwen 2.5 Coder 7B)...")
    # Kill old processes if any
    os.system("fuser -k 8001/tcp 2>/dev/null")
    os.system("fuser -k 8000/tcp 2>/dev/null")
    
    cmd = (
        "python -m vllm.entrypoints.openai.api_server "
        "--model Qwen/Qwen2.5-Coder-7B-Instruct "
        "--dtype half "
        "--max-model-len 2048 "
        "--gpu-memory-utilization 0.90 "
        "--port 8001 > /content/vllm.log 2>&1"
    )
    subprocess.Popen(cmd, shell=True)
    
    print("⏳ Đang đợi vLLM khởi động (khoảng 2-3 phút)...")
    # Tạm dừng 2 phút cho chắc ăn
    for i in range(120):
        if i % 10 == 0:
            print(f"   Đã đợi {i} giây...")
        time.sleep(1)
        
    print("✅ vLLM có vẻ đã sẵn sàng!")

def start_server():
    print("\n🚀 [3/3] Khởi động Proxy Server & Ngrok...")
    # Import sau khi đã cài đặt xong thư viện
    from server import start_ngrok, run_server, app
    
    public_url = start_ngrok(port=8000)
    print(f"\n=======================================================")
    print(f"🔗 TÊN MIỀN NGROK CỦA BẠN LÀ: {public_url}")
    print(f"📋 Copy link này dán vào .env của Local GPU Agent nhé!")
    print(f"=======================================================\n")
    
    run_server(app, port=8000)

if __name__ == "__main__":
    install_dependencies()
    start_vllm()
    start_server()
