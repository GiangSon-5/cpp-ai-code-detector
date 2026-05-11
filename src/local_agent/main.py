import os
import signal
import subprocess
import hashlib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from .local_orchestrator import local_agent_app

# --- Tự động giải phóng cổng 8080 ---
try:
    # Tìm PID đang chiếm cổng 8080
    result = subprocess.check_output(["lsof", "-t", "-i:8080"], stderr=subprocess.STDOUT)
    for pid in result.decode().split():
        print(f"⚠️ [SYSTEM] Đang tắt process cũ (PID: {pid}) đang chiếm cổng 8080...")
        os.kill(int(pid), signal.SIGTERM)
    import time
    time.sleep(1) # Chờ 1 giây để cổng được giải phóng hoàn toàn
except (subprocess.CalledProcessError, ProcessLookupError):
    pass # Cổng đang trống, không cần làm gì

app = FastAPI(title="Local Brain Orchestrator", version="1.0")

class CodeRequest(BaseModel):
    code_base64: str

# Local Caching
RESULT_CACHE = {}
MAX_CACHE_SIZE = 100

@app.get("/")
async def home():
    return {"status": "Online", "mode": "Local Brain"}

@app.post("/api/analyze")
async def analyze(request: CodeRequest):
    import base64
    
    if not request.code_base64.strip():
        return {"error": "Empty code"}
        
    try:
        decoded_str = base64.b64decode(request.code_base64).decode('utf-8').replace('\r\n', '\n').strip()
        code_hash = hashlib.md5(decoded_str.encode('utf-8')).hexdigest()
        
        # Caching logic
        if code_hash in RESULT_CACHE:
            print("\n" + "⚡"*30)
            print("⚡ [CACHE HIT] Đã tìm thấy kết quả cũ trong bộ nhớ Local.")
            print(f"⚡ [CACHE HIT] Trả kết quả ngay lập tức (0.01s)")
            print("⚡"*30 + "\n")
            return {"result": RESULT_CACHE[code_hash], "cached": True}
            
        print("\n" + "🆕"*30)
        print(f"🆕 [NEW REQUEST] Đang xử lý Input mới hoàn toàn...")
        print(f"🆕 [NEW REQUEST] Gửi vào LangGraph Pipeline (RoBERTa + vLLM + LightGBM)")
        print("🆕"*30 + "\n")
        print("="*60)
        final_state = await local_agent_app.ainvoke({
            "code_input": decoded_str, 
            "retry_count": 0, 
            "is_ambiguous": False
        })
        
        result = final_state["final_output"]
        print("="*60)
        print(f"✅ [REQUEST COMPLETE] FUSED SCORE: {result.get('final_score', 0):.4f}")
        print("="*60 + "\n")
        
        if "error" not in result:
            if len(RESULT_CACHE) >= MAX_CACHE_SIZE:
                oldest = next(iter(RESULT_CACHE))
                del RESULT_CACHE[oldest]
            RESULT_CACHE[code_hash] = result
            
        return {"result": result, "cached": False}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("🧠 Khởi động Local Brain tại port 8080...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
