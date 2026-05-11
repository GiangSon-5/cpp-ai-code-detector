import os
import threading
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pyngrok import ngrok, conf
import nest_asyncio

# Khởi tạo FastAPI
app = FastAPI(title="Colab Qwen Worker (PPL Only)")

# Cấu hình
API_KEY = "colab-secret-key-123"
VLLM_API_URL = "http://localhost:8001/v1/completions"

@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    # Public health check
    if request.url.path == "/":
        return await call_next(request)
        
    client_api_key = request.headers.get("X-API-Key")
    if client_api_key != API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized: Invalid API Key"})
        
    response = await call_next(request)
    return response

@app.get("/")
def read_root():
    return {
        "status": "Online", 
        "mode": "Qwen PPL Worker", 
        "vllm_port": 8001
    }

@app.post("/api/proxy/vllm")
async def proxy_vllm(request: Request):
    """
    Proxy request tới vLLM server (port 8001) để lấy Perplexity.
    Chỉ dùng cho Completions API (lấy prompt_logprobs).
    """
    try:
        body = await request.json()
        payload = body.get("payload", {})
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                VLLM_API_URL, 
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"vLLM Proxy Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_ngrok(port=8000):
    try:
        NGROK_TOKEN = "2a80Xg1J7V8mOQ8FvWl4jOQ8FvW_4r5f8a" # Bạn nên thay bằng token xịn
        conf.get_default().auth_token = NGROK_TOKEN
        
        # Tắt tunnel cũ
        tunnels = ngrok.get_tunnels()
        for t in tunnels:
            ngrok.disconnect(t.public_url)
            
        # Mở tunnel mới
        public_url = ngrok.connect(port).public_url
        return public_url
    except Exception as e:
        print(f"❌ Ngrok Error: {e}")
        return "ERROR_NGROK"

def run_server(app, port=8000):
    nest_asyncio.apply()
    
    def run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
        
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
