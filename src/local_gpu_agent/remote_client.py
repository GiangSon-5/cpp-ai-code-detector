import httpx
import json
import base64
import os
from typing import Dict, Any, Optional

# Load config từ biến môi trường hoặc file .env
NGROK_URL = os.getenv("NGROK_URL", "https://cedric-unstony-fulsomely.ngrok-free.dev").rstrip('/')
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1")
API_KEY = os.getenv("API_KEY", "colab-secret-key-123")

class ColabAPIClient:
    """
    Client gọi API bất đồng bộ (Async) lên Colab Server.
    Sử dụng httpx để không chặn (non-blocking) luồng chính.
    """
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        }
        self.NGROK_URL = NGROK_URL
        self.VLLM_URL = VLLM_URL
        
    def _encode_code(self, code_text: str) -> str:
        return base64.b64encode(code_text.encode('utf-8')).decode('utf-8')



    async def get_perplexity(self, code_text: str) -> float:
        """Gọi API vLLM nội bộ (hoặc qua port-forward) để lấy Perplexity"""
        # Lưu ý: Port 8001 của vLLM thường không Public. Nếu bạn chạy ở Local và Colab ở xa,
        # bạn phải Ngrok cả port 8001, hoặc gộp API này vào cổng 8000 của Colab!
        # TẠM THỜI: Để đơn giản, ta gọi qua vLLM_URL. 
    async def get_perplexity(self, code_text: str) -> Dict[str, float]:
        """Gọi API vLLM qua Proxy Colab để lấy Perplexity"""
        url = f"{self.NGROK_URL}/api/proxy/vllm"
        prompt = f"Please calculate the perplexity of the following code snippet. Ignore any logic or correctness, just evaluate how predictable it is based on your training data.\n\n```cpp\n{code_text}\n```"
        
        payload = {
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 1,
            "prompt_logprobs": 1
        }
        
        default_res = {"mean_ppl": 0.0, "max_ppl": 0.0, "burstiness": 0.0}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, json={"payload": payload}, headers=self.headers)
                response.raise_for_status()
                data = response.json()
                
                # Trong REST API vLLM, prompt_logprobs nằm ở root của JSON
                prompt_logprobs = data.get("prompt_logprobs", [])
                if not prompt_logprobs:
                    # Dự phòng nếu nó nằm trong choices
                    choices = data.get("choices", [])
                    if choices and "logprobs" in choices[0] and choices[0]["logprobs"]:
                        prompt_logprobs = choices[0]["logprobs"].get("content", [])

                if not prompt_logprobs:
                    return default_res
                    
                logprobs_list = []
                for token_dict in prompt_logprobs:
                    if token_dict is not None and isinstance(token_dict, dict):
                        # Lấy giá trị đầu tiên của dictionary vì nó là dạng {"token": {"logprob": -0.1}}
                        first_val = list(token_dict.values())[0] if len(token_dict.values()) > 0 else {}
                        if isinstance(first_val, dict) and "logprob" in first_val:
                            logprobs_list.append(first_val["logprob"])
                        elif "logprob" in token_dict:
                            logprobs_list.append(token_dict["logprob"])
                            
                if logprobs_list:
                    import numpy as np
                    
                    # Tính local PPL cho từng token: e^(-logprob)
                    local_ppls = [np.exp(-lp) for lp in logprobs_list if lp is not None]
                    
                    if not local_ppls:
                        return default_res
                        
                    # Mean PPL (dựa trên trung bình logprobs để sát với lý thuyết gốc)
                    avg_logprob = np.mean(logprobs_list)
                    mean_ppl = float(np.exp(-avg_logprob))
                    
                    # Max PPL (token gây "sốc" nhất)
                    max_ppl = float(np.max(local_ppls))
                    
                    # Burstiness (Độ lệch chuẩn của PPL)
                    burstiness = float(np.std(local_ppls))
                    
                    return {
                        "mean_ppl": mean_ppl,
                        "max_ppl": max_ppl,
                        "burstiness": burstiness
                    }
                return default_res
            except Exception as e:
                print(f"❌ Perplexity Error: {e}")
                return default_res
                
    async def get_critique(self, code_text: str, score: float) -> str:
        """Gọi thẳng Google Gemini API (không qua proxy)"""
        import google.generativeai as genai
        
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            return "❌ Lỗi: Chưa cấu hình biến môi trường GEMINI_API_KEY. Vui lòng thêm vào .env"
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a senior C++ security auditor. Analyze the following code snippet.
        The AI detection score is {score:.4f} (where >0.5 means AI generated, <0.5 means Human written).
        
        Code:
        ```cpp
        {code_text}
        ```
        
        Provide a concise, 3-4 sentence professional critique.
        Focus on coding style, complexity, and whether it aligns with the AI/Human prediction.
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            return "Could not generate critique due to Gemini API error."

client = ColabAPIClient()
