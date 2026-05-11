import base64
import requests
import json

# 1. Thay bằng URL Local Brain của bạn (mặc định 8080)
NGROK_URL = "http://127.0.0.1:8080"

# 2. Đoạn code C++ muốn test
cpp_code = """
#include <iostream>
using namespace std;

int main() {
    int n, sum = 0;
    cout << "Enter a positive integer: ";
    cin >> n;

    for (int i = 1; i <= n; ++i) {
        sum += i;
    }

    cout << "Sum = " << sum;
    return 0;
}
"""

# 3. Encode base64
code_base64 = base64.b64encode(cpp_code.encode('utf-8')).decode('utf-8')

# 4. Gửi Request lên API đồng bộ
print(f"🚀 Đang gửi mã nguồn lên {NGROK_URL}/api/analyze ...\n")
response = requests.post(
    f"{NGROK_URL}/api/analyze",
    json={"code_base64": code_base64},
    headers={"Content-Type": "application/json"}
)

# 5. In kết quả
if response.status_code == 200:
    data = response.json()
    if "result" in data:
        res = data["result"]
        print("✅ PHÂN TÍCH THÀNH CÔNG!")
        print("-" * 50)
        
        # Tạo bản sao để in JSON sạch (không kèm HTML dài)
        clean_json = {k: v for k, v in res.items() if k not in ['global_html', 'chunks']}
        if 'chunks' in res:
            clean_json['chunks_count'] = len(res['chunks'])
            # Chỉ lấy metadata của chunk đầu tiên làm mẫu
            if len(res['chunks']) > 0:
                first_chunk = res['chunks'][0].copy()
                if 'html' in first_chunk: del first_chunk['html']
                clean_json['sample_chunk_metadata'] = first_chunk

        print("📦 FULL GOLD METADATA JSON:")
        print(json.dumps(clean_json, indent=2, ensure_ascii=False))
        
        print("-" * 50)
        print("🧠 CHỈ SỐ PERPLEXITY (vLLM Qwen 2.5 Coder):")
        print(f"   • Mean PPL   : {res.get('perplexity', 0.0):.2f}")
        print(f"   • Max PPL    : {res.get('max_ppl', 0.0):.2f} (Dòng làm AI sốc nhất)")
        print(f"   • Burstiness : {res.get('burstiness', 0.0):.2f} (Độ biến thiên)")
        
        print("-" * 50)
        print("💡 GLOBAL CRITIQUE:")
        print(res.get('global_critique', 'N/A'))
    else:
        print("⚠️ Lỗi:", data)
else:
    print(f"❌ HTTP Error {response.status_code}: {response.text}")
