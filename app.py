import streamlit as st
import requests
import streamlit.components.v1 as components
import base64
import os
import json
import time
import threading

# ==============================================================================
# 1. CẤU HÌNH GIAO DIỆN
# ==============================================================================
st.set_page_config(layout="wide", page_title="C++ AI Detector Pro", page_icon="🕵️‍♂️")

st.markdown("""
<style>
    .metric-box { background-color: #262730; border: 1px solid #464b5f; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .metric-box h3 { color: #bfbfbf !important; margin: 0; font-size: 1rem; font-weight: normal; }
    .metric-box h1 { color: #ffffff !important; margin: 5px 0 0 0; }
    .gemini-box { background-color: #1e2a36; border-left: 5px solid #2196f3; padding: 15px; border-radius: 5px; color: #e3f2fd; font-family: 'Consolas', monospace; white-space: pre-wrap; line-height: 1.5; margin-bottom: 20px;}
    .chunk-explanation { background-color: #2e2e2e; border: 1px solid #444; border-radius: 8px; padding: 15px; margin: 20px 0; color: #ddd; font-size: 0.95rem; }
    .chunk-explanation strong { color: #f39c12; }
    .gemini-chunk-critique { background-color: #333; padding: 10px; border-radius: 5px; border-left: 3px solid #f39c12; margin-bottom: 10px; font-style: italic; color: #ccc; }
    .model-badge { display: inline-block; padding: 5px 12px; background-color: #4CAF50; color: white; border-radius: 20px; font-size: 0.9em; font-weight: bold; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .model-badge.oop { background-color: #9C27B0; }
    .model-badge.normal { background-color: #2196F3; }
    
    /* Style log thinking Grok */
    .thinking-log { color: #aaa; font-family: 'Courier New', Courier, monospace; font-size: 0.95em; padding: 8px; border-left: 2px solid #555; margin-bottom: 5px; background: rgba(255,255,255,0.05);}
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ C++ Source Code Analysis (Pro View)")
st.caption("Hệ thống tự động phát hiện và chọn Model phù hợp (OOP vs Normal).")
st.markdown("---")

# ==============================================================================
# 2. SIDEBAR & INPUT
# ==============================================================================
with st.sidebar:
    st.header("🔌 Kết nối Server")
    default_api_url = os.getenv("API_URL", "https://cedric-unstony-fulsomely.ngrok-free.dev")
    api_url = st.text_input("Ngrok URL (từ Colab):", value=default_api_url)

uploaded_file = st.file_uploader("📂 Tải lên file code (.cpp, .txt, .c)", type=['cpp', 'c', 'h', 'txt'])
initial_code = ""
if uploaded_file:
    try:
        initial_code = uploaded_file.getvalue().decode("utf-8")
        st.toast("✅ Đã tải file thành công!")
    except:
        st.error("Lỗi đọc file.")

code_input = st.text_area("Nhập code C++ cần kiểm tra:", value=initial_code, height=300)

# ==============================================================================
# 3. HÀM CHẠY LUỒNG MẠNG (ĐỂ KHÔNG BLOCK UI)
# ==============================================================================
def fetch_sse_data(url, payload, event_queue):
    try:
        response = requests.post(url, json=payload, stream=True, timeout=600)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        payload_str = decoded_line[6:]
                        try:
                            data_json = json.loads(payload_str)
                            event_queue.append(data_json)
                        except json.JSONDecodeError:
                            pass
        else:
            event_queue.append({"error": f"HTTP {response.status_code}"})
    except Exception as e:
        event_queue.append({"error": str(e)})

# ==============================================================================
# 4. LOGIC XỬ LÝ CHÍNH (STREAMING + NỘI SUY DÂY THUN)
# ==============================================================================
if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
    if not api_url or "ngrok" not in api_url or not code_input.strip():
        st.warning("⚠️ Vui lòng nhập URL Ngrok hợp lệ và Code!")
    else:
        b64_code = base64.b64encode(code_input.encode('utf-8')).decode('utf-8')
        full_url = f"{api_url.rstrip('/')}/api/analyze_stream" 
        
        # Danh sách chứa tín hiệu gửi từ Server
        event_queue = []
        
        # Bật luồng mạng chạy ngầm
        thread = threading.Thread(target=fetch_sse_data, args=(full_url, {"code_base64": b64_code}, event_queue))
        thread.start()
        
        res = None 
        
        # --- GIAO DIỆN XỬ LÝ "DÂY THUN" ---
        with st.status("🚀 Đang mở kết nối luồng trí tuệ nhân tạo...", expanded=True) as status_ui:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Khởi tạo các mốc của dây thun
            current_displayed_progress = 0.0 # Thanh đang bò tới đâu
            target_progress = 0.0          # Đích đến tiếp theo mà server chỉ định
            interpolation_speed = 0.5      # Tốc độ bò mặc định (% mỗi 0.1s)
            
            # Vòng lặp giao diện chạy liên tục cho đến khi luồng mạng kết thúc
            while thread.is_alive() or event_queue:
                
                # 1. Nếu có tin nhắn mới từ Server (Server giật dây thun)
                if event_queue:
                    payload = event_queue.pop(0) # Rút tin nhắn ra
                    
                    if "error" in payload:
                        st.error(f"❌ Lỗi: {payload['error']}")
                        status_ui.update(label="Thất bại!", state="error", expanded=True)
                        break
                        
                    # Lấy mốc Target mới từ Server (Ví dụ: Từ 20 nhảy vọt lên 65)
                    if "progress" in payload:
                        target_progress = float(payload["progress"])
                        
                        # Nếu Server báo xong, ép thanh hiển thị chạy một lèo cho xong (Tua nhanh)
                        if target_progress >= 100:
                            current_displayed_progress = 100
                            
                        # Nếu Server báo "Khúc này sẽ kẹt lâu" -> Tính toán giảm tốc độ bò
                        if "estimated_time" in payload and payload["estimated_time"] > 0:
                            gap = 90.0 - current_displayed_progress # Trần nhà là 90%
                            # Chia nhỏ quãng đường cho thời gian (0.1s update 1 lần)
                            interpolation_speed = gap / (payload["estimated_time"] * 10) 
                        else:
                            # Nếu không báo gì, bò tốc độ bình thường
                            interpolation_speed = 0.5 
                            
                    if "message" in payload:
                        status_text.markdown(f'<div class="thinking-log">🧠 <i>{payload["message"]}</i></div>', unsafe_allow_html=True)
                        status_ui.update(label=f"Đang xử lý: {int(current_displayed_progress)}%")
                        
                    if "result" in payload:
                        res = payload["result"]
                        status_ui.update(label="✅ Đã phân tích xong!", state="complete", expanded=False)
                        break

                # 2. Logic "Nội suy dây thun" (Chạy từ từ về đích)
                if current_displayed_progress < target_progress:
                    # Nếu Backend chạy quá nhanh (Target vượt xa Display) -> Tua nhanh x3 tốc độ
                    if target_progress - current_displayed_progress > 20:
                        current_displayed_progress += 2.0
                    else:
                        # Còn lại thì bò từ từ
                        current_displayed_progress += interpolation_speed
                        
                elif current_displayed_progress == target_progress and target_progress < 100:
                    # Nếu đã chạm đích (VD: 20%) mà chưa có mốc mới -> Bò tiếp về "Trần nhà"
                    # Trần nhà của Router (20) là 60%. Trần nhà của RoBERTa (65) là 90%
                    ceiling = 60 if target_progress < 60 else 90
                    if current_displayed_progress < ceiling:
                        current_displayed_progress += interpolation_speed
                
                # Ép kiểu dữ liệu an toàn cho progress bar (0.0 đến 100.0)
                safe_progress = max(0.0, min(100.0, current_displayed_progress))
                progress_bar.progress(int(safe_progress) / 100.0)
                
                # Cập nhật số % lên thanh Status
                if safe_progress < 100:
                    status_ui.update(label=f"Đang xử lý: {int(safe_progress)}% (Đang chạy Model)")
                
                # Ngủ một nhịp để tạo hiệu ứng animation mượt mà (60 khung hình/giây)
                time.sleep(0.05) 

        # Đảm bảo luồng tắt hẳn
        thread.join()

        # --- HIỂN THỊ KẾT QUẢ KHI STREAM HOÀN TẤT ---
        if res:
            score = res.get("final_score", 0)
            pred = res.get("final_pred", "UNKNOWN")
            gemini_global = res.get("global_critique", "LLM Analysis unavailable")
            global_html = res.get("global_html", "")
            chunks = res.get("chunks", [])
            total_tok = res.get("total_tokens", 0)
            total_chk = res.get("total_chunks", 1)
            model_used = res.get("model_used", "Unknown Model")
            ppl_score = res.get("perplexity", 0.0)

            st.success("🎉 Quá trình phân tích hoàn tất thành công!")

            badge_class = "oop" if "OOP" in model_used else "normal"
            st.markdown(f'<div class="model-badge {badge_class}">⚡ Model được định tuyến: {model_used}</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-box"><h3>AI Score</h3><h1>{score:.4f}</h1></div>', unsafe_allow_html=True)
            with c2:
                color = "#ff4b4b" if "AI" in pred else "#00c853"
                st.markdown(f'<div class="metric-box"><h3>Kết luận</h3><h2 style="color:{color};">{pred}</h2></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-box"><h3>Perplexity (PPL)</h3><h1>{ppl_score:.2f}</h1></div>', unsafe_allow_html=True)

            st.subheader("💡 Đánh giá chuyên môn (Tổng kết)")
            st.markdown(f'<div class="gemini-box">{gemini_global}</div>', unsafe_allow_html=True)

            if global_html:
                st.subheader("🎨 Tổng quan (Global Heatmap)")
                st.caption("Kéo chuột trong khung dưới để xem toàn bộ code và các tín hiệu (Signals).")
                components.html(global_html, height=800, scrolling=True)

            if chunks:
                st.markdown("---")
                st.subheader(f"🧩 Chi tiết từng phần phân tích ({len(chunks)} Chunks)")
                
                if total_chk > 1:
                    msg = f"""
                    ℹ️ <b>Thông tin xử lý:</b> Mã nguồn có tổng cộng <strong>{total_tok} tokens</strong>. 
                    Hệ thống đã tự động chia thành <strong>{total_chk} chunks</strong> để đảm bảo độ chính xác. 
                    """
                    st.markdown(f'<div class="chunk-explanation">{msg}</div>', unsafe_allow_html=True)
                
                for chunk in chunks:
                    idx = chunk['index']
                    lbl = chunk['label']
                    scr = chunk['score']
                    crit = chunk.get("critique", "N/A")
                    
                    icon = "🤖" if lbl == "AI" else "👤"
                    with st.expander(f"{icon} Chunk {idx}: {lbl} (Score: {scr:.4f})"):
                        st.markdown("**🤖 Phân tích kỹ thuật:**")
                        st.markdown(f'<div class="gemini-chunk-critique">{crit}</div>', unsafe_allow_html=True)
                        st.markdown("**🎨 Báo cáo LIG (Layer Integrated Gradients):**")
                        components.html(chunk["html"], height=500, scrolling=True)