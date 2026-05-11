import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Thêm BEHAVIORAL_INSIGHTS ngay dưới FEATURE_DETAILS
behavioral_dict = """

BEHAVIORAL_INSIGHTS = {
    # Layout & Formatting
    "comment_ratio": {
        "human": "Sinh viên thường lười comment hoặc comment rất ngắn gọn bằng tiếng Việt lóng ngóng. Đôi khi có những comment rác (comment-out code cũ).",
        "ai": "AI thường sinh ra các comment rất chi tiết, chuẩn mực ngữ pháp, giải thích rõ ràng từng khối code nhỏ và hiếm khi để lại code thừa bị comment."
    },
    "empty_line_ratio": {
        "human": "Con người thường xuống dòng tùy tiện, có thể để trống nhiều dòng liên tiếp do thói quen nhìn hoặc do copy/paste từ nhiều nguồn thiếu kiểm soát.",
        "ai": "AI thường phân chia các hàm, các block code bằng đúng 1 dòng trống một cách cực kỳ nhất quán và gọn gàng, ít khi dư thừa."
    },
    "tab_vs_space_ratio": {
        "human": "Đoạn code có sự trộn lẫn lộn xộn giữa dấu Tab và Space. Đây là lỗi định dạng kinh điển của con người khi copy-paste hoặc do IDE cấu hình chưa chuẩn.",
        "ai": "AI luôn sinh ra code với định dạng thụt lề chuẩn mực (thường là 100% Space). Sự hoàn hảo tuyệt đối này tố cáo việc code được sinh ra từ thuật toán."
    },
    "trailing_space_ratio": {
        "human": "Sinh viên thường để thừa dấu cách ở cuối dòng khi gõ phím nhanh mà không xóa đi.",
        "ai": "AI tạo ra văn bản một cách tối ưu, rất hiếm khi sinh ra các khoảng trắng vô nghĩa ở cuối dòng."
    },
    "brace_style_consistency": {
        "human": "Sinh viên có thể lúc thì mở ngoặc nhọn `{` ở cùng dòng, lúc thì xuống dòng (đặc biệt khi làm việc nhóm hoặc copy từ StackOverflow).",
        "ai": "AI có tính nhất quán cực cao, thường chọn 1 style (như Allman hoặc K&R) và tuân thủ nó từ đầu đến cuối."
    },
    
    # Naming Conventions
    "single_char_var_ratio": {
        "human": "Việc sử dụng nhiều biến 1 ký tự (i, j, n, x, s) để làm toán hay chạy vòng lặp là thói quen kinh điển của sinh viên để code cho nhanh.",
        "ai": "AI được huấn luyện dựa trên nguyên tắc Clean Code nên có xu hướng đặt tên biến đầy đủ ý nghĩa (VD: sum_of_array, total_count) thay vì dùng biến viết tắt."
    },
    "avg_identifier_length": {
        "human": "Độ dài tên biến trung bình thấp do thói quen viết tắt.",
        "ai": "Tên biến dài, mô tả chính xác chức năng do AI có khả năng chọn từ vựng rất phong phú và tuân thủ chuẩn mực kỹ nghệ phần mềm."
    },
    "keyword_to_identifier_ratio": {
        "human": "Tỉ lệ này thường cao do sinh viên dùng ít biến tự định nghĩa mà lạm dụng nhiều cấu trúc cơ bản của ngôn ngữ.",
        "ai": "AI thường tạo ra các cấu trúc đối tượng, hàm phụ trợ phong phú làm tăng số lượng định danh (identifier) so với từ khóa chuẩn."
    },

    # Structural Complexity
    "avg_cyclomatic_complexity": {
        "human": "Sinh viên thường dồn toàn bộ logic vào hàm `main` với các vòng lặp `for` và `if-else` lồng nhau sâu hoắm, khiến độ phức tạp tăng vọt.",
        "ai": "AI có thói quen tái cấu trúc (refactor) code tự động, chia nhỏ các hàm để giảm độ phức tạp lặp lại, giữ code dễ đọc."
    },
    "halstead_difficulty": {
        "human": "Mức độ khó của code cao hơn bình thường do cách sử dụng toán tử và biến lặp lại thiếu tối ưu, khiến người đọc khó theo dõi luồng dữ liệu.",
        "ai": "AI thường sinh code với mức Halstead Difficulty vừa phải, cân đối giữa toán tử và toán hạng để đạt hiệu năng hiểu tốt nhất."
    },

    # Habits & Idioms
    "total_includes": {
        "human": "Sinh viên thường include hàng loạt thư viện cơ bản do thói quen, hoặc chỉ include đúng `<iostream>` và làm mọi thứ thủ công.",
        "ai": "AI thường include chính xác những thư viện cần thiết cho các hàm thuật toán cụ thể (VD: `<algorithm>`, `<vector>`, `<numeric>`)."
    },
    "macro_count": {
        "human": "Nhiều sinh viên lập trình thi đấu (Competitive Programming) hay dùng hàng loạt macro như `#define ll long long`, `#define pb push_back`.",
        "ai": "AI thường ít khi lạm dụng macro (trừ khi được yêu cầu) vì nó đi ngược lại với triết lý C++ hiện đại (Modern C++ ưu tiên `const` và `inline`). Tuy nhiên nếu AI học từ kho dữ liệu thi đấu, nó có thể sinh ra macro một cách rập khuôn."
    },
    "has_bits_stdc": {
        "human": "Sử dụng `#include <bits/stdc++.h>` là 'đặc sản' của sinh viên làm thuật toán để đỡ phải nhớ nhiều thư viện.",
        "ai": "AI thường tránh dùng thư viện gom chung này trong môi trường production, nó thích include rõ ràng từng module hơn."
    },
    "has_fast_io": {
        "human": "Dùng `ios::sync_with_stdio(0)` là kỹ thuật tối ưu I/O của người học thuật toán.",
        "ai": "AI hiếm khi tự động thêm các lệnh tối ưu tốc độ I/O này nếu người dùng chỉ nhờ viết một chức năng thông thường."
    },

    # Information Theory
    "shannon_entropy": {
        "human": "Code có Entropy thấp (ít đa dạng ký tự) do việc tái sử dụng copy-paste nhiều dòng code giống hệt nhau hoặc lặp lại cấu trúc một cách cơ học.",
        "ai": "Code AI thường có Entropy cao hơn vì từ vựng nó sử dụng rất đa dạng, comment phong phú, cấu trúc đa dạng và ít khi để lại các cụm lặp thừa thãi."
    },
    "whitespace_entropy": {
        "human": "Sự ngẫu nhiên cao trong việc dùng Space và Tab ở các vị trí khác nhau (trước toán tử, sau dấu phẩy...) phản ánh sự tùy hứng của con người.",
        "ai": "AI tuân thủ nghiêm ngặt quy tắc khoảng trắng (vd: luôn có 1 space quanh toán tử `=`), khiến chuỗi whitespace mất đi tính ngẫu nhiên (Entropy thấp)."
    }
}
"""

content = re.sub(r'(FEATURE_DETAILS = \{.*?\n\})', r'\1' + behavioral_dict, content, flags=re.DOTALL)

# Thay thế khối in ra top features
old_top_features_block = """            for _, row in top_features.iterrows():
                f_name = row['Feature']
                f_val = row['Value']
                s_val = row['SHAP']
                
                f_info = FEATURE_DETAILS.get(f_name, {})
                f_desc = f_info.get('desc', 'Đặc trưng mã nguồn')
                
                # Logic nhận xét
                direction = "tăng khả năng là code AI 🤖" if s_val > 0 else "tăng khả năng là code người 🧑‍💻"
                color = "#e74c3c" if s_val > 0 else "#2ecc71"
                
                # Format value
                val_str = f"{f_val:.4f}" if isinstance(f_val, float) else str(f_val)
                
                st.markdown(f\"\"\"
                <div style='padding: 10px; border-left: 4px solid {color}; background-color: #1e2130; margin-bottom: 10px; border-radius: 4px;'>
                    <strong style='color:#00d4ff; font-size: 16px;'>{f_name.replace('_', ' ').title()} = {val_str}</strong><br/>
                    <i>👉 Giá trị này làm <b>{direction}</b>.</i><br/>
                    <small style='color:#aaaaaa;'>{f_desc}</small>
                </div>
                \"\"\", unsafe_allow_html=True)"""

new_top_features_block = """            # Lời bình tự động (Executive Summary)
            top_feature = top_features.iloc[0]['Feature']
            second_feature = top_features.iloc[1]['Feature']
            
            summary_text = f"Dựa trên phân tích hình thái (Fingerprint), đoạn code này mang đậm dấu ấn của **{'thuật toán máy học' if is_ai else 'con người'}**."
            summary_text += f" Quyết định của hệ thống bị chi phối mạnh mẽ nhất bởi hành vi sử dụng **{top_feature.replace('_', ' ').title()}** và thói quen liên quan đến **{second_feature.replace('_', ' ').title()}**."
            
            st.info("💡 **Tổng quan Hành vi (Executive Summary):**\\n\\n" + summary_text)

            st.markdown("### Lực lượng chi phối quyết định (Top Features)")
            for _, row in top_features.iterrows():
                f_name = row['Feature']
                f_val = row['Value']
                s_val = row['SHAP']
                
                b_info = BEHAVIORAL_INSIGHTS.get(f_name, {})
                f_info = FEATURE_DETAILS.get(f_name, {})
                
                # Lấy insight hành vi tương ứng với nhãn
                if s_val > 0:
                    insight = b_info.get("ai", f"Thống kê cho thấy thói quen {f_info.get('desc', 'này').lower()} ở mức này nghiêng về phía AI.")
                else:
                    insight = b_info.get("human", f"Thống kê cho thấy thói quen {f_info.get('desc', 'này').lower()} ở mức này nghiêng về phía Human.")
                
                direction = "Kéo dự đoán về phía AI 🤖" if s_val > 0 else "Kéo dự đoán về phía Con Người 🧑‍💻"
                color = "#e74c3c" if s_val > 0 else "#2ecc71"
                
                # Format value
                val_str = f"{f_val:.4f}" if isinstance(f_val, float) else str(f_val)
                
                st.markdown(f\"\"\"
                <div style='padding: 15px; border-left: 5px solid {color}; background-color: #1e2130; margin-bottom: 15px; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <strong style='color:#00d4ff; font-size: 18px;'>{f_name.replace('_', ' ').title()}</strong>
                        <span style='background-color: #2b3040; padding: 4px 10px; border-radius: 20px; font-family: monospace; font-size: 14px; border: 1px solid #3d425c;'>Giá trị đo được: {val_str}</span>
                    </div>
                    <div style='margin-bottom: 10px; color: {color}; font-weight: bold;'>
                        <i>👉 {direction}</i>
                    </div>
                    <div style='color: #e0e0e0; font-size: 15px; line-height: 1.6;'>
                        {insight}
                    </div>
                </div>
                \"\"\", unsafe_allow_html=True)"""

content = content.replace(old_top_features_block, new_top_features_block)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated XAI UI successfully!")
