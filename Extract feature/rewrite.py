import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add imports and load_models
imports_str = """import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import shap
from core.feature_extractor.extractor import CppFeatureExtractorV8
from utils.shared.helpers import strip_metadata_headers"""

content = re.sub(r"import streamlit as st.*?from utils\.shared\.helpers import strip_metadata_headers", imports_str, content, flags=re.DOTALL)

cache_str = """@st.cache_resource
def get_extractor():
    return CppFeatureExtractorV8()

@st.cache_resource
def load_models():
    model = joblib.load("models/zoo/LightGBM_Regulated.pkl")
    scaler = joblib.load("models/zoo/scaler.pkl")
    final_features = joblib.load("models/zoo/final_features.pkl")
    return model, scaler, final_features"""

content = re.sub(r"@st\.cache_resource\s*def get_extractor\(\):\s*return CppFeatureExtractorV8\(\)", cache_str, content, flags=re.DOTALL)

# 2. Extract feature_details out of render_glossary_page to be a global FEATURE_DETAILS
feature_details_match = re.search(r'    # Dictionary chứa thông tin chi tiết từng đặc trưng.*?feature_details = (\{.*?\n    \})\n\n    # Sidebar', content, flags=re.DOTALL)
if feature_details_match:
    feature_details_code = feature_details_match.group(1)
    # Dedent the dict by 4 spaces
    lines = feature_details_code.split("\n")
    dedented_lines = [line[4:] if line.startswith("    ") else line for line in lines]
    feature_details_code = "\n".join(dedented_lines)
    
    # Place it before render_analysis_page
    content = content.replace("def render_analysis_page(extractor):", f"FEATURE_DETAILS = {feature_details_code}\n\ndef render_analysis_page(extractor, model, scaler, final_features):")
    
    # Replace in render_glossary_page
    content = re.sub(r'    # Dictionary chứa thông tin chi tiết từng đặc trưng.*?feature_details = \{.*?\n    \}\n', '    feature_details = FEATURE_DETAILS\n', content, flags=re.DOTALL)

# 3. Rewrite render_analysis_page
new_analysis_page = """def render_analysis_page(extractor, model, scaler, final_features):
    st.title("🔍 C++ Source Code AI Detection & Fingerprint")
    st.markdown("Hệ thống phát hiện mã nguồn AI dựa trên **32 đặc trưng định danh** và mô hình **Machine Learning (LightGBM)**.")

    # Sidebar Settings
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Analysis Settings")
    clean_metadata = st.sidebar.checkbox("Clean AI Metadata Headers", value=True)
    
    # Layout chính
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Input Source Code")
        code_input = st.text_area("Dán mã nguồn C++ vào đây:", height=500, placeholder="int main() {\\n    // Write your code here...\\n}")
        analyze_btn = st.button("🚀 Analyze & Detect", use_container_width=True)

    if analyze_btn and code_input:
        final_code = strip_metadata_headers(code_input) if clean_metadata else code_input
        
        with st.spinner("Đang mổ xẻ mã nguồn và chạy Inference..."):
            # 1. Trích xuất đặc trưng
            features = extractor.extract(final_code)
            
            # 2. Xử lý dữ liệu cho Model
            original_columns = scaler.feature_names_in_
            df_feat = pd.DataFrame([features])
            df_feat = df_feat.reindex(columns=original_columns, fill_value=0)
            
            # Scale
            scaled_feat = scaler.transform(df_feat)
            df_scaled = pd.DataFrame(scaled_feat, columns=original_columns)
            
            # Predict
            X_infer = df_scaled[final_features]
            pred_proba = model.predict_proba(X_infer)[0, 1]
            
            # 3. Tính SHAP
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_infer)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] # Lấy values cho class 1 (AI)
            shap_values = shap_values[0] # Lấy sample đầu tiên

        with col2:
            st.subheader("🎯 Final Prediction Result")
            is_ai = pred_proba >= 0.5
            
            if is_ai:
                st.markdown(f"<div class='ai-box'><h2 style='color:#e74c3c; margin:0;'>🤖 AI GENERATED</h2><p style='margin:0; font-size:18px;'>Confidence: <b>{pred_proba*100:.2f}%</b></p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='human-box'><h2 style='color:#2ecc71; margin:0;'>🧑‍💻 HUMAN WRITTEN</h2><p style='margin:0; font-size:18px;'>Confidence: <b>{(1 - pred_proba)*100:.2f}%</b></p></div>", unsafe_allow_html=True)
            
            st.progress(float(pred_proba))
            
            st.markdown("### Lực lượng chi phối quyết định (Top Features)")
            
            # Phân tích top features
            shap_df = pd.DataFrame({
                'Feature': final_features,
                'SHAP': shap_values,
                'Abs_SHAP': np.abs(shap_values),
                'Value': [features.get(f, 0) for f in final_features]
            }).sort_values(by='Abs_SHAP', ascending=False)
            
            top_n = 5
            top_features = shap_df.head(top_n)
            
            for _, row in top_features.iterrows():
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
                \"\"\", unsafe_allow_html=True)

        st.divider()
        with st.expander("📊 Xem chi tiết toàn bộ các đặc trưng và Fingerprint Radar"):
            categories = {
                "Layout & Formatting": ["comment_ratio", "empty_line_ratio", "avg_line_length", "max_line_length", "tab_vs_space_ratio", "trailing_space_ratio", "brace_style_consistency"],
                "Naming Conventions": ["avg_identifier_length", "identifier_length_variance", "single_char_var_ratio", "unique_identifier_ratio", "keyword_to_identifier_ratio"],
                "Structural Complexity": ["avg_cyclomatic_complexity", "num_functions", "avg_function_loc", "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs", "maintainability_index", "code_to_comment_ratio", "max_nesting_depth"],
                "Coding Habits": ["total_includes", "has_bits_stdc", "macro_count", "modern_cpp_ratio", "const_usage_ratio", "has_fast_io", "newline_style_ratio"],
                "Information Theory": ["shannon_entropy", "bigram_entropy", "whitespace_entropy"]
            }

            radar_data = []
            for cat, f_list in categories.items():
                avg_val = sum([features.get(f, 0) for f in f_list]) / len(f_list)
                radar_data.append(dict(Category=cat, Value=min(avg_val * 10, 100)))
            
            df_radar = pd.DataFrame(radar_data)
            fig = px.line_polar(df_radar, r='Value', theta='Category', line_close=True, range_r=[0,100])
            fig.update_traces(fill='toself', line_color='#00d4ff')
            fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            cat_tabs = st.tabs(list(categories.keys()))
            for i, (cat, f_list) in enumerate(categories.items()):
                with cat_tabs[i]:
                    cols = st.columns(3)
                    for j, f_name in enumerate(f_list):
                        val = features.get(f_name, 0)
                        with cols[j % 3]:
                            st.metric(label=f_name.replace("_", " ").title(), value=f"{val:.4f}" if isinstance(val, float) else val)

    else:
        st.info("👈 Hãy dán mã nguồn C++ vào ô bên trái và nhấn 'Analyze & Detect' để bắt đầu.")
"""

# We need to replace the old render_analysis_page with the new one.
# It starts at "def render_analysis_page(extractor" and ends before "def render_glossary_page():"
content = re.sub(r'def render_analysis_page\(extractor, model, scaler, final_features\):.*?else:\n        st\.info\("👈 Hãy dán mã nguồn C\+\+ vào ô bên trái và nhấn \'Analyze Fingerprint\' để bắt đầu\."\)', new_analysis_page, content, flags=re.DOTALL)

content = content.replace('render_analysis_page(extractor)', 'render_analysis_page(extractor, model, scaler, final_features)')

# Update main block
main_block = """def main():
    # Sidebar Navigation
    st.sidebar.title("🎮 Navigator")
    page = st.sidebar.radio("Chọn chức năng:", ["🚀 Phân tích Fingerprint", "📖 Từ điển 32 Đặc trưng", "⚙️ Quy trình Tuyển chọn"], index=0)
    extractor = get_extractor()
    model, scaler, final_features = load_models()
    
    if page == "🚀 Phân tích Fingerprint": render_analysis_page(extractor, model, scaler, final_features)"""

content = re.sub(r'def main\(\):.*?if page == "🚀 Phân tích Fingerprint": render_analysis_page\(extractor, model, scaler, final_features\)', main_block, content, flags=re.DOTALL)


with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated app.py successfully!")
