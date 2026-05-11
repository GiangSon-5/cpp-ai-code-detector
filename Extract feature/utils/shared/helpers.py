import re

def strip_metadata_headers(code_raw):
    """
    XÓA BỎ DATA LEAKAGE TỪ METADATA HEADER CỦA AI
    Xóa các dòng comment đầu file có chứa chữ DATASET, MODEL, GEMINI, GPT...
    """
    lines = code_raw.split('\n')
    cleaned_lines = []
    header_passed = False

    for line in lines:
        if not header_passed:
            if line.strip().startswith('//') and any(kw in line.upper() for kw in ['DATASET', 'MODEL', 'GEMINI', 'GPT', 'CATEGORY']):
                continue # Bỏ qua dòng này (leakage)
            if line.strip() != '':
                header_passed = True # Đã hết vùng header
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def unzip_file(zip_path, extract_dir):
    import zipfile
    import os
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ Đã giải nén xong vào: {extract_dir}")
