import os
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from utils.shared.helpers import strip_metadata_headers

class CodeInference:
    def __init__(self, extractor, model, scaler, final_features, original_columns, threshold=0.5):
        self.extractor = extractor
        self.model = model
        self.scaler = scaler
        self.final_features = final_features
        self.original_columns = original_columns
        self.threshold = threshold

    def predict_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_raw = f.read()
            
            clean_code = strip_metadata_headers(code_raw)
            features = self.extractor.extract(clean_code)
            
            df_feat = pd.DataFrame([features])
            df_feat = df_feat.reindex(columns=self.original_columns, fill_value=0)
            
            scaled_feat = self.scaler.transform(df_feat)
            df_scaled = pd.DataFrame(scaled_feat, columns=self.original_columns)
            
            X_infer = df_scaled[self.final_features]
            pred_proba = self.model.predict_proba(X_infer)[0, 1]
            
            return pred_proba
        except Exception as e:
            return None

    def scan_directory(self, parent_dir):
        print(f"\n🚀 PHẦN INFERENCE: Đang quét {parent_dir}")
        summary_results = []
        
        for root, dirs, files_in_dir in os.walk(parent_dir):
            files = []
            for ext in ('*.c', '*.cpp', '*.h', '*.hpp'):
                files.extend(glob.glob(os.path.join(root, ext)))
            
            if not files:
                continue

            folder_name = os.path.relpath(root, parent_dir)
            ai_count = 0
            human_count = 0
            error_count = 0

            for file_path in tqdm(files, desc=f"Infer {os.path.basename(root)}", leave=False):
                prob = self.predict_file(file_path)
                if prob is None:
                    error_count += 1
                    continue
                
                if prob >= self.threshold:
                    ai_count += 1
                else:
                    human_count += 1

            total = ai_count + human_count
            summary_results.append({
                "Thư mục": folder_name,
                "Tổng số file": len(files),
                "Số file AI": ai_count,
                "Số file Human": human_count,
                "Tỷ lệ AI": f"{(ai_count/total if total > 0 else 0):.2%}"
            })
        
        return pd.DataFrame(summary_results)
