import os
import pandas as pd
from data.pipeline.pipeline import DataPreprocessor, get_feature_selection_pipeline
from models.engine.model_manager import ModelManager
from inference.reporter.predictor import CodeInference
from core.feature_extractor.extractor import CppFeatureExtractorV8

def main():
    # 1. Cấu hình đường dẫn
    INPUT_JSONL = "data/raw/train_data_10_model_AI.jsonl"
    SAVE_DIR = "models/zoo"
    
    # 2. Tiền xử lý dữ liệu
    preprocessor = DataPreprocessor()
    df = preprocessor.process_jsonl(INPUT_JSONL)
    
    # Chia dữ liệu (Không cần scale ở đây vì đã có trong Pipeline)
    X = df.drop(columns=['label'])
    y = df['label']
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    # 3. Lựa chọn đặc trưng thông qua Pipeline
    fs_pipeline = get_feature_selection_pipeline(k_best=20, lasso_c=0.5)
    print("\n🛡️ ĐANG CHẠY PIPELINE LỰA CHỌN ĐẶC TRƯNG...")
    X_train_final = fs_pipeline.fit_transform(X_train, y_train)
    X_val_final = fs_pipeline.transform(X_val)
    X_test_final = fs_pipeline.transform(X_test)
    
    # Lấy tên các feature cuối cùng
    final_features = fs_pipeline.get_feature_names_out()
    print(f"✅ Đã chốt {len(final_features)} features tinh túy.")
    
    # 4. Huấn luyện Model
    imbalance_ratio = float((y_train == 0).sum()) / (y_train == 1).sum()
    manager = ModelManager()
    manager.initialize_models(imbalance_ratio)
    manager.train(X_train_final, y_train, X_val_final, y_val)
    
    # 5. Đánh giá
    manager.evaluate(X_test_final, y_test)
    
    # 6. Lưu trữ
    manager.save_artifacts(SAVE_DIR, preprocessor.scaler, final_features)
    
    # 7. Chạy Inference (Ví dụ)
    inference = CodeInference(
        extractor=preprocessor.extractor,
        model=manager.best_model,
        scaler=preprocessor.scaler,
        final_features=final_features,
        original_columns=original_columns
    )
    
    # Giả sử có thư mục test
    # summary_df = inference.scan_directory("test-set")
    # print(summary_df)

if __name__ == "__main__":
    main()
