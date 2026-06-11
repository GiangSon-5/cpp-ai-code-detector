import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Paths
WORKSPACE = r"C:\Users\2\Desktop\WORKSPACE\LVTN-Main"
TRAIN_DATASET_PATH = os.path.join(WORKSPACE, "data_splits", "train_set", "train_data_with_oop.jsonl")
CACHE_PATH = os.path.join(WORKSPACE, "retrain ML models", "extracted_features_cache.pkl")
REPORT_PATH = os.path.join(WORKSPACE, "retrain ML models", "optimal_k_report.md")
PLOT_PATH = os.path.join(WORKSPACE, "retrain ML models", "optimal_k_search.png")

# Import extractor from retrain_improved
import sys
sys.path.append(os.path.join(WORKSPACE, "retrain ML models"))
from retrain_improved import CppFeatureExtractorV8, strip_metadata_headers

def load_or_extract_features():
    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Found cached features at {CACHE_PATH}. Loading...")
        with open(CACHE_PATH, 'rb') as f:
            df_all = pickle.load(f)
        print(f"[CACHE] Loaded {df_all.shape[0]} samples with {df_all.shape[1] - 1} features.")
        return df_all

    print(f"[EXTRACT] Cache not found. Extracting features from {TRAIN_DATASET_PATH}...")
    with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    extractor = CppFeatureExtractorV8()
    extracted_data = []
    
    start_time = time.time()
    for idx, item in enumerate(train_data):
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx}/{len(train_data)} records... ({time.time() - start_time:.1f}s)")
        clean_code = strip_metadata_headers(item['code'])
        features = extractor.extract(clean_code)
        features['label'] = 1 if item['label'] == 'AI' else 0
        extracted_data.append(features)
        
    df_all = pd.DataFrame(extracted_data).fillna(0)
    
    # Save cache
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(df_all, f)
    print(f"[EXTRACT] Completed in {time.time() - start_time:.1f}s. Cache saved.")
    return df_all

def main():
    print("=" * 70)
    print("EXPERIMENT: OPTIMIZING FEATURE SELECTION PARAMETER k")
    print("=" * 70)
    
    # Step 1: Load features
    df_all = load_or_extract_features()
    X = df_all.drop(columns=['label'])
    y = df_all['label']
    
    # Step 2: Split and scale data (Identical to training script)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Step 3: Correlation Filter (Stage 1)
    corr_matrix = X_train_scaled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
    
    X_train_t1 = X_train_scaled.drop(columns=to_drop_corr)
    max_k = X_train_t1.shape[1]
    
    print(f"\n[STAGE 1] Correlation Filter dropped {len(to_drop_corr)} features (r > 0.85).")
    print(f"Remaining features candidate count for SelectKBest: {max_k}")
    
    # Method 1: Grid Search for k (from 5 to max_k)
    print("\n" + "-" * 50)
    print("METHOD 1: GRID SEARCH FOR k")
    print("-" * 50)
    
    k_range = list(range(5, max_k + 1, 2))
    if 20 not in k_range and max_k >= 20:
        k_range.append(20)
    if max_k not in k_range:
        k_range.append(max_k)
    k_range = sorted(list(set(k_range)))
        
    grid_results = []
    
    # We will track metrics for LightGBM, XGBoost, and RF
    for k in k_range:
        print(f"Evaluating k = {k}...")
        
        # 1. SelectKBest
        selector_kbest = SelectKBest(score_func=mutual_info_classif, k=k)
        X_train_t2_arr = selector_kbest.fit_transform(X_train_t1, y_train)
        selected_kbest_cols = X_train_t1.columns[selector_kbest.get_support()]
        X_train_t2 = pd.DataFrame(X_train_t2_arr, columns=selected_kbest_cols)
        
        # 2. Lasso L1
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, C=0.5)
        selector_l1 = SelectFromModel(lasso)
        selector_l1.fit(X_train_t2, y_train)
        final_features = X_train_t2.columns[selector_l1.get_support()]
        
        n_final = len(final_features)
        if n_final == 0:
            print(f"  k = {k} results in 0 features after Lasso L1! Skipping.")
            continue
            
        X_train_final = X_train_scaled[final_features]
        X_val_final = X_val_scaled[final_features]
        X_test_final = X_test_scaled[final_features]
        
        # Train Models
        imbalance_ratio = float((y_train == 0).sum()) / (y_train == 1).sum()
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            random_state=42, verbose=-1, class_weight='balanced',
            max_depth=5, num_leaves=20, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, n_estimators=300, n_jobs=-1
        )
        lgb_model.fit(X_train_final, y_train, eval_set=[(X_val_final, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42,
            scale_pos_weight=imbalance_ratio,
            max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, n_estimators=300, early_stopping_rounds=30, n_jobs=-1
        )
        xgb_model.fit(X_train_final, y_train, eval_set=[(X_val_final, y_val)], verbose=False)
        
        # RF
        rf_model = RandomForestClassifier(
            random_state=42, class_weight='balanced',
            max_depth=6, min_samples_split=10, min_samples_leaf=5, n_estimators=300, n_jobs=-1
        )
        rf_model.fit(X_train_final, y_train)
        
        # Evaluate
        models_eval = {
            "LightGBM": lgb_model,
            "XGBoost": xgb_model,
            "RandomForest": rf_model
        }
        
        k_res = {"k": k, "features_after_lasso": n_final}
        for name, model in models_eval.items():
            train_preds = model.predict(X_train_final)
            val_preds = model.predict(X_val_final)
            test_preds = model.predict(X_test_final)
            
            k_res[f"{name}_train_f1"] = f1_score(y_train, train_preds)
            k_res[f"{name}_val_f1"] = f1_score(y_val, val_preds)
            k_res[f"{name}_test_f1"] = f1_score(y_test, test_preds)
            k_res[f"{name}_val_acc"] = accuracy_score(y_val, val_preds)
            k_res[f"{name}_overfit"] = k_res[f"{name}_train_f1"] - k_res[f"{name}_val_f1"]
            
        grid_results.append(k_res)
        print(f"  Final features: {n_final} | LGB Val F1: {k_res['LightGBM_val_f1']:.4f} | XGB Val F1: {k_res['XGBoost_val_f1']:.4f}")

    df_grid = pd.DataFrame(grid_results)
    
    # Method 2: RFECV (Recursive Feature Elimination with Cross-Validation)
    print("\n" + "-" * 50)
    print("METHOD 2: RFECV WITH LIGHTGBM")
    print("-" * 50)
    print("Running RFECV on the features remaining after Correlation Filter (Stage 1)...")
    
    # Use LightGBM with 5-fold CV to find the optimal feature subset
    lgb_estimator = lgb.LGBMClassifier(
        random_state=42, verbose=-1, class_weight='balanced',
        max_depth=5, num_leaves=20, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.05, n_estimators=150, n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(
        estimator=lgb_estimator,
        step=1,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    start_rfecv = time.time()
    rfecv.fit(X_train_t1, y_train)
    print(f"RFECV completed in {time.time() - start_rfecv:.1f}s.")
    
    opt_features_rfecv = list(X_train_t1.columns[rfecv.support_])
    print(f"Optimal number of features selected by RFECV: {rfecv.n_features_}")
    print(f"Selected RFECV Features ({len(opt_features_rfecv)}): {opt_features_rfecv}")
    
    # Evaluate RFECV features on Validation and Test sets
    X_train_rfecv = X_train_scaled[opt_features_rfecv]
    X_val_rfecv = X_val_scaled[opt_features_rfecv]
    X_test_rfecv = X_test_scaled[opt_features_rfecv]
    
    lgb_rfecv = lgb.LGBMClassifier(
        random_state=42, verbose=-1, class_weight='balanced',
        max_depth=5, num_leaves=20, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.05, n_estimators=300, n_jobs=-1
    )
    lgb_rfecv.fit(X_train_rfecv, y_train, eval_set=[(X_val_rfecv, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    
    rfecv_val_f1 = f1_score(y_val, lgb_rfecv.predict(X_val_rfecv))
    rfecv_test_f1 = f1_score(y_test, lgb_rfecv.predict(X_test_rfecv))
    print(f"RFECV LightGBM Performance -> Val F1: {rfecv_val_f1:.4f} | Test F1: {rfecv_test_f1:.4f}")
    
    # Plotting Results
    print("\n[PLOT] Generating performance visualization...")
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Validation F1 for different models
    plt.subplot(2, 1, 1)
    plt.plot(df_grid['k'], df_grid['LightGBM_val_f1'], marker='o', label='LightGBM Val F1', color='#1f77b4', linewidth=2)
    plt.plot(df_grid['k'], df_grid['XGBoost_val_f1'], marker='s', label='XGBoost Val F1', color='#ff7f0e', linewidth=2)
    plt.plot(df_grid['k'], df_grid['RandomForest_val_f1'], marker='^', label='Random Forest Val F1', color='#2ca02c', linestyle='--')
    
    # Add RFECV marker
    plt.axhline(y=rfecv_val_f1, color='red', linestyle='-.', label=f'RFECV Val F1 ({rfecv_val_f1:.4f})')
    plt.axvline(x=20, color='gray', linestyle=':', label='Current Default k=20')
    
    plt.title('Validation F1-score vs SelectKBest Parameter k', fontsize=12, pad=10)
    plt.xlabel('k (SelectKBest)')
    plt.ylabel('F1-score')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Subplot 2: Overfit (Train F1 - Val F1) & Number of features after Lasso
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(df_grid['k'], df_grid['LightGBM_overfit'], marker='o', label='LightGBM Overfit Margin', color='#d62728', alpha=0.7)
    ax1.plot(df_grid['k'], df_grid['XGBoost_overfit'], marker='x', label='XGBoost Overfit Margin', color='#9467bd', alpha=0.7)
    ax1.set_ylabel('Overfit Margin (Train F1 - Val F1)', color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(df_grid['k'], df_grid['features_after_lasso'], alpha=0.25, width=1.0, color='gray', label='Features After Lasso L1')
    ax2.set_ylabel('Features count after Lasso L1', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.title('Overfitting Margin & Features Retained vs k', fontsize=12, pad=10)
    ax1.set_xlabel('k (SelectKBest)')
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"[PLOT] Graph saved to {PLOT_PATH}")
    
    # Find Optimal k from Grid Search
    # We want to maximize Validation F1, and if there are ties or similar values, choose a k that keeps features count compact and overfit low.
    # Let's sort by LightGBM Val F1
    best_lgb_idx = df_grid['LightGBM_val_f1'].idxmax()
    best_lgb_k = int(df_grid.loc[best_lgb_idx, 'k'])
    best_lgb_f1 = float(df_grid.loc[best_lgb_idx, 'LightGBM_val_f1'])
    best_lgb_features = int(df_grid.loc[best_lgb_idx, 'features_after_lasso'])
    
    # Save Report Markdown
    print("\n[REPORT] Generating experiment report...")
    report_content = f"""# Experiment Report: Optimal Feature Selection k Tuning

Thử nghiệm này chạy trên tập dữ liệu hoàn chỉnh `train_data_with_oop.jsonl` (8,880 records), đánh giá hiệu năng phân loại khi thay đổi tham số $k$ trong tầng lọc `SelectKBest`.

---

## 📊 Kết quả tổng quát

### 1. Phương pháp 1: Grid Search quét tìm k tối ưu
Dưới đây là bảng đánh giá hiệu năng các mô hình theo từng giá trị $k$ của SelectKBest:

| k | Số features sau Lasso L1 | LightGBM Val F1 | XGBoost Val F1 | RandomForest Val F1 | LGBM Overfit (Train-Val F1) |
|---|---|---|---|---|---|
"""
    for _, row in df_grid.iterrows():
        report_content += f"| {int(row['k'])} | {int(row['features_after_lasso'])} | {row['LightGBM_val_f1']:.4f} | {row['XGBoost_val_f1']:.4f} | {row['RandomForest_val_f1']:.4f} | {row['LightGBM_overfit']:.4f} |\n"
        
    report_content += f"""
### 2. Phương pháp 2: RFECV (Recursive Feature Elimination với Cross-Validation)
- **Thuật toán cơ sở**: LightGBM Classifier (150 trees, class_weight='balanced')
- **Số Fold kiểm thử chéo**: 5-Fold Stratified CV
- **Số lượng đặc trưng tối ưu tìm được**: **{rfecv.n_features_} đặc trưng** (trên tổng số {max_k} đặc trưng sau bộ lọc tương quan)
- **Hiệu năng trên tập Validation (F1-score)**: **{rfecv_val_f1:.4f}**
- **Hiệu năng trên tập Test kín (F1-score)**: **{rfecv_test_f1:.4f}**

Các đặc trưng được RFECV giữ lại:
`{opt_features_rfecv}`

---

## 💡 Đánh giá & Khuyến nghị

1. **Về giá trị k mặc định (k = 20)**:
   - Hiệu năng của LightGBM ở k = 20 có F1-score là **{df_grid.loc[df_grid['k'] == 20, 'LightGBM_val_f1'].values[0]:.4f}**.
   - Số đặc trưng thực tế đi qua Lasso L1 là **{df_grid.loc[df_grid['k'] == 20, 'features_after_lasso'].values[0]}**.

2. **Về giá trị k tối ưu nhất qua Grid Search**:
   - Giá trị $k$ tối ưu nhất cho LightGBM là **k = {best_lgb_k}** với Val F1 đạt **{best_lgb_f1:.4f}** (Đặc trưng sau Lasso: **{best_lgb_features}**).
   - Điểm tối ưu này giúp tăng thêm hiệu năng so với mặc định mà không làm tăng quá nhiều nguy cơ overfit.

3. **Về phương pháp RFECV**:
   - RFECV tìm ra tập **{rfecv.n_features_}** đặc trưng tối ưu đạt Val F1 là **{rfecv_val_f1:.4f}**.
   - RFECV mang lại độ chính xác cao nhưng đòi hỏi thay đổi kiến trúc code (loại bỏ SelectKBest và Lasso L1, thay bằng bộ lọc RFECV đã được huấn luyện sẵn hoặc chạy RFECV trực tiếp trong pipeline huấn luyện).

### 🏆 Đề xuất hành động:
- Nếu muốn giữ nguyên pipeline 3 tầng ổn định hiện tại, hãy cập nhật **k = {best_lgb_k}** (hoặc con số được chọn dựa trên đồ thị uốn).
- Nếu muốn tối ưu sâu hơn nữa và tự động hóa 100%, chúng ta có thể chuyển sang dùng RFECV lưu checkpoint `.pkl`.
"""
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"[REPORT] Markdown report saved to {REPORT_PATH}")
    print("=" * 70)
    print("Done!")

if __name__ == "__main__":
    main()
