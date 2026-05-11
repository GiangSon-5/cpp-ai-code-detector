# Data Pipeline — Đặc tả Kỹ thuật (SPEC)

## 1. Module Overview

Data Pipeline module thực hiện ETL theo kiến trúc Medallion: Bronze → Silver (ML + DL) → Gold. Kế thừa logic từ `extracted_with_markdown.py` (Hybrid model evaluation, SHAP, feature extraction).

**Kế thừa từ legacy code:**
- `extracted_with_markdown.py` Cell 25-26 → `bronze_to_silver/feature_extractor.py` (CppFeatureExtractorV8)
- `extracted_with_markdown.py` Cell 28 → `bronze_to_silver/tokenizer_pipeline.py` (FastInferenceDataset)
- `extracted_with_markdown.py` Cell 6 → `retraining/alpha_optimizer.py` (Grid search fusion α)
- `extracted_with_markdown.py` Cell 10 → `retraining/threshold_optimizer.py` (PR curve threshold)
- `extracted_with_markdown.py` Cell 12 → `silver_to_gold/shap_analyzer.py` (SHAP explanation)

## 2. Data Contracts & Examples

### Bronze → Silver-ML Pipeline

```python
# Input: list[dict] from Bronze
bronze_record = {
    "code_hash": "a1b2c3...",
    "raw_code": "#include <iostream>...",
    "label": "AI"  # optional, for supervised training
}

# Output: .parquet with 32 float columns + code_hash + label
# Columns: empty_line_ratio, avg_line_length, ..., trailing_space_ratio
```

### Bronze → Silver-DL Pipeline

```python
# Input: same bronze_record
# Output: .parquet with input_ids[512], attention_mask[512], code_hash, label
# Tokenizer: GraphCodeBERT (microsoft/graphcodebert-base)
# Max length: 512, stride: 256 for long code
```

### Retraining Pipeline

```python
# Input: Silver-ML parquet + Silver-DL parquet from DagsHub S3
# Process:
#   1. DVC pull latest Silver data
#   2. Train LightGBM (20 features selected) + RoBERTa (10-fold)
#   3. Grid search α (0-1, step 0.01) for fusion
#   4. Optimize threshold via PR curve
#   5. Evaluate: accuracy, F1, precision, recall, AUC-ROC
#   6. Log to MLflow on DagsHub
#   7. If improved → export ONNX → push to model registry
# Output: Updated model artifacts + Gold performance metrics
```

## 3. Core Logic (wrap từ code cũ)

### Feature Extraction (từ extracted_with_markdown.py)

```python
# Wrap CppFeatureExtractorV8 + strip_metadata_headers
# 32 features: empty_line_ratio, avg_line_length, ..., trailing_space_ratio
# Scaler: StandardScaler (fit trên 32 features, select 20 final)
```

### Hybrid Fusion (từ extracted_with_markdown.py Cell 6-10)

```python
# α = 0.48 (best from grid search)
# fused_probs = α * bert_probs + (1-α) * lgbm_probs
# threshold = optimized via precision_recall_curve (default 0.5)
```

## 4. End-to-End Trace Example

**Input:** 100 new Bronze records on DagsHub S3
**Trace:**
```
1. DVC pull bronze/2026-04-28/*.json → local
2. CppFeatureExtractorV8.extract() × 100 → DataFrame (100×32)
3. StandardScaler.transform() → scaled features
4. Save silver/ml/features_20260428.parquet → DagsHub S3
5. GraphCodeBERT tokenize × 100 → DataFrame (100×514)
6. Save silver/dl/tokens_20260428.parquet → DagsHub S3
7. (If retrain) Train LightGBM + BERT on accumulated Silver
8. (If retrain) Log metrics to MLflow
```

## 5. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|--------|
| 1 | Feature extractor crash (malformed C++) | Return zeros, log warning |
| 2 | Tokenizer exceeds 512 tokens | Chunk with stride=256, multiple rows per sample |
| 3 | Parquet write fails (disk full) | Alert, retry on DagsHub S3 directly |
| 4 | Scaler feature mismatch | Validate feature names against saved scaler |
| 5 | DVC conflict | Auto-resolve with latest timestamp |
