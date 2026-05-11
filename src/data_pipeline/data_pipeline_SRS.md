# Data Pipeline — Đặc tả Nghiệp vụ (SRS)

### UC: Xử lý dữ liệu Medallion và Tái huấn luyện - Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Data Pipeline thực hiện biến đổi dữ liệu theo kiến trúc Medallion (Bronze → Silver → Gold), trích xuất features cho ML/DL models, và orchestrate closed-loop retraining khi có đủ dữ liệu mới.

| Primary Actor: | System (Celery/Scheduler) | Secondary Actor: | DagsHub S3 / MLflow |
|----------------|--------------------------|-------------------|----------------------|
| **Description:** | ETL pipeline và retraining orchestration |
| **Trigger:** | Celery task hoặc scheduled job (daily) |
| **Preconditions:** | PRE1: Bronze data available trên DagsHub S3. PRE2: Model artifacts exist. |
| **Post-conditions:** | POST1: Silver parquet on S3. POST2: Gold metrics in PostgreSQL. POST3: Updated model (if retrained). |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** 100 samples mới tích lũy trên Bronze layer (DagsHub S3)
- **Hệ thống xử lý:** ETL Bronze→Silver (32 features + 512 tokens) → Trigger retraining → Evaluate → Log MLflow
- **Kết quả nhận được:** Improved model ONNX artifact, Gold performance metrics, DVC-versioned Silver data

**Normal Flow:**

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Scheduler triggers daily ETL | DVC pull Bronze data from DagsHub S3 |
| 2 | — | Extract 32 ML features via CppFeatureExtractorV8 |
| 3 | — | Tokenize 512 DL tokens via GraphCodeBERT tokenizer |
| 4 | — | Save Silver-ML + Silver-DL parquet, DVC push to S3 |
| 5 | — | Check if Silver has > 100 new labeled samples |
| 6 | — | If yes: train LightGBM + RoBERTa ensemble |
| 7 | — | Grid search optimal α (fusion weight) and threshold |
| 8 | — | Evaluate on holdout set, log metrics to MLflow |
| 9 | — | If improved: export ONNX, push to model registry |
| 10 | — | Write GoldModelPerformance record to PostgreSQL |

**Exception:**

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Insufficient labeled data (<100) | Skip retraining, log info |
| 2 | Training OOM | Reduce batch size, retry with gradient accumulation |
| 3 | Model regression (worse F1) | Keep current model, log warning, alert admin |
| 4 | DagsHub S3 unreachable | Retry 3x, fallback to local parquet |

**Business Rules:**

| No | Rule |
|----|------|
| 1 | Silver-ML phải có đúng 32 features matching scaler.feature_names_in_ |
| 2 | Silver-DL token length = 512 (padded), stride = 256 |
| 3 | Retraining chỉ trigger khi có ≥ 100 new labeled samples |
| 4 | Model chỉ deploy khi F1 macro > current production F1 |
| 5 | Mọi experiment log lên MLflow trên DagsHub (offload local) |
| 6 | Data version control bằng DVC, mỗi batch Silver có DVC tag |

**Bảng mô tả Data Mapping:**

| Tên trường | Mô tả | Kiểu | Mặc định | Bắt buộc | Ví dụ | Direct to |
|------------|-------|------|----------|----------|-------|-----------|
| bronze_path | S3 path Bronze JSONL | string | — | Y | s3://bronze/2026/04/28/ | DVC pull |
| silver_ml_path | S3 path Silver-ML | string | — | Y | s3://silver/ml/feat_20260428.parquet | LightGBM train |
| silver_dl_path | S3 path Silver-DL | string | — | Y | s3://silver/dl/tok_20260428.parquet | RoBERTa train |
| alpha | Fusion weight | float | 0.48 | Y | 0.48 | Grid search |
| threshold | Decision boundary | float | 0.5 | Y | 0.4823 | PR curve |
| f1_macro | Evaluation metric | float | — | Y | 0.9532 | Gold DB |
