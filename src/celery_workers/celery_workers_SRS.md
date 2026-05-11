# Celery Workers — Đặc tả Nghiệp vụ (SRS)

### UC: Xử lý nền và đồng bộ Data Lake - Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Celery Workers chịu trách nhiệm các tác vụ nền: đẩy dữ liệu thô lên DagsHub S3 (Bronze), trích xuất features và push Silver, ghi log prediction vào Gold, và kiểm tra trigger tái huấn luyện.

| Primary Actor: | System (Auto-triggered) | Secondary Actor: | DagsHub S3 / PostgreSQL |
|----------------|------------------------|-------------------|--------------------------|
| **Description:** | Background processing cho data pipeline |
| **Trigger:** | Sự kiện từ Django/FastAPI hoặc Celery Beat schedule |
| **Preconditions:** | PRE1: Redpanda broker running. PRE2: DagsHub credentials configured. |
| **Post-conditions:** | POST1: Data pushed to S3 Bronze/Silver. POST2: Gold records in PostgreSQL. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** (Implicit) User submit code → Django triggers Celery task
- **Hệ thống xử lý:** Serialize data → Upload S3 → Extract features → Push Silver
- **Kết quả nhận được:** Data available on DagsHub for DVC versioning + MLflow training

**Normal Flow:**

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Django publishes "code.submitted" event | Celery picks up push_bronze_to_s3 task |
| 2 | — | Upload raw code JSON to DagsHub S3 Bronze bucket |
| 3 | FastAPI completes inference | Celery picks up extract_and_push_silver task |
| 4 | — | CppFeatureExtractorV8 extracts 32 features |
| 5 | — | RobertaTokenizer tokenizes 512 tokens |
| 6 | — | Push Silver-ML + Silver-DL parquet to S3 |
| 7 | — | log_prediction_to_gold writes to PostgreSQL |

**Exception:**

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | S3 upload timeout | Retry 3x exponential backoff, alert via Prometheus |
| 2 | Feature extraction crash | Log error, skip Silver push, continue Gold write |
| 3 | Broker connection lost | Queue tasks in memory, flush when reconnected |

**Business Rules:**

| No | Rule |
|----|------|
| 1 | Bronze push là fire-and-forget, không block API response |
| 2 | Silver extraction chỉ chạy khi label available (supervised) |
| 3 | Retrain trigger: check daily nếu Silver có > 100 new samples |
| 4 | All tasks có max retry = 3, backoff = 2^retry seconds |

**Bảng mô tả Data Mapping:**

| Tên trường | Mô tả | Kiểu | Mặc định | Bắt buộc | Ví dụ | Direct to |
|------------|-------|------|----------|----------|-------|-----------|
| code_hash | Hash SHA-256 | string | — | Y | "a1b2c3..." | S3 key |
| raw_code | Code thô | text | — | Y | "#include..." | S3 Bronze |
| features_32 | 32 ML features | dict | — | Y (Silver) | {empty_line_ratio: 0.12,...} | S3 Silver-ML |
| input_ids_512 | Token IDs | list[int] | — | Y (Silver) | [101, 7592,...] | S3 Silver-DL |
