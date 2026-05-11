# Celery Workers — Đặc tả Kỹ thuật (SPEC)

## 1. Module Overview

Celery Workers xử lý các tác vụ nền (background tasks) không block response API: ghi log, đẩy dữ liệu lên DagsHub S3 Data Lake, và trigger ETL pipelines.

**Broker:** Redpanda (Kafka-compatible, nhẹ hơn Redis cho use case này)

## 2. Data Contracts & Examples

### Task Definitions

```python
# tasks/bronze_tasks.py
@celery_app.task(name="push_bronze_to_s3")
def push_bronze_to_s3(code_hash: str, raw_code: str, metadata: dict):
    """Push raw code + metadata lên DagsHub S3 Bronze bucket"""
    ...

# tasks/silver_tasks.py
@celery_app.task(name="extract_and_push_silver")
def extract_and_push_silver(code_hash: str, raw_code: str, label: int = None):
    """Extract 32 ML features + 512 DL tokens, push Silver parquet"""
    ...

# tasks/gold_tasks.py
@celery_app.task(name="log_prediction_to_gold")
def log_prediction_to_gold(prediction_data: dict):
    """Write prediction result to PostgreSQL Gold table"""
    ...

# tasks/retraining_tasks.py
@celery_app.task(name="check_retrain_trigger")
def check_retrain_trigger():
    """Periodic: check if enough new Silver data to trigger retraining"""
    ...
```

## 3. Core Logic & Integrations

### DagsHub S3 Integration

```python
import boto3

s3_client = boto3.client('s3',
    endpoint_url='https://dagshub.com/user/repo.s3',
    aws_access_key_id=DAGSHUB_TOKEN,
    aws_secret_access_key=DAGSHUB_TOKEN
)

def upload_to_dagshub(bucket, key, data):
    s3_client.put_object(Bucket=bucket, Key=key, Body=data)
```

### Task Flow

```
Django save Bronze → Celery: push_bronze_to_s3 (async)
FastAPI inference done → Celery: extract_and_push_silver + log_prediction_to_gold
Periodic (daily) → Celery Beat: check_retrain_trigger
```

## 4. End-to-End Trace Example

**Sample Input:** Celery receives `push_bronze_to_s3(code_hash="a1b2c3", raw_code="...", metadata={...})`

**Execution Trace:**
```
1. Serialize raw_code + metadata to JSON
2. Upload to DagsHub S3: s3://bronze/2026/04/28/a1b2c3.json
3. Log success to application logger
4. If upload fails → retry 3 times with exponential backoff
```

## 5. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|--------|
| 1 | DagsHub S3 unreachable | Retry 3x, exponential backoff, alert Prometheus |
| 2 | Redpanda broker down | Celery fallback to memory broker, log warning |
| 3 | Large file (>1MB raw code) | Compress with gzip before upload |
| 4 | Duplicate push (same hash) | S3 overwrite is idempotent, no error |
