# Monitoring — Đặc tả Kỹ thuật (SPEC)

## 1. Module Overview

Monitoring module triển khai observability stack: Prometheus (metrics), Grafana (dashboards), Loki (log aggregation). Giám sát health, performance, và AI model drift.

## 2. Data Contracts & Examples

### Prometheus Metrics (Custom)

```python
# FastAPI custom metrics
ai_inference_duration = Histogram('ai_inference_duration_seconds', 'AI inference latency', ['model_type'])
ai_predictions_total = Counter('ai_predictions_total', 'Total predictions', ['prediction', 'model_used'])
ai_score_distribution = Histogram('ai_score_distribution', 'AI score distribution', buckets=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
cache_hits_total = Counter('cache_hits_total', 'Cache hit count')
gpu_memory_usage = Gauge('gpu_memory_usage_bytes', 'GPU VRAM usage')
```

### Grafana Dashboard Panels

| Panel | Metric | Type |
|-------|--------|------|
| Inference Latency (p50/p95/p99) | ai_inference_duration_seconds | Histogram |
| Predictions per Hour | rate(ai_predictions_total[1h]) | Counter |
| AI vs Human Ratio | ai_predictions_total by prediction | Pie chart |
| GPU Memory Usage | gpu_memory_usage_bytes | Gauge |
| Cache Hit Rate | cache_hits_total / total requests | Percentage |
| Model Score Distribution | ai_score_distribution | Heatmap |
| Error Rate | rate(http_requests_total{status=~"5.."}[5m]) | Line |

### Loki Log Labels

```yaml
{app="fastapi-ai-service", level="INFO"}
{app="django-web", level="ERROR"}
{app="celery-worker", task="push_bronze_to_s3"}
```

## 3. Core Logic & Integrations

### Prometheus Scrape Config

```yaml
# prometheus/prometheus.yml
scrape_configs:
  - job_name: 'fastapi'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['fastapi-ai-service:8001']
  - job_name: 'django'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['django-web:8000']
```

### Alert Rules

```yaml
# prometheus/alert_rules.yml
groups:
  - name: ai-detector-alerts
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, ai_inference_duration_seconds) > 30
        for: 5m
        annotations:
          summary: "P95 inference latency > 30s"
      - alert: GPUMemoryHigh
        expr: gpu_memory_usage_bytes > 14e9
        for: 2m
        annotations:
          summary: "GPU VRAM usage > 14GB (OOM risk)"
      - alert: ModelDrift
        expr: avg_over_time(ai_score_distribution[24h]) < 0.3 or > 0.8
        annotations:
          summary: "Possible model drift detected"
```

## 4. End-to-End Trace Example

**Input:** FastAPI processes 100 requests/hour
**Trace:**
```
1. FastAPI exposes /metrics endpoint (prometheus_fastapi_instrumentator)
2. Prometheus scrapes every 15s → stores TSDB
3. Grafana queries Prometheus → renders dashboards
4. Loki collects logs from all pods → searchable in Grafana
5. Alert fires if P95 latency > 30s → notification channel
```

## 5. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|--------|
| 1 | Prometheus disk full | Retention policy 15 days, auto-compact |
| 2 | Grafana dashboard corrupt | Version-controlled JSON in git |
| 3 | Loki ingestion rate exceeded | Rate limit per tenant, drop oldest |
