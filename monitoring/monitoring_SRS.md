# Monitoring — Đặc tả Nghiệp vụ (SRS)

### UC: Giám sát hệ thống và phát hiện bất thường - Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Monitoring giám sát toàn bộ hệ thống: hiệu năng API, GPU usage, model drift, error rate. Cung cấp dashboard real-time và alert tự động.

| Primary Actor: | DevOps / Admin | Secondary Actor: | Prometheus / Grafana |
|----------------|----------------|-------------------|-----------------------|
| **Description:** | Observability stack cho toàn bộ platform |
| **Trigger:** | Tự động (scrape metrics mỗi 15s) |
| **Preconditions:** | PRE1: Prometheus + Grafana deployed trên K8s. PRE2: Apps expose /metrics. |
| **Post-conditions:** | POST1: Dashboards hiển thị real-time. POST2: Alerts gửi khi threshold vượt. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** DevOps truy cập Grafana dashboard
- **Hệ thống xử lý:** Prometheus scrape → Grafana query → render panels
- **Kết quả nhận được:** Dashboard hiển thị latency, throughput, GPU usage, model drift

**Normal Flow:**

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | Apps expose /metrics | Prometheus scrape every 15s |
| 2 | DevOps opens Grafana | Query Prometheus → render dashboard |
| 3 | Latency > 30s sustained | Alert fires → notification (Slack/email) |
| 4 | DevOps investigates | Drill down with Loki logs |

**Exception:**

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | Prometheus down | Grafana shows "No data", alert via uptime check |
| 2 | Metrics endpoint unreachable | Prometheus logs scrape error, alert "target down" |

**Business Rules:**

| No | Rule |
|----|------|
| 1 | P95 inference latency phải < 30s |
| 2 | GPU VRAM usage alert > 14GB (OOM prevention) |
| 3 | Error rate (5xx) phải < 1% trong 5-minute window |
| 4 | Model drift: nếu avg score shift > 20% trong 24h → alert |
| 5 | Dashboard JSON version-controlled trong git |

**Bảng mô tả Data Mapping:**

| Tên trường | Mô tả | Kiểu | Mặc định | Bắt buộc | Ví dụ | Direct to |
|------------|-------|------|----------|----------|-------|-----------|
| inference_latency | P50/P95/P99 | histogram | — | Y | 4.5s (p95) | Grafana panel |
| predictions_total | Total predictions | counter | 0 | Y | 1523 | Grafana panel |
| gpu_memory | VRAM usage | gauge | — | Y | 12.4 GB | Grafana panel |
| cache_hit_rate | Cache efficiency | percentage | — | Y | 23.5% | Grafana panel |
| error_rate | 5xx rate | percentage | — | Y | 0.3% | Alert rule |
