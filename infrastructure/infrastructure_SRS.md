# Infrastructure — Đặc tả Nghiệp vụ (SRS)

### UC: Triển khai và quản lý hạ tầng - Hệ thống AI Code Detector

**Mô tả chức năng tổng quan**
Module Infrastructure tự động hóa việc cài đặt server (OS, GPU drivers, K8s), quản lý tài nguyên Kubernetes qua Terraform, và public web qua Cloudflare Tunnels.

| Primary Actor: | DevOps Engineer | Secondary Actor: | MicroK8s / Cloudflare |
|----------------|-----------------|-------------------|-----------------------|
| **Description:** | IaC cho bare-metal → production-ready K8s cluster |
| **Trigger:** | DevOps chạy Ansible playbook hoặc Terraform apply |
| **Preconditions:** | PRE1: Ubuntu 22.04 với NVIDIA GPU. PRE2: SSH access. PRE3: Cloudflare account. |
| **Post-conditions:** | POST1: K8s cluster ready. POST2: All services deployed. POST3: Web publicly accessible. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** Server Ubuntu mới với GPU NVIDIA
- **Hệ thống xử lý:** Ansible cài drivers + K8s → Terraform deploy services → Cloudflare tunnel
- **Kết quả nhận được:** Hệ thống AI Code Detector chạy production trên K8s, accessible via HTTPS

**Normal Flow:**

| Step | Actor Action | System Response |
|------|-------------|-----------------|
| 1 | DevOps cấu hình inventory/hosts.yml | Define target server IP, GPU info |
| 2 | DevOps chạy Ansible playbook | Install NVIDIA driver, CUDA, MicroK8s |
| 3 | DevOps chạy Terraform apply (dev) | Deploy PostgreSQL, Redpanda, Grafana via Helm |
| 4 | DevOps apply K8s manifests | Deploy Django, FastAPI, Celery pods |
| 5 | DevOps verify | Check pods running, GPU allocated, web accessible |

**Exception:**

| No | Cause | System Response |
|----|-------|-----------------|
| 1 | GPU driver incompatible | Ansible fallback to older driver version |
| 2 | Helm chart version mismatch | Pin chart versions in Terraform |
| 3 | Cloudflare API error | Manual tunnel setup fallback |

**Business Rules:**

| No | Rule |
|----|------|
| 1 | GPU chỉ allocate cho FastAPI pod (inference) |
| 2 | Django và Celery chạy CPU-only |
| 3 | PostgreSQL PVC tối thiểu 10Gi |
| 4 | Cloudflare Tunnel chạy systemd daemon (auto-restart) |
| 5 | Terraform state lưu remote (DagsHub S3) |

**Bảng mô tả Data Mapping:**

| Tên trường | Mô tả | Kiểu | Mặc định | Bắt buộc | Ví dụ | Direct to |
|------------|-------|------|----------|----------|-------|-----------|
| ansible_host | Server IP | string | — | Y | 192.168.1.100 | SSH |
| gpu_driver_version | NVIDIA driver | string | "535" | Y | "535" | apt install |
| microk8s_channel | K8s version | string | "1.28/stable" | Y | "1.28/stable" | snap |
| namespace | K8s namespace | string | "ai-detector" | Y | "ai-detector" | Terraform |
| cloudflare_token | Tunnel auth | string | — | Y | "eyJ..." | cloudflared |
