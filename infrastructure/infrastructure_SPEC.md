# Infrastructure — Đặc tả Kỹ thuật (SPEC)

## 1. Module Overview

Infrastructure module quản lý toàn bộ hạ tầng: Ansible (bare-metal setup), Terraform (K8s resources), và Kubernetes manifests. Triển khai trên Ubuntu + MicroK8s với GPU support.

## 2. Data Contracts & Examples

### Ansible Inventory

```yaml
# inventory/hosts.yml
all:
  hosts:
    gpu-server:
      ansible_host: 192.168.1.100
      ansible_user: admin
      gpu_driver_version: "535"
      microk8s_channel: "1.28/stable"
```

### Terraform Variables

```hcl
# environments/dev/terraform.tfvars
cluster_name     = "ai-detector-dev"
namespace        = "ai-detector"
postgresql_size  = "10Gi"
redpanda_replicas = 1
gpu_enabled      = true
dagshub_s3_endpoint = "https://dagshub.com/user/repo.s3"
cloudflare_tunnel_token = "eyJ..."
```

## 3. Core Logic & Integrations

### Ansible Roles

```
roles/
├── nvidia_driver/     # Install NVIDIA driver + CUDA toolkit
│   └── tasks/main.yml # apt install nvidia-driver-535, nvidia-cuda-toolkit
├── microk8s/          # Install MicroK8s + addons
│   └── tasks/main.yml # snap install microk8s, enable gpu dns storage
└── cloudflare_tunnel/ # Install cloudflared daemon
    └── tasks/main.yml # cloudflared tunnel create, systemd service
```

### Terraform Modules

```
modules/
├── k8s/               # Kubernetes resources
│   ├── namespace.tf
│   ├── secrets.tf     # DB credentials, API keys, DagsHub tokens
│   ├── configmap.tf   # App config, model paths
│   └── gpu_quota.tf   # GPU resource limits
└── helm/              # Helm chart releases
    ├── postgresql.tf  # Bitnami PostgreSQL chart
    ├── redpanda.tf    # Redpanda Helm chart
    ├── grafana.tf     # Grafana + Prometheus stack
    └── loki.tf        # Grafana Loki for logs
```

### K8s Manifests

```yaml
# base/fastapi-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-ai-service
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: fastapi
        image: ai-detector/fastapi-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            memory: "4Gi"
        ports:
        - containerPort: 8001
```

## 4. End-to-End Trace Example

**Input:** Fresh Ubuntu 22.04 server with NVIDIA GPU

**Trace:**
```
1. ansible-playbook playbooks/setup_server.yml
   → Install NVIDIA Driver 535 + CUDA
   → Install MicroK8s + enable gpu,dns,storage addons
   → Install cloudflared + create tunnel

2. cd terraform/environments/dev && terraform apply
   → Create namespace "ai-detector"
   → Deploy PostgreSQL via Helm (10Gi PVC)
   → Deploy Redpanda via Helm (single node)
   → Deploy Grafana + Prometheus + Loki
   → Create secrets (DB creds, API keys)

3. kubectl apply -k k8s_manifests/overlays/dev/
   → Deploy Django (2 replicas, no GPU)
   → Deploy FastAPI (1 replica, 1 GPU)
   → Deploy Celery worker (1 replica, no GPU)
   → Ingress via Cloudflare Tunnel
```

## 5. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|--------|
| 1 | GPU driver install fails | Ansible retry with different driver version |
| 2 | MicroK8s GPU addon not available | Fallback CPU-only mode |
| 3 | Terraform state corruption | Remote state on S3 with locking |
| 4 | Cloudflare tunnel drops | systemd auto-restart, health check |
| 5 | PVC storage full | Alert via Prometheus, auto-expand if supported |
