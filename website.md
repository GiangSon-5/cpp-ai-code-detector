Viewed run_local.md:1-22

## 🔗 Toàn bộ URL hệ thống

### 👤 User Portal — `localhost:8000`

| URL | Trang | Ghi chú |
|---|---|---|
| `http://localhost:8000/accounts/login/` | 🔐 Login | Trang đầu tiên cần vào |
| `http://localhost:8000/accounts/register/` | 📝 Register | Tạo tài khoản mới |
| `http://localhost:8000/accounts/logout/` | 🚪 Logout | |
| `http://localhost:8000/accounts/profile/` | ⚙️ Settings | Cần login |
| `http://localhost:8000/submit/` | 🧬 Analyze Code | Trang chính — paste/upload C++ |
| `http://localhost:8000/submit/history/` | 📂 Analysis Vault | Lịch sử submissions |
| `http://localhost:8000/submit/result/<hash>/` | 📊 Deep Analysis | Thay `<hash>` bằng code hash thật |

---

### 🛡️ Admin Command Center — `localhost:8000/dashboard/`
> ⚠️ Yêu cầu tài khoản có `is_staff=True`

| URL | Trang | Nội dung |
|---|---|---|
| `http://localhost:8000/dashboard/` | 📡 Live Monitoring | KPI cards, Medallion flow, doughnut chart |
| `http://localhost:8000/dashboard/metrics/` | 📈 Metrics | Chart.js latency, VRAM, confidence dist |
| `http://localhost:8000/dashboard/infra/` | 🏗️ Infrastructure | Healthcheck 6 services thật |
| `http://localhost:8000/dashboard/database/` | 🗄️ Database | All submissions table |
| `http://localhost:8000/dashboard/users/` | 👥 Users | User management + Access Matrix |
| `http://localhost:8000/dashboard/models/` | 🤖 Model Registry | Active model, VRAM, history |
| `http://localhost:8000/dashboard/api/metrics/` | 🔌 Metrics API | JSON endpoint (polled by JS) |

---

### ⚡ FastAPI AI Service — `localhost:8001`

| URL | Mô tả |
|---|---|
| `http://localhost:8001/` | Root info |
| `http://localhost:8001/health` | ✅ Health + latency P95 + GPU stats |
| `http://localhost:8001/docs` | 📖 Swagger UI — test API thủ công |
| `http://localhost:8001/redoc` | 📄 ReDoc — API reference |
| `http://localhost:8001/api/analyze` | POST — sync analysis |
| `http://localhost:8001/api/analyze_stream` | POST — SSE streaming |

---

### 🔧 Django Built-in

| URL | Mô tả |
|---|---|
| `http://localhost:8000/admin/` | Django Admin — set `is_staff`, xem models |

---

### 💡 Quick start gợi ý

```
1. http://localhost:8000/accounts/login/     ← Login
2. http://localhost:8000/submit/             ← Phân tích code
3. http://localhost:8000/dashboard/          ← Admin (cần is_staff)
4. http://localhost:8001/health              ← Kiểm tra FastAPI online
```