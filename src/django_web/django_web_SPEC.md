# Django Web Application — Đặc tả Kỹ thuật (SPEC)

## 1. Module Overview

Django Web Application là tầng giao diện chính, quản lý User Authentication, Bronze Layer CRUD, và Dashboard analytics. Sử dụng Django ORM (ACID-compliant) kết nối PostgreSQL.

**Kế thừa từ legacy code:**
- `app.py` (Streamlit) → Django templates + JavaScript SSE client
- Giao diện metric-box, progress bar, chunk expander → Django templates + CSS

## 2. Data Contracts & Examples

### Bronze Submission (Django ORM → PostgreSQL)

```python
class BronzeSubmission(models.Model):
    code_hash       = models.CharField(max_length=64, unique=True, db_index=True)
    user            = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    raw_code        = models.TextField()
    language        = models.CharField(max_length=10, default='cpp')
    source          = models.CharField(max_length=50)
    file_size_bytes = models.IntegerField(default=0)
    timestamp       = models.DateTimeField(auto_now_add=True)
```

### Apps Structure

```
django_web/
├── apps/
│   ├── accounts/        # User registration, login, profile
│   │   ├── models.py    # UserProfile (extend AbstractUser)
│   │   ├── views.py     # Login, Register, Profile views
│   │   └── urls.py
│   ├── submissions/     # Bronze layer CRUD
│   │   ├── models.py    # BronzeSubmission
│   │   ├── repositories/
│   │   │   └── bronze_repository.py
│   │   ├── views.py     # Submit code, history, detail
│   │   └── urls.py
│   └── dashboard/       # Gold layer analytics
│       ├── views.py     # Stats, charts, model performance
│       └── urls.py
├── templates/           # HTML templates
└── static/              # CSS/JS assets
```

## 3. Core Logic & Integrations

### Bronze Repository (wrap vào Django ORM)

```python
class BronzeRepository:
    def save(self, user, raw_code, source='web') -> BronzeSubmission:
        code_hash = hashlib.sha256(raw_code.encode()).hexdigest()
        obj, created = BronzeSubmission.objects.get_or_create(
            code_hash=code_hash,
            defaults={...}
        )
        return obj

    def get_by_hash(self, code_hash: str): ...
    def get_user_history(self, user, limit=50): ...
```

### Communication with FastAPI

```
Django → Redpanda (topic: code.submitted) → FastAPI consumes
Django → HTTP call to FastAPI /api/analyze_stream (SSE proxy)
```

## 4. End-to-End Trace Example

**Sample Input:** User paste code C++ vào textarea, click "Phân tích"

**Execution Trace:**
```
1. Django view nhận POST form data
2. Validate: code không rỗng, < 500KB
3. BronzeRepository.save() → PostgreSQL (code_hash + raw_code)
4. Celery task: push raw to DagsHub S3 (async, không block)
5. Base64 encode → forward to FastAPI /api/analyze_stream
6. SSE proxy: stream events back to browser JavaScript
7. Browser renders progress bar + results
```

**Sample Output:** Redirect to result page với heatmap + scores

## 5. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|--------|
| 1 | User chưa đăng nhập | Redirect to login, lưu URL redirect |
| 2 | File upload > 500KB | Reject with error message |
| 3 | Duplicate submission | Return cached result từ Bronze |
| 4 | FastAPI service down | Show error page, retry button |
| 5 | Celery broker down | Ghi log trực tiếp, không push S3 |
