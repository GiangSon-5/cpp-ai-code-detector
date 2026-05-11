# Metadata Implementation Plan — Medallion Architecture

## Tổng Quan

Hệ thống tổ chức dữ liệu theo kiến trúc **Medallion (Bronze → Silver → Gold)**, tích hợp với 2 service:
- **Django Web** (CRUD người dùng, lịch sử, Bronze layer)
- **FastAPI AI Service** (inference, Gold layer, SSE streaming)

Hai pipeline AI riêng biệt:
- **RoBERTa/GraphCodeBERT Ensemble** ← 512 token IDs + LIG attribution + K-Fold
- **LightGBM** ← 32 features tĩnh từ `CppFeatureExtractorV8`
- **Hybrid Fusion** ← α·BERT + (1-α)·LightGBM (α=0.48 tối ưu)

---

## Output JSON Schema (Chiết xuất từ code cũ)

Trường dữ liệu chiết xuất từ `extract_1.py` (line 465-470) và `app.py` (line 177-186):

```json
{
  "final_pred": "AI GENERATED | HUMAN WRITTEN",
  "final_score": 0.8742,
  "model_used": "C++ OOP Model | C++ Normal Model",
  "perplexity": 2.34,
  "max_ppl": 15.42,
  "burstiness": 3.85,
  "is_ambiguous": false,
  "total_tokens": 1024,
  "total_chunks": 2,
  "global_critique": "The code exhibits consistent AI-generated patterns...",
  "global_html": "<html>...(LIG heatmap)...</html>",
  "chunks": [
    {
      "index": 1,
      "score": 0.9123,
      "label": "AI | HUMAN",
      "top_ai": ["'iostream'", "'endl'"],
      "top_hu": ["'ptr'", "'idx'"],
      "snippet": "#include <iostream>...",
      "html": "<html>...(chunk heatmap)...</html>",
      "critique": "This chunk shows typical AI patterns..."
    }
  ]
}
```

### Metadata Flags (từ code cũ)

| Flag | Nguồn | Mô tả |
|------|-------|--------|
| `model_used` | `extract_1.py` L466, L520 | Model được Router chọn: `"C++ OOP Model"` hoặc `"C++ Normal Model"` |
| `is_ambiguous` | `extract_1.py` L481, L544-550 | Judge phát hiện score nằm trong vùng nhập nhằng (0.40-0.60) hoặc PPL conflict |
| `classification` | `extract_1.py` L478 | Router decision: `"OOP"` hoặc `"NORMAL"` |
| `final_pred` | `extract_1.py` L461 | Nhãn cuối: `"AI GENERATED"` hoặc `"HUMAN WRITTEN"` |
| `final_score` | `extract_1.py` L460 | Weighted average probability across chunks |
| `perplexity` | `extract_1.py` L529-530 | Perplexity score trung bình (toàn bài) từ Qwen |
| `max_ppl` | Mới bổ sung | Giá trị Perplexity cao nhất (dòng code làm AI bất ngờ nhất) |
| `burstiness` | Mới bổ sung | Độ biến thiên (Lệch chuẩn) của Perplexity toàn bài |
| `retry_count` | `extract_1.py` L480, L539 | Số lần Judge trigger self-correction (max 1) |
| `threshold` | `extract_1.py` L110, L170 | Default 0.5, có thể load từ `threshold.json` |

---

## 🥉 Tầng BRONZE — Dữ Liệu Thô

### Mục đích
Lưu trữ nguyên bản code C++ thô, không xử lý. Backup trên DagsHub S3, metadata trên PostgreSQL.

### Storage: DagsHub S3 + PostgreSQL

**PostgreSQL Table (Django ORM):**

```python
class BronzeSubmission(models.Model):
    id              = models.AutoField(primary_key=True)
    code_hash       = models.CharField(max_length=64, unique=True, db_index=True)
    user            = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    raw_code        = models.TextField()
    language        = models.CharField(max_length=10, default='cpp')
    source          = models.CharField(max_length=50)  # 'web_upload' | 'api' | 'batch'
    file_size_bytes = models.IntegerField(default=0)
    timestamp       = models.DateTimeField(auto_now_add=True)
    schema_version  = models.CharField(max_length=10, default='1.0')
```

**DagsHub S3 Object (JSONL backup):**

```json
{
  "code_hash": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
  "user_id": 42,
  "raw_code": "#include <iostream>\nusing namespace std;\nint main() {\n    cout << \"Hello\" << endl;\n    return 0;\n}",
  "language": "cpp",
  "source": "web_upload",
  "file_size_bytes": 98,
  "timestamp": "2026-04-28T10:33:40Z",
  "schema_version": "1.0"
}
```

---

## 🥈 Tầng SILVER — Dữ Liệu Đã Xử Lý (DagsHub S3)

### Silver-ML: 32 Features cho LightGBM

**Format:** `.parquet` trên DagsHub S3, managed by DVC

| # | Feature | Kiểu | Mô tả |
|---|---------|------|--------|
| 1 | `empty_line_ratio` | float32 | Tỷ lệ dòng trống |
| 2 | `avg_line_length` | float32 | Độ dài dòng trung bình |
| 3 | `max_line_length` | float32 | Độ dài dòng max |
| 4 | `tab_vs_space_ratio` | float32 | Tỷ lệ tab/space |
| 5 | `brace_style_consistency` | float32 | Nhất quán ngoặc nhọn |
| 6 | `avg_identifier_length` | float32 | Độ dài tên biến TB |
| 7 | `identifier_length_variance` | float32 | Phương sai tên biến |
| 8 | `single_char_var_ratio` | float32 | Tỷ lệ biến 1 ký tự |
| 9 | `unique_identifier_ratio` | float32 | Tỷ lệ định danh duy nhất |
| 10 | `keyword_to_identifier_ratio` | float32 | Tỷ lệ keyword/identifier |
| 11 | `avg_cyclomatic_complexity` | float32 | Độ phức tạp cyclomatic |
| 12 | `num_functions` | float32 | Số lượng hàm |
| 13 | `avg_function_loc` | float32 | Dòng trung bình/hàm |
| 14 | `halstead_volume` | float32 | Halstead volume |
| 15 | `halstead_difficulty` | float32 | Halstead difficulty |
| 16 | `halstead_effort` | float32 | Halstead effort |
| 17 | `halstead_bugs` | float32 | Halstead bug estimate |
| 18 | `maintainability_index` | float32 | Maintainability Index |
| 19 | `code_to_comment_ratio` | float32 | Tỷ lệ code/comment |
| 20 | `max_nesting_depth` | float32 | Độ sâu lồng nhau max |
| 21 | `total_includes` | float32 | Số #include |
| 22 | `has_bits_stdc` | float32 | Dùng bits/stdc++ |
| 23 | `macro_count` | float32 | Số #define |
| 24 | `modern_cpp_ratio` | float32 | Cú pháp C++ hiện đại |
| 25 | `const_usage_ratio` | float32 | Tỷ lệ dùng const |
| 26 | `has_fast_io` | float32 | Dùng fast IO |
| 27 | `newline_style_ratio` | float32 | \\n vs endl |
| 28 | `shannon_entropy` | float32 | Shannon Entropy |
| 29 | `bigram_entropy` | float32 | Bigram Entropy |
| 30 | `whitespace_entropy` | float32 | Whitespace entropy |
| 31 | `comment_ratio` | float32 | Tỷ lệ comment |
| 32 | `trailing_space_ratio` | float32 | Khoảng trắng thừa |
| — | `code_hash` | string | FK → Bronze |
| — | `label` | int8 | 0=Human, 1=AI |

**Ví dụ bản ghi Silver-ML:**
```json
{
  "code_hash": "a1b2c3d4...",
  "empty_line_ratio": 0.12,
  "avg_line_length": 28.5,
  "brace_style_consistency": 0.95,
  "shannon_entropy": 4.23,
  "has_bits_stdc": 0.0,
  "label": 1
}
```

### Silver-DL: 512 Token IDs cho RoBERTa

**Format:** `.parquet` trên DagsHub S3

| Cột | Kiểu | Mô tả |
|-----|------|--------|
| `code_hash` | string | FK → Bronze |
| `input_ids` | list[int32] | 512 token IDs (GraphCodeBERT tokenizer) |
| `attention_mask` | list[int32] | 512 mask (1=real, 0=pad) |
| `chunk_index` | int32 | Chunk index (stride=256) |
| `label` | int8 | 0=Human, 1=AI |

---

## 🥇 Tầng GOLD — Kết Quả Dự Đoán (PostgreSQL)

### Mục đích
Phục vụ Dashboard, báo cáo, analytics. ACID-compliant trên PostgreSQL.

### Schema (SQLAlchemy Async cho FastAPI)

```python
class GoldPrediction(Base):
    __tablename__ = 'gold_predictions'
    id              = Column(Integer, primary_key=True)
    code_hash       = Column(String(64), index=True)
    user_id         = Column(Integer, index=True)
    model_used      = Column(String(30))       # 'C++ OOP Model' | 'C++ Normal Model'
    classification  = Column(String(10))       # 'OOP' | 'NORMAL'
    prediction      = Column(String(20))       # 'AI GENERATED' | 'HUMAN WRITTEN'
    confidence      = Column(Float)            # 0.0 → 1.0 (final_score)
    perplexity      = Column(Float)            # Mean PPL score từ Qwen
    max_ppl         = Column(Float)            # Max PPL (dòng sốc nhất)
    burstiness      = Column(Float)            # Variance PPL
    is_ambiguous    = Column(Boolean)          # Judge flag
    retry_count     = Column(Integer)          # Self-correction count
    total_tokens    = Column(Integer)
    total_chunks    = Column(Integer)
    global_critique = Column(Text)             # LLM Map-Reduce summary
    top_ai_signals  = Column(JSON)             # Top AI tokens
    top_hu_signals  = Column(JSON)             # Top Human tokens
    chunk_details   = Column(JSON)             # Full chunk breakdown
    inference_ms    = Column(Integer)
    timestamp       = Column(DateTime, default=datetime.utcnow)
```

### Ví dụ Gold Record

```json
{
  "id": 5001,
  "code_hash": "a1b2c3d4...",
  "user_id": 42,
  "model_used": "C++ OOP Model",
  "classification": "OOP",
  "prediction": "AI GENERATED",
  "confidence": 0.8742,
  "perplexity": 2.34,
  "max_ppl": 15.42,
  "burstiness": 3.85,
  "is_ambiguous": false,
  "retry_count": 0,
  "total_tokens": 1024,
  "total_chunks": 2,
  "global_critique": "The code consistently uses AI-typical patterns...",
  "top_ai_signals": ["iostream", "endl", "return"],
  "top_hu_signals": ["ptr", "idx"],
  "chunk_details": [
    {"index": 1, "score": 0.91, "label": "AI", "critique": "..."},
    {"index": 2, "score": 0.83, "label": "AI", "critique": "..."}
  ],
  "inference_ms": 4520,
  "timestamp": "2026-04-28T10:34:02Z"
}
```

### Gold Model Performance

```python
class GoldModelPerformance(Base):
    __tablename__ = 'gold_model_performance'
    id            = Column(Integer, primary_key=True)
    model_name    = Column(String(50))
    eval_date     = Column(DateTime)
    accuracy      = Column(Float)
    f1_macro      = Column(Float)
    precision_val = Column(Float)
    recall_val    = Column(Float)
    auc_roc       = Column(Float)
    total_samples = Column(Integer)
    alpha         = Column(Float)        # Fusion weight
    threshold     = Column(Float)        # Decision threshold
    notes         = Column(String(500))
```

---

## Luồng Dữ Liệu Khi User Submit Code

```
[1] User upload code → Django
    → BronzeRepository.save() → PostgreSQL
    → Celery Task: push raw to DagsHub S3 (Bronze)

[2] Django → Redpanda topic `code.submitted` → FastAPI consumes

[3] FastAPI LangGraph Agent:
    ① Router: OOP/Normal classification (Qwen LLM + heuristic fallback)
    ② Analyzer: RoBERTa Ensemble (K-Fold) + LIG + Perplexity
    ③ Judge: Self-correction if ambiguous (score 0.40-0.60 or PPL conflict)
    ④ Critique: Map-Reduce LLM analysis per chunk → global summary

[4] FastAPI → GoldRepository.save_prediction() → PostgreSQL (Gold)
    → Celery Task: extract features → push Silver-ML/DL to DagsHub S3

[5] SSE Stream response → User (real-time progress)
```

---

## Kế Hoạch Thực Thi

| Bước | Công việc | Ưu tiên |
|------|-----------|---------|
| 1 | Django Bronze models + Repository | 🔴 Cao |
| 2 | FastAPI Gold models + Repository (Async) | 🔴 Cao |
| 3 | Engine wrap (RoBERTa + LIG + Router) | 🔴 Cao |
| 4 | ETL Bronze → Silver-ML pipeline | 🔴 Cao |
| 5 | ETL Bronze → Silver-DL pipeline | 🟡 TB |
| 6 | Celery tasks (S3 push, logging) | 🟡 TB |
| 7 | Retraining closed-loop pipeline | 🟡 TB |
| 8 | Infrastructure (Terraform + Ansible) | 🟢 Thấp |
| 9 | Monitoring (Prometheus/Grafana/Loki) | 🟢 Thấp |
| 10 | CI/CD GitHub Actions | 🟢 Thấp |
