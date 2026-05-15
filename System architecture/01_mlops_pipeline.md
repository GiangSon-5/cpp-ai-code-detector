# 🤖 Tài Liệu Nội Bộ — Bước 2: MLOps Pipeline Chi Tiết

> **Phạm vi:** FastAPI AI Service — pipeline suy luận AI từ đầu đến cuối  
> **File chính:** `src/fastapi_service/services/agent_service.py`

---

## 1. Tổng Quan Pipeline — 5 Node Tuần Tự

```
Code C++ (raw_code)
        │
        ▼
┌──────────────────────────────────────────────────┐
│  NODE ①  ROUTER                                  │
│  LLM classify_code() + heuristic_classify()       │
│  → classification: "OOP" | "NORMAL"              │
│  → model_used: "C++ OOP Model" | "C++ Normal"    │
└────────────────────────┬─────────────────────────┘
                         │
        ┌────────────────┼────────────────────┐
        │ (parallel)                          │
        ▼                                     ▼
┌──────────────────┐              ┌──────────────────────┐
│  NODE ②a          │              │  NODE ②b              │
│  RoBERTaEngine   │              │  LLMHandler           │
│  .analyze()      │              │  .compute_perplexity()│
│  (→ Colab/GPU)   │              │  (→ vLLM Qwen 7B)    │
│  mean_score,     │              │  perplexity,          │
│  chunks, html    │              │  max_ppl, burstiness  │
└─────────┬────────┘              └────────┬─────────────┘
          │                               │
          └──────────────┬────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│  NODE ③  JUDGE  (Self-Correction)                │
│  Phát hiện vùng nhập nhằng → thử model ngược     │
│  Ambiguous: 0.40 ≤ score ≤ 0.60                  │
│  PPL Conflict: score > 0.60 AND ppl > 5.0        │
│  → final_score, is_ambiguous, retry_count        │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│  NODE ④  CRITIQUE  (Map-Reduce LLM)              │
│  Mỗi chunk → generate_critique() [Map]           │
│  Tất cả critique → generate_global_critique()[Reduce]│
│  → chunk.critique, global_critique               │
└────────────────────────┬─────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────┐
│  NODE ⑤  FINGERPRINT XAI  (Local, no GPU)       │
│  CppFeatureExtractorV8.extract()                 │
│  → 32 features → LightGBM.predict_proba()        │
│  → SHAP.shap_values() → ShapFeature list         │
│  → FingerprintResult (executive_summary, top5)   │
└────────────────────────┬─────────────────────────┘
                         ▼
              AnalyzeResponse (JSON)
              GoldPredictionRecord (DB)
```

---

## 2. Node ① — Router: Phân Loại OOP/NORMAL

**File:** `agent_service._node_router()`  
**Mục đích:** Chọn đúng model checkpoint phù hợp với loại code

### Chiến lược quyết định

```
1. LLM classify_code()    → "OOP" hoặc "NORMAL" (JSON structured)
2. heuristic_classify()   → "OOP" hoặc "NORMAL" (regex score)

Logic:
  - Nếu LLM == Heuristic   → dùng kết quả đó
  - Nếu LLM unavailable    → tin heuristic
  - Nếu bất đồng + heuristic score > 0.4 → tin heuristic
  - Còn lại               → tin LLM
```

### Heuristic Classifier (`heuristic_classifier.py`)

Chấm điểm dựa trên regex pattern:

| Pattern | Điểm | Mô tả |
|---------|------|-------|
| `class \w+` | +0.25 | Khai báo class |
| `virtual` | +0.15 | Virtual function |
| `public:` | +0.10 | Access specifier |
| `protected:` | +0.08 | Access specifier |
| `private:` | +0.08 | Access specifier |
| `: public \w+` | +0.12 | Kế thừa |
| `override` | +0.08 | Override keyword |
| `template<` | +0.06 | Template |
| Operator overload | +0.05 | `operator+`, v.v. |
| `namespace` | +0.03 | Namespace |

**Ngưỡng:** `score ≥ 0.25` → OOP, còn lại → NORMAL

---

## 3. Node ② — Analyzer: RoBERTa Ensemble

**File:** `agent_service._node_analyzer()` → `roberta_engine.analyze()`  
**File:** `engine/roberta_engine.py`

> ⚠️ **Quan trọng:** `RoBERTaEngine` là **HTTP client**, KHÔNG chạy model local!  
> Nó gọi POST tới `FASTAPI_AI_URL/api/predict/roberta` (Colab hoặc Local GPU Server).

### Request flow

```
Orchestrator (8001)
    │ POST /api/predict/roberta
    │ { code_base64: "...", model_type: "OOP"|"NORMAL" }
    │ Header: X-API-Key: <LOCAL_GPU_API_KEY hoặc COLAB_API_KEY>
    ▼
Local GPU Server (8002) hoặc Colab
    │  RoBERTa Ensemble K-Fold
    │  Layer Integrated Gradients (LIG)
    │  trả về: { final_pred, final_score, total_tokens,
    │            total_chunks, chunks:[{index, score, label,
    │                                   tokens, attrs, snippet}] }
    ▼
Orchestrator parse response
    → render_html_heatmap(tokens, attrs)
    → ChunkResult list
    → mean_score, global_html
```

### Kết quả trả về từ Analyzer

| Field | Ý nghĩa |
|-------|---------|
| `mean_score` | Trung bình AI probability qua các chunk (0-1) |
| `chunks` | List `ChunkResult` (mỗi chunk 512 token) |
| `total_tokens` | Tổng token sau tokenization |
| `total_chunks` | Số chunk (code dài sẽ có nhiều chunk) |
| `global_html` | HTML heatmap attribution (LIG visualization) |

---

## 4. Node ②b — Perplexity: Qwen 2.5 Coder vLLM

**File:** `llm_handler.compute_perplexity()`  
**Chạy song song** với RoBERTa Analyzer (asyncio await cả 2)

### Cách tính Perplexity

```python
# Gửi code tới vLLM với prompt_logprobs=1
# → nhận logprobs của từng token

logprobs_list = [...]  # log P(token_i | context)

# Bỏ 30 token đầu (instructions noise)
# Clip logprob tại -10.0 (tránh PPL hàng tỷ từ tên biến lạ)
clipped = np.clip(logprobs_list[30:], -10.0, 0.0)

perplexity = exp(-mean(clipped))    # Trung bình (thấp = predictable = AI)
max_ppl     = max(exp(-lp) for lp)  # Dòng "bất ngờ" nhất
burstiness  = std(exp(-lp) for lp)  # Biến thiên PPL
```

### Ý nghĩa thực tế

| Metric | Giá trị thấp | Giá trị cao |
|--------|-------------|------------|
| `perplexity` | Code rất "predictable" → AI-like | Khó đoán → Human-like |
| `max_ppl` | Mọi dòng đều thông thường | Có dòng code cực kỳ lạ |
| `burstiness` | PPL đồng đều toàn bài | Biến thiên lớn giữa các đoạn |

---

## 5. Node ③ — Judge: Tự Sửa Lỗi (Self-Correction)

**File:** `agent_service._node_judge()`

### Điều kiện kích hoạt

```python
in_ambiguous_zone = 0.40 <= mean_score <= 0.60
ppl_conflict      = mean_score > 0.60 AND perplexity > 5.0

if in_ambiguous_zone or ppl_conflict:
    # Thử model ngược (OOP↔NORMAL)
    opposite = "OOP" if current == "NORMAL" else "NORMAL"
    retry_result = roberta_engine.analyze(code, model_type=opposite, enable_lig=False)
    
    # Chỉ chấp nhận kết quả mới nếu nó TỰ TIN HƠN (xa 0.5 hơn)
    if abs(retry_score - 0.5) > abs(mean_score - 0.5):
        final_score = retry_score  # Chấp nhận self-correction
    else:
        final_score = mean_score   # Giữ nguyên kết quả cũ
```

### Kết quả

| Field | Mô tả |
|-------|-------|
| `final_score` | Score sau khi Judge (có thể khác mean_score) |
| `is_ambiguous` | True nếu Judge đã can thiệp |
| `retry_count` | Số lần thử lại (tối đa 1) |
| `model_switched` | True nếu đã chuyển sang model khác |

**Quyết định cuối:** `prediction = "AI GENERATED" if final_score >= 0.5 else "HUMAN WRITTEN"`

---

## 6. Node ④ — Critique: LLM Map-Reduce

**File:** `agent_service._node_critique()` + `llm_handler.generate_critique/generate_global_critique()`

### LLM Fallback Chain

Tất cả LLM calls đều thử theo thứ tự:

```
1. vLLM (Qwen 2.5 Coder 7B) — qua Colab proxy /api/proxy/vllm
2. Gemini 2.0 Flash           — Google Generative Language API
3. OpenAI GPT-4o-mini         — OpenAI Chat API
4. Fallback: "Analysis unavailable"
```

### Các nhiệm vụ của LLM

| Hàm | Input | Output |
|-----|-------|--------|
| `classify_code()` | Code C++ | `{"classification": "OOP"|"NORMAL", "reasoning": "..."}` |
| `generate_critique()` | chunk snippet + score + label | Đoạn văn 2-3 câu giải thích |
| `generate_global_critique()` | Tất cả chunk critiques | Tóm tắt toàn bộ 3-5 câu (Map-Reduce) |
| `compute_perplexity()` | Code (max 3000 chars) | `{perplexity, max_ppl, burstiness}` |

---

## 7. Node ⑤ — Fingerprint XAI: LightGBM + SHAP

**File:** `engine/fingerprint_engine.py`  
**Chạy local, không cần GPU** — `FingerprintEngine`

### Pipeline

```
Code C++ (raw)
     │
     ▼
CppFeatureExtractorV8.extract()   ← lizard, numpy, regex
     │ → dict of 32 float features
     ▼
StandardScaler.transform()         ← scaler.pkl
     │ → scaled feature vector
     ▼
LightGBM.predict_proba()           ← LightGBM_Regulated.pkl
     │ → prob_ai (0-1)
     ▼
shap.TreeExplainer.shap_values()
     │ → shap_values per feature
     ▼
Sort by |SHAP|, top-5 features
     │ + behavioral insight (tiếng Việt)
     ▼
FingerprintResult(
    lgbm_score, lgbm_prediction,
    executive_summary,
    top_features, all_features
)
```

### Model Artifacts (tại `local_models/saved_models/`)

| File | Mô tả |
|------|-------|
| `LightGBM_Regulated.pkl` | Model LightGBM đã train |
| `scaler.pkl` | StandardScaler (fit trên training data) |
| `final_features.pkl` | Danh sách features LightGBM sử dụng |
| `baselines.json` | Giá trị trung bình feature của AI vs Human |

### 32 Features — Phân Loại

#### A. Layout & Formatting (6 features)
| Feature | Ý nghĩa AI | Ý nghĩa Human |
|---------|-----------|--------------|
| `empty_line_ratio` | 1 dòng trống giữa blocks, nhất quán | Nhiều dòng trống tùy tiện |
| `avg_line_length` | Đồng đều ~80-100 ký tự | Biến động lớn, lúc ngắn lúc dài |
| `tab_vs_space_ratio` | Luôn dùng Space (100%) | Trộn Tab và Space |
| `trailing_space_ratio` | Không có khoảng trắng thừa | Hay gõ thừa space cuối dòng |
| `brace_style_consistency` | Nhất quán 1 kiểu | Lộn xộn K&R vs Allman |
| `max_line_length` | Không vượt 100 ký tự | Đôi khi dòng rất dài |

#### B. Naming Conventions (5 features)
| Feature | Ý nghĩa AI | Ý nghĩa Human |
|---------|-----------|--------------|
| `single_char_var_ratio` | Đặt tên dài có ý nghĩa | Lạm dụng `i, j, n, k` |
| `avg_identifier_length` | Tên biến dài (descriptive) | Tên viết tắt, ngắn |
| `unique_identifier_ratio` | Mỗi biến tên riêng | Tái dùng `i, j` nhiều lần |
| `identifier_length_variance` | Đồng đều về độ dài tên | Biến thiên lớn |
| `keyword_to_identifier_ratio` | Ít keyword/nhiều biến custom | Nhiều keyword/ít biến custom |

#### C. Structural Complexity (8 features — Halstead + Cyclomatic)
| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `avg_cyclomatic_complexity` | Từ lizard | AI chia nhỏ hàm → CC thấp |
| `num_functions` | Từ lizard | AI tạo nhiều hàm helper |
| `avg_function_loc` | Từ lizard | AI: hàm ngắn (~20 dòng) |
| `halstead_volume` | `N * log2(n)` | Độ phức tạp thông tin |
| `halstead_difficulty` | `(n1/2) * (N2/n2)` | Khó hiểu hay dễ đọc |
| `halstead_effort` | `D * V` | Công sức đọc hiểu |
| `halstead_bugs` | `V / 3000` | Ước tính bug density |
| `maintainability_index` | `171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC)` | AI: MI cao |

#### D. Coding Habits (7 features)
| Feature | Đặc trưng Human | Đặc trưng AI |
|---------|----------------|-------------|
| `has_bits_stdc` | Hay dùng bits/stdc++.h | Ít dùng |
| `macro_count` | Lạm dụng #define | Theo Modern C++ |
| `modern_cpp_ratio` | Ít dùng auto, nullptr, ... | Dùng nhiều Modern C++ |
| `const_usage_ratio` | Ít const | Dùng const đúng chỗ |
| `has_fast_io` | Hay dùng ios_base::sync | Dùng ít hơn |
| `newline_style_ratio` | Hay dùng `\n` | Hay dùng `endl` |
| `total_includes` | Bừa bãi include | Include chính xác |

#### E. Information Theory (3 features)
| Feature | Ý nghĩa |
|---------|---------|
| `shannon_entropy` | Độ đa dạng ký tự — AI cao hơn |
| `bigram_entropy` | Đa dạng cặp ký tự kế nhau |
| `whitespace_entropy` | Mẫu khoảng trắng — AI nhất quán hơn |

---

## 8. Hybrid Score Computation

```python
dl_score    = roberta_engine.mean_score          # RoBERTa (0-1)
ml_score    = fingerprint_result.lgbm_score      # LightGBM (0-1)
hybrid_score = 0.6 * dl_score + 0.4 * ml_score  # Trọng số: DL:60%, ML:40%

# final_score là dl_score đã qua Judge (không phải hybrid)
# hybrid_score chỉ dùng để hiển thị trên UI
```

> **Lưu ý:** `final_pred` được quyết định dựa trên `final_score` (từ Judge), không phải `hybrid_score`.

---

## 9. Schemas & Data Contracts

**File:** `src/shared/data_contracts.py` — **Single Source of Truth** cho toàn hệ thống

```
BronzeRecord           ← Django → S3 (raw submission)
SilverMLRecord         ← 32 features → S3/LightGBM training
SilverDLRecord         ← 512 token IDs → S3/RoBERTa training
ChunkResult            ← Shared: FastAPI ↔ Django ↔ Celery
GoldPredictionRecord   ← FastAPI → PostgreSQL (full result)
FingerprintResult      ← LightGBM + SHAP result
ShapFeature            ← Từng feature trong SHAP
CodeSubmittedEvent     ← Redpanda topic: code.submitted
PredictionCompletedEvent ← Redpanda topic: prediction.completed
```

**File:** `src/fastapi_service/schemas/prediction_schema.py` — FastAPI-specific schemas

```
AnalyzeRequest     ← POST /api/analyze_stream body
AnalyzeResponse    ← Final JSON to client
HealthResponse     ← GET /health response
SSEProgressEvent   ← Từng event trong SSE stream
```

---

## 10. Cấu Hình Ngưỡng (Settings)

**File:** `src/shared/config.py` — Singleton `settings`

| Constant | Giá trị | Ý nghĩa |
|----------|---------|---------|
| `DEFAULT_THRESHOLD` | 0.5 | score ≥ 0.5 → AI GENERATED |
| `AMBIGUOUS_LOW` | 0.40 | Dưới ngưỡng này Judge can thiệp |
| `AMBIGUOUS_HIGH` | 0.60 | Trên ngưỡng này không còn nhập nhằng |
| `FUSION_ALPHA` | 0.48 | α trong α·BERT + (1-α)·LightGBM (dùng trong training) |

> **Lưu ý thực tế:** `FUSION_ALPHA=0.48` là thông số huấn luyện, nhưng trong inference hiện tại hệ thống dùng `0.6*DL + 0.4*ML` cho hybrid_score display, và `final_score = judge(roberta_score)` cho quyết định.

---

## 11. SSE Streaming (Realtime Progress)

Khi Django gọi FastAPI để phân tích, kết quả được stream về theo từng bước:

```
 5% → step: "router"    | "Classifying code type..."
20% → step: "router"    | "Code classified as OOP"
25% → step: "analyzer"  | "Running RoBERTa Ensemble..."
65% → step: "analyzer"  | "Ensemble complete — score 0.8742, PPL 2.34"
68% → step: "judge"     | "Evaluating confidence..."
75% → step: "judge"     | "Judge: confident"
78% → step: "critique"  | "Generating LLM analysis..."
90% → step: "critique"  | "Generating XAI fingerprint..."
100%→ step: "complete"  | "Done in 4520ms" + full AnalyzeResponse JSON
```

---

## Tóm Tắt Nhanh — Cheat Sheet

| Component | File | Vai trò |
|-----------|------|---------|
| `AgentService` | `services/agent_service.py` | Điều phối 5 node |
| `RoBERTaEngine` | `engine/roberta_engine.py` | HTTP client → Colab/GPU |
| `LLMHandler` | `engine/llm_handler.py` | vLLM → Gemini → OpenAI |
| `FingerprintEngine` | `engine/fingerprint_engine.py` | LightGBM + SHAP (local) |
| `CppFeatureExtractorV8` | `engine/feature_extractor/extractor.py` | 32 features từ regex+lizard |
| `ExpertExplainer` | `engine/explainer.py` | LIG attribution (on GPU server) |
| `heuristic_classify` | `engine/heuristic_classifier.py` | Fallback OOP/NORMAL |
| `ModelManager` | `engine/model_manager.py` | Proxy status to Colab |
| `data_contracts` | `shared/data_contracts.py` | Schema toàn hệ thống |
| `settings` | `shared/config.py` | Config singleton |

---

## Bước Tiếp Theo

| Bước | Nội dung | Trạng thái |
|------|----------|------------|
| **Bước 1** | Tổng quan hệ thống, Tech Stack, Medallion Architecture | ✅ Hoàn thành |
| **Bước 2** | MLOps Pipeline: LangGraph 5-node, RoBERTa, LightGBM, XAI | ✅ Hoàn thành |
| **Bước 3** | Admin Dashboard: 6 views, metrics API, infra healthcheck | 🔲 Chờ review |
| **Bước 4** | Shared infrastructure: Logger, Config, Data Contracts, S3, Celery | 🔲 |
| **Bước 5** | Vận hành: khởi động, debugging, monitoring, CI/CD | 🔲 |
