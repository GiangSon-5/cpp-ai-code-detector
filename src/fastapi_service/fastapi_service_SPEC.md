# FastAPI AI Service — Đặc tả Kỹ thuật (SPEC)

## 1. Module Overview

FastAPI AI Service là microservice chịu trách nhiệm toàn bộ inference AI. Nó wrap logic xử lý từ `extract_1.py` (legacy) vào kiến trúc enterprise với SQLAlchemy Async, Repository Pattern, Pydantic validation, và SSE streaming.

**Kế thừa từ legacy code:**
- `extract_1.py` → `engine/` (RoBERTa, LIG, Router, Judge, Critique)
- Heuristic OOP classifier → `engine/heuristic_classifier.py`
- LocalLLMHandler → `engine/llm_handler.py`
- ModelManager → `engine/model_manager.py`
- ExpertExplainer + render_html_view → `engine/explainer.py`
- LangGraph workflow → `services/agent_service.py`

## 2. Data Contracts & Examples

### Request Schema (Pydantic)

```python
class AnalyzeRequest(BaseModel):
    code_base64: str = Field(..., min_length=4, description="Base64-encoded C++ source code")

class AnalyzeResponse(BaseModel):
    final_pred: str          # "AI GENERATED" | "HUMAN WRITTEN"
    final_score: float       # 0.0 - 1.0
    model_used: str          # "C++ OOP Model" | "C++ Normal Model"
    perplexity: float        # PPL score
    is_ambiguous: bool
    total_tokens: int
    total_chunks: int
    global_critique: str
    global_html: str
    chunks: list[ChunkDetail]

class ChunkDetail(BaseModel):
    index: int
    score: float
    label: str               # "AI" | "HUMAN"
    top_ai: list[str]
    top_hu: list[str]
    snippet: str
    html: str
    critique: str
```

## 3. Core Logic & Integrations

### Cách wrap code cũ vào kiến trúc mới

```
extract_1.py (monolithic)
    │
    ├── LocalLLMHandler class      → engine/llm_handler.py
    ├── ModelManager class         → engine/model_manager.py  
    ├── heuristic_classify()       → engine/heuristic_classifier.py
    ├── ExpertExplainer class      → engine/explainer.py
    ├── render_html_view()         → engine/html_renderer.py
    ├── run_roberta_engine()       → engine/roberta_engine.py
    ├── LangGraph nodes            → services/agent_service.py
    ├── FastAPI endpoints          → routers/predict.py
    ├── CodeRequest schema         → schemas/prediction_schema.py
    └── RESULT_CACHE               → Redpanda cache layer
```

### Repository Pattern

```python
# repositories/gold_repository.py
class GoldRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_prediction(self, data: PredictionCreate) -> GoldPrediction: ...
    async def get_by_hash(self, code_hash: str) -> GoldPrediction | None: ...
    async def get_user_stats(self, user_id: int) -> dict: ...
    async def list_recent(self, limit: int = 50) -> list[GoldPrediction]: ...
```

## 4. End-to-End Trace Example

**Sample Input:**
```
POST /api/analyze_stream
Content-Type: application/json

{
  "code_base64": "I2luY2x1ZGUgPGlvc3RyZWFtPgp1c2luZyBuYW1lc3BhY2Ugc3RkOwppbnQgbWFpbigpIHsKICAgIGNvdXQgPDwgIkhlbGxvIiA8PCBlbmRsOwogICAgcmV0dXJuIDA7Cn0="
}
```

**Execution Trace:**
```
1. [Router] Decode Base64 → "#include <iostream>..."
2. [Router] Qwen LLM classify → "NORMAL" (no class/virtual keywords)
3. [Router] Fallback heuristic score = 0 → confirms "NORMAL"
4. [Analyzer] Load C++ Normal Model (5 folds)
5. [Analyzer] Tokenize → 45 tokens, 1 chunk
6. [Analyzer] RoBERTa Ensemble: fold_1=0.87, fold_2=0.91, fold_3=0.85, fold_4=0.89, fold_5=0.88
7. [Analyzer] LIG attribution → 45 scores per fold → averaged
8. [Analyzer] Perplexity = 1.82 (low → AI-like)
9. [Judge] Score=0.88 > 0.60, PPL=1.82 < 5.0 → NOT ambiguous → Accept
10. [Critique MAP] Gemini analyzes chunk 1: "boilerplate iostream pattern..."
11. [Critique REDUCE] Global summary: "Code is AI-generated with high confidence"
12. [Response] SSE stream with progress events → final JSON
```

**Sample Output:**
```json
{
  "final_pred": "AI GENERATED",
  "final_score": 0.8800,
  "model_used": "C++ Normal Model",
  "perplexity": 1.82,
  "is_ambiguous": false,
  "total_tokens": 45,
  "total_chunks": 1,
  "global_critique": "Code is AI-generated with high confidence. Consistent naming and boilerplate.",
  "global_html": "<html>...(heatmap)...</html>",
  "chunks": [{
    "index": 1, "score": 0.8800, "label": "AI",
    "top_ai": ["'iostream'", "'endl'", "'return'"],
    "top_hu": [],
    "snippet": "#include <iostream>...",
    "html": "<html>...</html>",
    "critique": "Typical AI boilerplate with consistent formatting."
  }]
}
```

## 5. Edge Cases

| # | Tình huống | Xử lý |
|---|-----------|--------|
| 1 | Code rỗng hoặc chỉ có whitespace | Return HTTP 400 `{"error": "Empty code"}` |
| 2 | Base64 decode thất bại | Return HTTP 422 validation error |
| 3 | Code quá dài (>50000 tokens) | Chunk với stride=256, xử lý tuần tự, tăng timeout |
| 4 | GPU OOM khi chạy LIG | Catch exception, fallback chạy inference-only (không LIG) |
| 5 | LLM API key hết hạn | Fallback chain: Gemini → OpenAI → Local Qwen → "unavailable" |
| 6 | Score nằm vùng nhập nhằng (0.40-0.60) | Judge trigger self-correction, đổi model OOP↔Normal |
| 7 | PPL conflict với score | Judge trigger self-correction (e.g., high score + high PPL) |
| 8 | Duplicate code submission | Cache hit via code_hash (MD5) → return cached result |
| 9 | Model files không tồn tại | Return HTTP 503 "Model not loaded" |
| 10 | Redpanda connection lost | Fallback ghi log trực tiếp PostgreSQL, retry queue |
