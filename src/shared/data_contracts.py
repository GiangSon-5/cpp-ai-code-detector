"""
shared/data_contracts.py — Global Data Contracts (Pydantic v2)

These schemas are the SINGLE SOURCE OF TRUTH shared across Django, FastAPI,
Celery workers, and the data pipeline.  Every module that produces or consumes
data MUST use these contracts to guarantee schema compatibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# =====================================================================
# XAI — Feature-based explainability (LightGBM + SHAP)
# =====================================================================
class ShapFeature(BaseModel):
    """Single SHAP-explained feature for LightGBM XAI output."""
    name: str = Field(..., description="Feature identifier e.g. 'whitespace_entropy'")
    display_name: str = Field("", description="Human-readable label")
    value: float = Field(0.0, description="Raw extracted feature value")
    shap_value: float = Field(0.0, description="SHAP contribution to AI probability")
    direction: str = Field("AI", description="'AI' if shap > 0, 'HUMAN' if shap < 0")
    human_baseline: float = Field(0.0)
    ai_baseline: float = Field(0.0)
    insight: str = Field("", description="Behavioral explanation in Vietnamese")


class FingerprintResult(BaseModel):
    """LightGBM + SHAP fingerprint analysis result — human-readable XAI."""
    lgbm_score: float = Field(0.5, ge=0.0, le=1.0)
    lgbm_prediction: str = Field("HUMAN WRITTEN")
    executive_summary: str = Field("")
    top_features: list[ShapFeature] = Field(default_factory=list)
    all_features: list[ShapFeature] = Field(default_factory=list)


# =====================================================================
# BRONZE — Raw code submission
# =====================================================================
class BronzeRecord(BaseModel):
    """Represents a raw code submission (Bronze layer)."""
    code_hash: str = Field(..., max_length=64, description="SHA-256 of raw_code")
    user_id: Optional[int] = Field(None, description="FK to auth_user")
    raw_code: str = Field(..., description="Unprocessed C++ source code")
    language: str = Field("cpp", max_length=10)
    source: str = Field("web_upload", max_length=50, description="web_upload | api | batch")
    file_size_bytes: int = Field(0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    schema_version: str = Field("1.0", max_length=10)


# =====================================================================
# SILVER-ML — 32 Features for LightGBM
# =====================================================================
class SilverMLRecord(BaseModel):
    """32 static features extracted by CppFeatureExtractorV8."""
    code_hash: str
    empty_line_ratio: float = 0.0
    avg_line_length: float = 0.0
    max_line_length: float = 0.0
    tab_vs_space_ratio: float = 0.0
    brace_style_consistency: float = 0.0
    avg_identifier_length: float = 0.0
    identifier_length_variance: float = 0.0
    single_char_var_ratio: float = 0.0
    unique_identifier_ratio: float = 0.0
    keyword_to_identifier_ratio: float = 0.0
    avg_cyclomatic_complexity: float = 0.0
    num_functions: float = 0.0
    avg_function_loc: float = 0.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_bugs: float = 0.0
    maintainability_index: float = 0.0
    code_to_comment_ratio: float = 0.0
    max_nesting_depth: float = 0.0
    total_includes: float = 0.0
    has_bits_stdc: float = 0.0
    macro_count: float = 0.0
    modern_cpp_ratio: float = 0.0
    const_usage_ratio: float = 0.0
    has_fast_io: float = 0.0
    newline_style_ratio: float = 0.0
    shannon_entropy: float = 0.0
    bigram_entropy: float = 0.0
    whitespace_entropy: float = 0.0
    comment_ratio: float = 0.0
    trailing_space_ratio: float = 0.0
    label: Optional[int] = Field(None, description="0=Human, 1=AI (for training data)")

    # Ordered feature names used by LightGBM
    FEATURE_NAMES: list[str] = Field(
        default=[
            "empty_line_ratio", "avg_line_length", "max_line_length",
            "tab_vs_space_ratio", "brace_style_consistency",
            "avg_identifier_length", "identifier_length_variance",
            "single_char_var_ratio", "unique_identifier_ratio",
            "keyword_to_identifier_ratio", "avg_cyclomatic_complexity",
            "num_functions", "avg_function_loc",
            "halstead_volume", "halstead_difficulty", "halstead_effort",
            "halstead_bugs", "maintainability_index",
            "code_to_comment_ratio", "max_nesting_depth",
            "total_includes", "has_bits_stdc", "macro_count",
            "modern_cpp_ratio", "const_usage_ratio", "has_fast_io",
            "newline_style_ratio", "shannon_entropy", "bigram_entropy",
            "whitespace_entropy", "comment_ratio", "trailing_space_ratio",
        ],
        exclude=True,  # excluded from serialisation
    )

    def to_feature_vector(self) -> list[float]:
        """Return a flat list of 32 floats in the canonical order."""
        d = self.model_dump()
        return [float(d[f]) for f in self.FEATURE_NAMES]


# =====================================================================
# SILVER-DL — 512 Token IDs for RoBERTa
# =====================================================================
class SilverDLRecord(BaseModel):
    """Tokenised chunk for RoBERTa / GraphCodeBERT."""
    code_hash: str
    input_ids: list[int] = Field(..., min_length=512, max_length=512)
    attention_mask: list[int] = Field(..., min_length=512, max_length=512)
    chunk_index: int = Field(0, ge=0)
    label: Optional[int] = Field(None, description="0=Human, 1=AI")


# =====================================================================
# Chunk-level prediction result
# =====================================================================
class ChunkResult(BaseModel):
    """One chunk's analysis output (part of the full prediction)."""
    index: int
    score: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., description="AI | HUMAN")
    top_ai: list[str] = Field(default_factory=list, description="Top AI-signal tokens")
    top_hu: list[str] = Field(default_factory=list, description="Top Human-signal tokens")
    snippet: str = ""
    html: str = ""
    critique: str = ""


# =====================================================================
# GOLD — Full prediction result
# =====================================================================
class GoldPredictionRecord(BaseModel):
    """Complete prediction output (Gold layer)."""
    code_hash: str
    user_id: Optional[int] = None
    model_used: str = Field(..., description="C++ OOP Model | C++ Normal Model")
    classification: str = Field(..., description="OOP | NORMAL")
    prediction: str = Field(..., description="AI GENERATED | HUMAN WRITTEN")
    confidence: float = Field(..., ge=0.0, le=1.0, description="final_score")
    perplexity: float = Field(0.0, description="Mean PPL from Qwen")
    max_ppl: float = Field(0.0, description="Max PPL (most surprising line)")
    burstiness: float = Field(0.0, description="PPL variance")
    is_ambiguous: bool = False
    retry_count: int = Field(0, ge=0)
    total_tokens: int = 0
    total_chunks: int = 0
    global_critique: str = ""
    global_html: str = ""
    top_ai_signals: list[str] = Field(default_factory=list)
    top_hu_signals: list[str] = Field(default_factory=list)
    chunk_details: list[ChunkResult] = Field(default_factory=list)
    inference_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


# =====================================================================
# API Response envelope
# =====================================================================
class PredictionAPIResponse(BaseModel):
    """Public JSON shape returned to the user / Django frontend."""
    final_pred: str
    final_score: float
    model_used: str
    perplexity: float = 0.0
    max_ppl: float = 0.0
    burstiness: float = 0.0
    is_ambiguous: bool = False
    total_tokens: int = 0
    total_chunks: int = 0
    global_critique: str = ""
    global_html: str = ""
    chunks: list[ChunkResult] = Field(default_factory=list)


# =====================================================================
# Model performance tracking
# =====================================================================
class ModelPerformanceRecord(BaseModel):
    """Gold Model Performance row."""
    model_name: str
    eval_date: datetime
    accuracy: float
    f1_macro: float
    precision_val: float
    recall_val: float
    auc_roc: float
    total_samples: int
    alpha: float = 0.48
    threshold: float = 0.5
    notes: str = ""


# =====================================================================
# Redpanda / Kafka event envelopes
# =====================================================================
class CodeSubmittedEvent(BaseModel):
    """Published to topic `code.submitted`."""
    code_hash: str
    user_id: Optional[int] = None
    source: str = "web_upload"
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class PredictionCompletedEvent(BaseModel):
    """Published to topic `prediction.completed`."""
    code_hash: str
    prediction: str
    confidence: float
    inference_ms: int
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class RetrainTriggerEvent(BaseModel):
    """Published to topic `retrain.trigger`."""
    reason: str = "scheduled"
    sample_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
