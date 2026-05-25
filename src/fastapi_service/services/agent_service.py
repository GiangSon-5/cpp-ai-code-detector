"""
services/agent_service.py — LangGraph Agent Workflow.

Implements the 4-node agentic pipeline:
    ① Router  — classify OOP/NORMAL via LLM + heuristic fallback
    ② Analyzer — RoBERTa Ensemble + LIG + Perplexity
    ③ Judge   — self-correction if ambiguous (score 0.40-0.60 or PPL conflict)
    ④ Critique — Map-Reduce LLM analysis per chunk → global summary

This is the main entry point called by the /api/analyze_stream router.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any, AsyncGenerator, Optional

from src.fastapi_service.engine.heuristic_classifier import heuristic_classify
from src.fastapi_service.engine.llm_handler import LLMHandler
from src.fastapi_service.engine.model_manager import ModelManager
from src.fastapi_service.engine.roberta_engine import RoBERTaEngine
from src.fastapi_service.engine.fingerprint_engine import FingerprintEngine
from src.fastapi_service.schemas.prediction_schema import (
    AnalyzeResponse,
    ChunkResult,
    SSEProgressEvent,
)
from src.shared.config import settings
from src.shared.data_contracts import GoldPredictionRecord
from src.shared.hashing import compute_code_hash
from src.shared.logger import AppLogger

logger = AppLogger()


class AgentService:
    """Orchestrates the full AI detection pipeline as a LangGraph-style workflow."""

    def __init__(self) -> None:
        self._model_manager = ModelManager()
        self._roberta_engine = RoBERTaEngine(self._model_manager)
        self._llm_handler = LLMHandler()
        self._fingerprint_engine = FingerprintEngine()

    @AppLogger.log_function(module="agent_service")
    def ensure_models_loaded(self) -> dict[str, Any]:
        """Load models if not already loaded. Called on server startup."""
        if not self._model_manager.models_loaded:
            return self._model_manager.load_models(model_type="both")
        return {"status": "already_loaded"}

    @AppLogger.log_function(module="agent_service")
    async def analyze_code(
        self,
        raw_code: str,
        user_id: Optional[int] = None,
    ) -> tuple[AnalyzeResponse, GoldPredictionRecord]:
        """Run the full 4-node analysis pipeline.

        Args:
            raw_code: decoded C++ source code
            user_id: optional user ID for tracking

        Returns:
            (AnalyzeResponse for API, GoldPredictionRecord for DB)
        """
        t0 = time.perf_counter()
        code_hash = compute_code_hash(raw_code)

        # ─── NODE 1: ROUTER ───────────────────────────────────────
        router_result = await self._node_router(raw_code)
        classification = router_result["classification"]
        model_type = classification  # "OOP" or "NORMAL"
        model_used = f"C++ {classification} Model"

        # ─── NODE 2: ANALYZER ─────────────────────────────────────
        analyzer_result = self._node_analyzer(raw_code, model_type)
        mean_score = analyzer_result["mean_score"]
        chunks = analyzer_result["chunks"]
        total_tokens = analyzer_result["total_tokens"]
        total_chunks = analyzer_result["total_chunks"]
        global_html = analyzer_result["global_html"]

        # Compute perplexity in parallel
        ppl_result = await self._llm_handler.compute_perplexity(raw_code)
        perplexity = ppl_result["perplexity"]
        max_ppl = ppl_result["max_ppl"]
        burstiness = ppl_result["burstiness"]

        # ─── NODE 5: FINGERPRINT XAI (LightGBM + SHAP) (Moved Up) ──
        fingerprint_result = self._fingerprint_engine.analyze(raw_code)
        ml_score = fingerprint_result.lgbm_score if fingerprint_result else 0.5

        # ─── NODE 3: JUDGE ────────────────────────────────────────
        judge_result = await self._node_judge(
            mean_score=mean_score,
            perplexity=perplexity,
            raw_code=raw_code,
            current_model_type=model_type,
            chunks=chunks,
            ml_score=ml_score,
        )
        # Judge may have updated values
        mean_score = judge_result["final_score"]
        is_ambiguous = judge_result["is_ambiguous"]
        retry_count = judge_result["retry_count"]
        chunks = judge_result["chunks"]
        ml_dl_conflict = judge_result.get("ml_dl_conflict", False)
        ml_dl_gap = judge_result.get("ml_dl_gap", 0.0)
        fusion_applied = judge_result.get("fusion_applied", False)
        if judge_result.get("model_switched"):
            model_type = judge_result["new_model_type"]
            model_used = f"C++ {model_type} Model"

        prediction = "AI GENERATED" if mean_score >= settings.DEFAULT_THRESHOLD else "HUMAN WRITTEN"

        # ─── NODE 4: CRITIQUE ─────────────────────────────────────
        chunks = await self._node_critique(chunks, mean_score)

        # Build global critique
        chunk_critiques = [c.critique for c in chunks if c.critique]
        global_critique = await self._llm_handler.generate_global_critique(
            chunk_critiques, mean_score
        )

        # Aggregate top signals
        top_ai_signals = []
        top_hu_signals = []
        for c in chunks:
            top_ai_signals.extend(c.top_ai)
            top_hu_signals.extend(c.top_hu)
        # Deduplicate, keep top 5
        top_ai_signals = list(dict.fromkeys(top_ai_signals))[:5]
        top_hu_signals = list(dict.fromkeys(top_hu_signals))[:5]

        inference_ms = int((time.perf_counter() - t0) * 1000)

        # ─── Compute individual model scores ─────────────────────
        dl_score_raw = round(analyzer_result["mean_score"], 4)  # RoBERTa before judge
        ml_score_raw = round(ml_score, 4)
        hybrid_score_raw = round(0.6 * dl_score_raw + 0.4 * ml_score_raw, 4)

        # ─── BUILD RESPONSE ───────────────────────────────────────
        response = AnalyzeResponse(
            final_pred=prediction,
            final_score=round(mean_score, 4),
            model_used=model_used,
            dl_score=dl_score_raw,
            ml_score=ml_score_raw,
            hybrid_score=hybrid_score_raw,
            perplexity=round(perplexity, 2),
            max_ppl=round(max_ppl, 2),
            burstiness=round(burstiness, 2),
            is_ambiguous=is_ambiguous,
            ml_dl_conflict=ml_dl_conflict,
            ml_dl_gap=ml_dl_gap,
            fusion_applied=fusion_applied,
            total_tokens=total_tokens,
            total_chunks=total_chunks,
            global_critique=global_critique,
            global_html=global_html,
            chunks=chunks,
            fingerprint=fingerprint_result,
        )

        # Build Gold record for DB
        gold_record = GoldPredictionRecord(
            code_hash=code_hash,
            user_id=user_id,
            model_used=model_used,
            classification=classification,
            prediction=prediction,
            confidence=round(mean_score, 4),
            perplexity=round(perplexity, 2),
            max_ppl=round(max_ppl, 2),
            burstiness=round(burstiness, 2),
            is_ambiguous=is_ambiguous,
            retry_count=retry_count,
            ml_dl_conflict=ml_dl_conflict,
            ml_dl_gap=ml_dl_gap,
            fusion_applied=fusion_applied,
            total_tokens=total_tokens,
            total_chunks=total_chunks,
            global_critique=global_critique,
            top_ai_signals=top_ai_signals,
            top_hu_signals=top_hu_signals,
            chunk_details=chunks,
            inference_ms=inference_ms,
        )

        return response, gold_record

    async def analyze_code_stream(
        self,
        raw_code: str,
        user_id: Optional[int] = None,
    ) -> AsyncGenerator[SSEProgressEvent, None]:
        """Generator version that yields SSE progress events during analysis.

        Progress mapping:
            5%  — Router started
            20% — Router complete
            65% — Analyzer complete
            75% — Judge complete
            95% — Critique complete
            100% — Done (final result)
        """
        t0 = time.perf_counter()
        code_hash = compute_code_hash(raw_code)

        # 5% — Router
        yield SSEProgressEvent(step="router", progress=5, message="Classifying code type...")

        router_result = await self._node_router(raw_code)
        classification = router_result["classification"]
        model_type = classification
        model_used = f"C++ {classification} Model"

        yield SSEProgressEvent(step="router", progress=20, message=f"Code classified as {classification}")

        # 20-65% — Analyzer
        yield SSEProgressEvent(step="analyzer", progress=25, message="Running RoBERTa Ensemble...")

        analyzer_result = self._node_analyzer(raw_code, model_type)
        mean_score = analyzer_result["mean_score"]
        chunks = analyzer_result["chunks"]

        ppl_result = await self._llm_handler.compute_perplexity(raw_code)
        perplexity = ppl_result["perplexity"]
        max_ppl = ppl_result["max_ppl"]
        burstiness = ppl_result["burstiness"]

        yield SSEProgressEvent(
            step="analyzer", progress=65,
            message=f"Ensemble complete — score {mean_score:.4f}, PPL {perplexity:.2f}"
        )

        # 65-75% — Judge
        yield SSEProgressEvent(step="judge", progress=67, message="Running XAI fingerprint...")
        fingerprint_result = self._fingerprint_engine.analyze(raw_code)
        ml_score = fingerprint_result.lgbm_score if fingerprint_result else 0.5

        yield SSEProgressEvent(step="judge", progress=69, message="Evaluating confidence...")

        judge_result = await self._node_judge(
            mean_score=mean_score,
            perplexity=perplexity,
            raw_code=raw_code,
            current_model_type=model_type,
            chunks=chunks,
            ml_score=ml_score,
        )
        mean_score = judge_result["final_score"]
        is_ambiguous = judge_result["is_ambiguous"]
        retry_count = judge_result["retry_count"]
        chunks = judge_result["chunks"]
        ml_dl_conflict = judge_result.get("ml_dl_conflict", False)
        ml_dl_gap = judge_result.get("ml_dl_gap", 0.0)
        fusion_applied = judge_result.get("fusion_applied", False)
        if judge_result.get("model_switched"):
            model_type = judge_result["new_model_type"]
            model_used = f"C++ {model_type} Model"

        prediction = "AI GENERATED" if mean_score >= settings.DEFAULT_THRESHOLD else "HUMAN WRITTEN"

        yield SSEProgressEvent(step="judge", progress=75, message=f"Judge: {'ambiguous → self-corrected' if is_ambiguous else 'confident'}")

        # 75-95% — Critique
        yield SSEProgressEvent(step="critique", progress=78, message="Generating LLM analysis...")

        chunks = await self._node_critique(chunks, mean_score)
        chunk_critiques = [c.critique for c in chunks if c.critique]
        global_critique = await self._llm_handler.generate_global_critique(chunk_critiques, mean_score)

        top_ai_signals = list(dict.fromkeys(
            t for c in chunks for t in c.top_ai
        ))[:5]
        top_hu_signals = list(dict.fromkeys(
            t for c in chunks for t in c.top_hu
        ))[:5]

        yield SSEProgressEvent(step="critique", progress=90, message="Finalizing results...")

        # 100% — Final result
        inference_ms = int((time.perf_counter() - t0) * 1000)

        # Compute individual model scores
        dl_score_raw = round(analyzer_result["mean_score"], 4)
        ml_score_raw = round(ml_score, 4)
        hybrid_score_raw = round(0.6 * dl_score_raw + 0.4 * ml_score_raw, 4)

        final = AnalyzeResponse(
            final_pred=prediction,
            final_score=round(mean_score, 4),
            model_used=model_used,
            dl_score=dl_score_raw,
            ml_score=ml_score_raw,
            hybrid_score=hybrid_score_raw,
            perplexity=round(perplexity, 2),
            max_ppl=round(max_ppl, 2),
            burstiness=round(burstiness, 2),
            is_ambiguous=is_ambiguous,
            ml_dl_conflict=ml_dl_conflict,
            ml_dl_gap=ml_dl_gap,
            fusion_applied=fusion_applied,
            total_tokens=analyzer_result["total_tokens"],
            total_chunks=analyzer_result["total_chunks"],
            global_critique=global_critique,
            global_html=analyzer_result["global_html"],
            chunks=chunks,
            fingerprint=fingerprint_result,
        )

        yield SSEProgressEvent(
            step="complete", progress=100,
            message=f"Done in {inference_ms}ms",
            data=final.model_dump(),
        )

    # ──────────────────────────────────────────────────────────────
    # Pipeline Nodes
    # ──────────────────────────────────────────────────────────────

    @AppLogger.log_function(module="agent_service")
    async def _node_router(self, code: str) -> dict[str, Any]:
        """Node ①: Classify code as OOP or NORMAL.

        Strategy: LLM classification first, heuristic fallback for validation.
        """
        # LLM classification
        llm_result = await self._llm_handler.classify_code(code)
        llm_class = llm_result["classification"]

        # Heuristic validation
        heuristic_result = heuristic_classify(code)
        heur_class = heuristic_result.classification
        heur_score = heuristic_result.score

        # If LLM and heuristic agree → use it
        if llm_class == heur_class:
            final_class = llm_class
        elif llm_result["provider"] == "fallback":
            # LLM unavailable → trust heuristic
            final_class = heur_class
        else:
            # Disagreement → trust heuristic if it's confident (score > 0.4)
            if heur_score > 0.4:
                final_class = heur_class
            else:
                final_class = llm_class

        return {
            "classification": final_class,
            "llm_result": llm_result,
            "heuristic_result": {
                "classification": heur_class,
                "score": heur_score,
                "reasons": heuristic_result.reasons,
            },
        }

    @AppLogger.log_function(module="agent_service")
    def _node_analyzer(self, code: str, model_type: str) -> dict[str, Any]:
        """Node ②: Run RoBERTa Ensemble + LIG + chunking."""
        enable_lig = self._model_manager.gpu_available
        return self._roberta_engine.analyze(code, model_type=model_type, enable_lig=enable_lig)

    @AppLogger.log_function(module="agent_service")
    async def _node_judge(
        self,
        mean_score: float,
        perplexity: float,
        raw_code: str,
        current_model_type: str,
        chunks: list[ChunkResult],
        ml_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """Node ③: Self-correction logic.

        Business Rules:
            - Ambiguous zone: 0.40 ≤ score ≤ 0.60 → try opposite model
            - PPL conflict: score > 0.60 AND PPL > 5.0 → suspicious, retry
            - Maximum 1 retry
        """
        is_ambiguous = False
        retry_count = 0
        model_switched = False
        new_model_type = current_model_type
        final_score = mean_score

        # Check ambiguous zone
        in_ambiguous_zone = settings.AMBIGUOUS_LOW <= mean_score <= settings.AMBIGUOUS_HIGH

        # Check PPL conflict (high AI score but high PPL suggests human)
        ppl_conflict = mean_score > 0.60 and perplexity > 5.0

        if in_ambiguous_zone or ppl_conflict:
            is_ambiguous = True
            retry_count = 1

            # Try opposite model
            opposite = "OOP" if current_model_type == "NORMAL" else "NORMAL"

            logger.info(
                module="agent_service",
                function="_node_judge",
                message=f"Self-correction triggered: "
                        f"score={mean_score:.4f}, PPL={perplexity:.2f}, "
                        f"switching {current_model_type}→{opposite}",
            )

            try:
                retry_result = self._roberta_engine.analyze(
                    raw_code, model_type=opposite, enable_lig=False,
                )
                retry_score = retry_result["mean_score"]

                # Use retry result if it's MORE confident (further from 0.5)
                if abs(retry_score - 0.5) > abs(mean_score - 0.5):
                    final_score = retry_score
                    chunks = retry_result["chunks"]
                    new_model_type = opposite
                    model_switched = True
                    logger.info(
                        module="agent_service",
                        function="_node_judge",
                        message=f"Self-correction accepted: {mean_score:.4f} → {retry_score:.4f}",
                    )
                else:
                    logger.info(
                        module="agent_service",
                        function="_node_judge",
                        message=f"Self-correction rejected: original {mean_score:.4f} more confident than retry {retry_score:.4f}",
                    )
            except Exception as exc:
                logger.error(
                    module="agent_service",
                    function="_node_judge",
                    error=f"Self-correction failed: {exc}",
                )

        # --- THÊM MỚI: Kiểm tra xung đột DL-ML ---
        ml_dl_conflict = False
        ml_dl_gap = 0.0
        fusion_applied = False
        if ml_score is not None:
            gap = abs(final_score - ml_score)
            ml_dl_gap = round(gap, 3)

            # Mâu thuẫn khi nhãn đối lập và độ lệch >= 40%
            dl_label_ai = final_score >= 0.50
            ml_label_ai = ml_score >= 0.50
            opposite_labels = dl_label_ai != ml_label_ai

            ml_dl_conflict = opposite_labels and gap >= 0.40

            # Chỉ can thiệp khi DL không quá tự tin (tránh ML kéo sai)
            if ml_dl_conflict and 0.45 <= final_score <= 0.75:
                final_score = 0.70 * final_score + 0.30 * ml_score
                fusion_applied = True
                logger.info(
                    module="agent_service",
                    function="_node_judge",
                    message=f"Controlled Adaptive Fusion applied (Refined): "
                            f"DL={mean_score:.4f}, ML={ml_score:.4f}, Fusion Score={final_score:.4f}",
                )

        return {
            "final_score": final_score,
            "is_ambiguous": is_ambiguous,
            "retry_count": retry_count,
            "model_switched": model_switched,
            "new_model_type": new_model_type,
            "chunks": chunks,
            "ml_dl_conflict": ml_dl_conflict,
            "ml_dl_gap": ml_dl_gap,
            "fusion_applied": fusion_applied,
        }

    @AppLogger.log_function(module="agent_service")
    async def _node_critique(
        self, chunks: list[ChunkResult], overall_score: float
    ) -> list[ChunkResult]:
        """Node ④: Generate LLM critique for each chunk (Map phase)."""
        updated: list[ChunkResult] = []

        for chunk in chunks:
            try:
                critique = await self._llm_handler.generate_critique(
                    chunk.snippet, chunk.score, chunk.label,
                )
                updated.append(chunk.model_copy(update={"critique": critique}))
            except Exception as exc:
                logger.error(
                    module="agent_service",
                    function="_node_critique",
                    error=f"Critique failed for chunk {chunk.index}: {exc}",
                )
                updated.append(chunk)

        return updated
