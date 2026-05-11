"""
engine/explainer.py — LIG (Layer Integrated Gradients) and token attribution.

Ported from extract_1.py ExpertExplainer.
Provides per-token importance scores for AI/Human classification.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch

from src.shared.logger import AppLogger

logger = AppLogger()


class ExpertExplainer:
    """Computes Layer Integrated Gradients for RoBERTa models.

    Falls back to attention-weight-based attribution if LIG fails (e.g. GPU OOM).
    """

    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @AppLogger.log_function(module="explainer")
    def compute_attributions(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        n_steps: int = 50,
    ) -> dict[str, Any]:
        """Compute token-level attributions using LIG.

        Args:
            input_ids: shape (1, seq_len)
            attention_mask: shape (1, seq_len)
            n_steps: number of interpolation steps for IG

        Returns:
            dict with:
                - attributions: list[float] (per-token importance)
                - top_ai_tokens: list[str] (tokens contributing to AI label)
                - top_human_tokens: list[str] (tokens contributing to Human label)
                - method: "lig" or "attention_fallback"
        """
        try:
            return self._compute_lig(input_ids, attention_mask, n_steps)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            logger.warning(
                module="explainer",
                function="compute_attributions",
                message=f"LIG failed (likely OOM), using attention fallback: {exc}",
            )
            return self._compute_attention_fallback(input_ids, attention_mask)

    def _compute_lig(
        self, input_ids: np.ndarray, attention_mask: np.ndarray, n_steps: int
    ) -> dict[str, Any]:
        """Layer Integrated Gradients implementation."""
        ids_tensor = torch.tensor(input_ids, dtype=torch.long).to(self._device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(self._device)

        # Get embeddings layer
        embeddings = self._model.get_input_embeddings()
        baseline_embed = torch.zeros_like(embeddings(ids_tensor))
        input_embed = embeddings(ids_tensor)

        # Interpolation
        all_grads = []
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interp = baseline_embed + alpha * (input_embed - baseline_embed)
            interp.requires_grad_(True)

            # Forward pass with embeddings
            outputs = self._model(
                inputs_embeds=interp,
                attention_mask=mask_tensor,
            )
            logits = outputs.logits
            target = logits[0, 1]  # Score for AI class
            target.backward(retain_graph=False)

            if interp.grad is not None:
                all_grads.append(interp.grad.detach().cpu().numpy())
            interp.requires_grad_(False)

        if not all_grads:
            return self._compute_attention_fallback(input_ids, attention_mask)

        # Average gradients × (input - baseline) → attribution per embedding dim → sum
        avg_grads = np.mean(all_grads, axis=0)
        diff = (input_embed - baseline_embed).detach().cpu().numpy()
        attributions_per_dim = avg_grads * diff
        attributions = attributions_per_dim.sum(axis=-1)[0]  # shape: (seq_len,)

        return self._format_attributions(attributions, input_ids, attention_mask, method="lig")

    def _compute_attention_fallback(
        self, input_ids: np.ndarray, attention_mask: np.ndarray
    ) -> dict[str, Any]:
        """Fallback: use last-layer attention weights as proxy attributions."""
        ids_tensor = torch.tensor(input_ids, dtype=torch.long).to(self._device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long).to(self._device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=ids_tensor,
                attention_mask=mask_tensor,
                output_attentions=True,
            )

        # Use last layer, average across heads, take CLS token row
        last_attn = outputs.attentions[-1]  # (1, heads, seq, seq)
        avg_attn = last_attn.mean(dim=1)    # (1, seq, seq)
        cls_attn = avg_attn[0, 0, :].cpu().numpy()  # (seq,)

        return self._format_attributions(cls_attn, input_ids, attention_mask, method="attention_fallback")

    def _format_attributions(
        self,
        attributions: np.ndarray,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        method: str,
    ) -> dict[str, Any]:
        """Convert raw attributions into structured output with top tokens."""
        seq_len = int(attention_mask[0].sum())
        attr_slice = attributions[:seq_len]
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])

        # Filter special tokens
        valid_indices = [
            i for i, t in enumerate(tokens)
            if t not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>")
        ]
        valid_attrs = [(tokens[i], float(attr_slice[i])) for i in valid_indices]

        # Sort by attribution magnitude
        sorted_by_value = sorted(valid_attrs, key=lambda x: x[1], reverse=True)

        top_ai = [t for t, v in sorted_by_value[:10] if v > 0]
        top_human = [t for t, v in sorted(valid_attrs, key=lambda x: x[1])[:10] if v < 0]

        return {
            "attributions": [float(a) for a in attr_slice],
            "tokens": tokens,
            "top_ai_tokens": top_ai[:5],
            "top_human_tokens": top_human[:5],
            "method": method,
        }
