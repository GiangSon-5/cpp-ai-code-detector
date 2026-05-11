"""
engine/html_renderer.py — LIG Heatmap HTML rendering.

Ported exactly from Agent.ipynb render_html_view().
Generates a two-column HTML with code highlighting and top signal lists.
"""

from __future__ import annotations

import html
import numpy as np
from collections import defaultdict
from typing import Any

from src.shared.logger import AppLogger


@AppLogger.log_function(module="html_renderer")
def render_html_heatmap(
    tokens: list[str],
    attributions: list[float],
    code_snippet: str = "",
    title: str = "CODE ANALYSIS",
) -> str:
    """Render an HTML heatmap visualization of token attributions.

    Matches the Jupyter Notebook implementation.
    """
    if not tokens or not attributions:
        return "<html><body><p>No attribution data available.</p></body></html>"

    aggregated_blocks = []
    current_tokens = []
    current_score_sum = 0.0
    token_count = 0

    for token, score in zip(tokens, attributions):
        if token in ['<s>', '</s>', '<pad>']:
            continue
            
        is_start_new_word = token.startswith('Ġ') or token.startswith('Ċ')
        if is_start_new_word and current_tokens:
            raw_str = "".join(current_tokens)
            full_word = raw_str.replace("Ġ", " ").replace("Ċ", "\n")
            if full_word:
                avg_score = current_score_sum / token_count if token_count > 0 else 0
                aggregated_blocks.append((full_word, avg_score))
            current_tokens = []
            current_score_sum = 0.0
            token_count = 0

        current_tokens.append(token)
        current_score_sum += score
        token_count += 1

    if current_tokens:
        raw_str = "".join(current_tokens)
        full_word = raw_str.replace("Ġ", " ").replace("Ċ", "\n")
        if full_word:
            aggregated_blocks.append((full_word, current_score_sum / token_count))

    word_stats = defaultdict(list)
    for word, score in aggregated_blocks:
        clean_word = word.strip()
        if clean_word:
            word_stats[clean_word].append(score)

    unique_words_list = [(word, sum(score_list) / len(score_list)) for word, score_list in word_stats.items()]
    pos_words = sorted([(w, s) for w, s in unique_words_list if s > 0], key=lambda x: x[1], reverse=True)
    neg_words = sorted([(w, s) for w, s in unique_words_list if s < 0], key=lambda x: x[1])

    top_k = 50
    ai_list_html = [
        f"<tr><td style='color:#2ecc71; font-weight:bold; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{html.escape(w)}</td><td style='text-align:right; color:#555; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{s:.3f}</td></tr>" 
        for w, s in pos_words[:top_k]
    ]
    human_list_html = [
        f"<tr><td style='color:#e74c3c; font-weight:bold; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{html.escape(w)}</td><td style='text-align:right; color:#555; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{s:.3f}</td></tr>" 
        for w, s in neg_words[:top_k]
    ]

    code_html_parts = []
    all_scores = [abs(s) for w, s in aggregated_blocks]
    max_impact = float(np.max(all_scores)) if len(all_scores) > 0 and float(np.max(all_scores)) > 0 else 1.0

    for word, score in aggregated_blocks:
        safe_word = html.escape(word)
        relative_score = abs(score) / max_impact
        alpha = min(float(np.power(relative_score, 0.35)), 1.0)
        bg_color = "transparent"
        if alpha >= 0.02:
            bg_color = f"rgba(46, 204, 113, {alpha:.2f})" if score > 0 else f"rgba(231, 76, 60, {alpha:.2f})"
        code_html_parts.append(f'<span style="background-color: {bg_color}; border-radius: 2px;">{safe_word}</span>')

    code_html = "".join(code_html_parts)
    
    human_table = "".join(human_list_html) if human_list_html else "<tr><td colspan='2' align='center' style='padding:10px; color:#999'>None</td></tr>"
    ai_table = "".join(ai_list_html) if ai_list_html else "<tr><td colspan='2' align='center' style='padding:10px; color:#999'>None</td></tr>"

    return f"""
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0; padding:0; background-color: #f8f9fa;">
        <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
            <div style="flex: 3; border: 1px solid #ccc; border-radius: 8px; background-color: #fff; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
                <div style="background-color: #2c3e50; padding: 12px 15px; border-bottom: 1px solid #000; font-weight: bold; color: #ecf0f1; font-size: 14px;">📑 {html.escape(title)}</div>
                <div style="padding: 15px; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.6; color: #222; height: auto; overflow-y: visible; white-space: pre-wrap;">{code_html}</div>
            </div>
            <div style="flex: 1.2; display: flex; flex-direction: column; gap: 15px; max-height: 800px; position: sticky; top: 0;">
                 <div style="border: 1px solid #f5c6cb; border-radius: 8px; background-color: #fff; overflow: hidden; display: flex; flex-direction: column; flex: 1;">
                    <div style="background-color: #f8d7da; padding: 10px; text-align: center; color: #721c24; font-weight: bold; font-size: 13px; border-bottom: 1px solid #f5c6cb;">👤 Human Signals (Negative)</div>
                    <div style="padding: 0; overflow-y: auto; flex: 1; max-height: 350px;">
                        <table style="width:100%; font-size:12px; border-collapse: collapse; margin: 0;"><tbody>{human_table}</tbody></table>
                    </div>
                 </div>
                 <div style="border: 1px solid #c3e6cb; border-radius: 8px; background-color: #fff; overflow: hidden; display: flex; flex-direction: column; flex: 1;">
                    <div style="background-color: #d4edda; padding: 10px; text-align: center; color: #155724; font-weight: bold; font-size: 13px; border-bottom: 1px solid #c3e6cb;">🤖 AI Signals (Positive)</div>
                    <div style="padding: 0; overflow-y: auto; flex: 1; max-height: 350px;">
                        <table style="width:100%; font-size:12px; border-collapse: collapse; margin: 0;"><tbody>{ai_table}</tbody></table>
                    </div>
                 </div>
            </div>
        </div>
    </body>
    </html>
    """
