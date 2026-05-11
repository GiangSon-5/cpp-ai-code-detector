import json
import httpx
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import html
import numpy as np
from collections import defaultdict
import google.generativeai as genai
import os

from .remote_client import client
from .engine import ModelManager, run_roberta_engine
from .hybrid_evaluator import HybridEvaluator

# Khởi tạo mô hình chạy Local GPU
model_manager = ModelManager()
lgbm_evaluator = HybridEvaluator()

class AgentState(TypedDict):
    code_input: str
    classification: Optional[str]
    is_ambiguous: bool
    retry_count: int
    final_output: Optional[Dict[str, Any]]
def roberta_decode(tokens: list) -> str:
    """Giả lập tokenizer.convert_tokens_to_string cho RoBERTa"""
    # RoBERTa dùng 'Ġ' thay cho space và 'Ċ' thay cho newline
    return "".join(tokens).replace('Ġ', ' ').replace('Ċ', '\n').strip()

# --- EXPERT EXPLAINER (Moved from Colab) ---
class ExpertExplainer:
    BLACKLIST = {'', '<s>', '</s>', '\n', ' ', '(', ')', '{', '}', ';', ',', '.'}

    def analyze(self, tokens, scores, top_k=50):
        clean_tokens = [t.replace('Ġ', '').replace('Ċ', '').strip() for t in tokens]
        top_ai, top_hu = [], []
        indices = np.argsort(np.abs(scores))[-top_k:][::-1]
        seen_ai, seen_hu = set(), set()

        for idx in indices:
            raw = clean_tokens[idx]
            s = scores[idx]
            if raw not in self.BLACKLIST and len(raw) > 1:
                if s > 0 and raw not in seen_ai and len(top_ai) < 10:
                    top_ai.append(f"'{raw}'")
                    seen_ai.add(raw)
                elif s < 0 and raw not in seen_hu and len(top_hu) < 10:
                    top_hu.append(f"'{raw}'")
                    seen_hu.add(raw)
        return top_ai, top_hu

# --- HTML HEATMAP RENDERER (Moved from Colab) ---
def render_html_view(tokens, scores, top_k=50, title="CODE ANALYSIS"):
    aggregated_blocks = []
    current_tokens = []
    current_score_sum = 0.0
    token_count = 0

    for token, score in zip(tokens, scores):
        if token in ['<s>', '</s>', '<pad>']: continue
        
        is_start_new_word = token.startswith('Ġ') or token.startswith('Ċ')
        if is_start_new_word and current_tokens:
            full_word = roberta_decode(current_tokens)
            if full_word:
                avg_score = current_score_sum / token_count if token_count > 0 else 0
                aggregated_blocks.append((full_word, avg_score))
            current_tokens, current_score_sum, token_count = [], 0.0, 0

        current_tokens.append(token)
        current_score_sum += score
        token_count += 1

    if current_tokens:
        full_word = roberta_decode(current_tokens)
        if full_word:
            aggregated_blocks.append((full_word, current_score_sum / token_count))

    word_stats = defaultdict(list)
    for word, score in aggregated_blocks:
        clean_word = word.strip()
        if clean_word: word_stats[clean_word].append(score)

    unique_words_list = [(word, sum(sl)/len(sl)) for word, sl in word_stats.items()]
    pos_words = sorted([(w,s) for w,s in unique_words_list if s > 0], key=lambda x: x[1], reverse=True)
    neg_words = sorted([(w,s) for w,s in unique_words_list if s < 0], key=lambda x: x[1])

    ai_list_html = [f"<tr><td style='color:#2ecc71; font-weight:bold;'>{html.escape(w)}</td><td style='text-align:right;'>{s:.3f}</td></tr>" for w, s in pos_words[:top_k]]
    human_list_html = [f"<tr><td style='color:#e74c3c; font-weight:bold;'>{html.escape(w)}</td><td style='text-align:right;'>{s:.3f}</td></tr>" for w, s in neg_words[:top_k]]

    all_scores = [abs(s) for _, s in aggregated_blocks]
    max_impact = np.max(all_scores) if all_scores and np.max(all_scores) > 0 else 1.0

    code_html_parts = []
    for word, score in aggregated_blocks:
        safe_word = html.escape(word)
        rel = abs(score) / max_impact
        alpha = min(np.power(rel, 0.35), 1.0)
        bg = "transparent"
        if alpha >= 0.02:
            bg = f"rgba(46, 204, 113, {alpha:.2f})" if score > 0 else f"rgba(231, 76, 60, {alpha:.2f})"
        code_html_parts.append(f'<span style="background-color: {bg}; border-radius: 2px;">{safe_word}</span>')

    code_html = "".join(code_html_parts)
    none_row = "<tr><td colspan='2' align='center' style='color:#999'>None</td></tr>"

    return f"""
    <html><body style="background-color: #f8f9fa; font-family: sans-serif;">
        <div style="display: flex; gap: 20px;">
            <div style="flex: 3; border: 1px solid #ccc; background:#fff; padding:15px; border-radius:8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="background:#2c3e50; color:#fff; padding:8px; font-weight:bold;">📑 {title}</div>
                <div style="white-space: pre-wrap; font-family: monospace; margin-top:10px;">{code_html}</div>
            </div>
            <div style="flex: 1; display: flex; flex-direction: column; gap: 10px;">
                <div style="border: 1px solid #f5c6cb; background:#fff; border-radius:8px;">
                    <div style="background:#f8d7da; color:#721c24; padding:5px; text-align:center; font-weight:bold;">Human Signals</div>
                    <table style="width:100%; font-size:12px;">{"".join(human_list_html) if human_list_html else none_row}</table>
                </div>
                <div style="border: 1px solid #c3e6cb; background:#fff; border-radius:8px;">
                    <div style="background:#d4edda; color:#155724; padding:5px; text-align:center; font-weight:bold;">AI Signals</div>
                    <table style="width:100%; font-size:12px;">{"".join(ai_list_html) if ai_list_html else none_row}</table>
                </div>
            </div>
        </div>
    </body></html>
    """

# --- ROUTER NODE ---
async def router_node(state: AgentState):
    code_text = state["code_input"]
    print("\n🤖 [LOCAL AGENT] Router Node: Analyzing Code Structure...")
    
    prompt = f"Phân tích nhanh mã nguồn C++ sau xem nó thuộc loại 'OOP' (có class, inheritance, polymorphism) hay 'NORMAL' (code thủ tục cơ bản, hàm main, for, while).\n\nChỉ trả về 1 chữ duy nhất: OOP hoặc NORMAL.\n\nCode:\n{code_text[:1000]}"
    payload = {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5, "temperature": 0.0
    }
    
    decision = "NORMAL"
    proxy_url = f"{client.NGROK_URL}/api/proxy/vllm"
    async with httpx.AsyncClient(timeout=15.0) as http:
        try:
            res = await http.post(proxy_url, json={"payload": payload}, headers=client.headers)
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"].strip().upper()
            if "OOP" in text: decision = "OOP"
        except Exception as e:
            print(f"⚠️ Router Error: {e}, default to NORMAL")

    print(f"🎯 Router Decision: {decision}")
    return {"classification": decision, "retry_count": 0}

# --- ANALYZER NODE ---
async def analyzer_node(state: AgentState):
    decision = state["classification"]
    print(f"⚙️ [LOCAL GPU AGENT] Analyzer Node: Running Models Locally...")

    # Đảm bảo model đã được tải lên GPU RAM
    if not model_manager.oop_models:
        model_manager.load_resources()

    # 1. Chạy RoBERTa (Local GPU)
    print(f"   ▶️  Running RoBERTa Ensemble 10-fold locally...")
    roberta_res = run_roberta_engine(state["code_input"], decision, model_manager)
    bert_score = roberta_res.get("final_score", 0.5)
    
    # 2. Chạy LightGBM (Local CPU)
    print(f"   ▶️  Running LightGBM locally...")
    lgbm_score = lgbm_evaluator.evaluate(state["code_input"], bert_score)
        
    # 3. Hybrid Fusion
    FUSION_ALPHA = 0.48
    fused_score = FUSION_ALPHA * bert_score + (1.0 - FUSION_ALPHA) * lgbm_score
    
    # --- RENDER HTML LOCALLY ---
    print(f"   🎨 Rendering XAI Heatmap locally...")
    explainer = ExpertExplainer()
    
    # Vẽ Global HTML
    global_tokens = roberta_res.get("tokens", [])
    global_attrs = roberta_res.get("attrs", [])
    final_pred_label = "AI" if fused_score > 0.5 else "HUMAN"
    
    global_html = render_html_view(
        global_tokens, global_attrs, 
        title=f"GLOBAL ANALYSIS ({final_pred_label})"
    )
    g_ai, g_hu = explainer.analyze(global_tokens, global_attrs)

    # Vẽ Chunk HTML
    chunks_data = roberta_res.get("chunks", [])
    for chunk in chunks_data:
        chunk["html"] = render_html_view(
            chunk["tokens"], chunk["attrs"],
            title=f"CHUNK {chunk['index']} ({chunk['score']:.4f})"
        )
        c_ai, c_hu = explainer.analyze(chunk["tokens"], chunk["attrs"])
        chunk["top_ai"] = c_ai
        chunk["top_hu"] = c_hu

    result = {
        "model_used": "C++ OOP Model" if decision == "OOP" else "C++ Normal Model",
        "classification": decision,
        "final_score": fused_score,
        "confidence": fused_score, # Gold schema field
        "bert_score": bert_score,
        "lgbm_score": lgbm_score,
        "total_tokens": roberta_res.get("total_tokens", 0),
        "total_chunks": roberta_res.get("total_chunks", 0),
        "top_ai_signals": g_ai,
        "top_hu_signals": g_hu,
        "global_html": global_html,
        "chunks": chunks_data,
        "retry_count": state["retry_count"]
    }
    
    # Chuẩn hóa nhãn final_pred theo schema
    result["final_pred"] = "AI GENERATED" if fused_score > 0.5 else "HUMAN WRITTEN"
        
    print(f"   📊  Scores -> BERT: {bert_score:.4f} | LGBM: {lgbm_score:.4f} | FUSED: {fused_score:.4f}")

    # 4. Perplexity
    print(f"   ▶️  Calling Perplexity API...")
    ppl_data = await client.get_perplexity(state["code_input"])
    
    if isinstance(ppl_data, dict):
        result["perplexity"] = ppl_data.get("mean_ppl", 0.0)
        result["max_ppl"] = ppl_data.get("max_ppl", 0.0)
        result["burstiness"] = ppl_data.get("burstiness", 0.0)
        print(f"   📊  Perplexity: {result['perplexity']:.2f} | Burstiness: {result['burstiness']:.2f} | Max PPL: {result['max_ppl']:.2f}")
    else:
        result["perplexity"] = float(ppl_data)
        result["max_ppl"] = 0.0
        result["burstiness"] = 0.0
        print(f"   📊  Perplexity Score: {result['perplexity']:.2f}")

    return {"final_output": result}

# --- JUDGE NODE ---
async def judge_node(state: AgentState):
    result = state["final_output"]
    score = result["final_score"]
    ppl = result["perplexity"]
    
    print(f"⚖️ [LOCAL AGENT] Judge Node: Analyzing Confidence...")
    
    is_ambiguous = (0.45 <= score <= 0.55) or (score > 0.6 and ppl > 3.0) or (score < 0.4 and ppl < 1.0)
        
    if is_ambiguous and state["retry_count"] < 1:
        print("⚠️ Phát hiện nhập nhằng. Sẽ thử lại (Retry)!")
        return {"is_ambiguous": True, "retry_count": state["retry_count"] + 1}
        
    print("✅ Result Accepted. Proceeding to Critique.")
    return {"is_ambiguous": False}

def should_retry(state: AgentState):
    return "analyzer" if state["is_ambiguous"] and state["retry_count"] <= 1 else "critique"

# --- CRITIQUE NODE ---
async def critique_node(state: AgentState):
    print("📝 [LOCAL GPU AGENT] Critique Node: Calling Gemini API...")
    result = state["final_output"]
    critique = await client.get_critique(state["code_input"], result["final_score"])
    result["global_critique"] = critique
    result["is_ambiguous"] = state.get("is_ambiguous", False)
    print("✅ [DONE] Global Critique generated.")
    return {"final_output": result}

# Build Local Graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("judge", judge_node)
workflow.add_node("critique", critique_node)

workflow.add_edge(START, "router")
workflow.add_edge("router", "analyzer")
workflow.add_edge("analyzer", "judge")
workflow.add_conditional_edges("judge", should_retry)
workflow.add_edge("critique", END)

local_agent_app = workflow.compile()
