# ===== Cell 1 (code) =====
%%capture
!pip install -q torch
!pip install -q transformers
!pip install -q scikit-learn
!pip install -q matplotlib
!pip install -q lime
!pip install -q captum



# ===== Cell 2 (code) =====
!pip install --upgrade numpy


# ===== Cell 3 (code) =====
!pip install fastapi uvicorn pyngrok nest_asyncio google-genai python-multipart bitsandbytes accelerate --quiet

# ===== Cell 4 (code) =====
# Cài đặt các thư viện cần thiết cho Agent và Model
!pip install -q langgraph langchain langchain-core langsmith


# ===== Cell 5 (code) =====
!pip install langchain-google-genai langchain-openai langchain_anthropic --quiet

# ===== Cell 6 (code) =====
os.system("killall ngrok")
os.system("fuser -k 8000/tcp")

# ===== Cell 7 (code) =====
# ==================================================================================
# 1. IMPORT VÀ CẤU HÌNH MÔI TRƯỜNG
# ==================================================================================
import os
import sys
import json
import torch
import numpy as np
import html
import base64
import time
import threading
import uvicorn
import nest_asyncio
import re
import warnings
import hashlib
import gc
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager

# Tắt cảnh báo để output sạch đẹp
warnings.filterwarnings("ignore")

from typing import List, Dict, Any, TypedDict, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pyngrok import ngrok
from google.colab import drive, userdata
from tqdm.auto import tqdm

# --- IMPORT LANGCHAIN CORE & AI PROVIDERS ---
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- IMPORT MODEL LOCAL & CAPTUM ---
from transformers import RobertaTokenizer, RobertaForSequenceClassification, logging as trans_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from captum.attr import LayerIntegratedGradients

trans_logging.set_verbosity_error()

# --- LẤY API KEY & CẤU HÌNH ---
def get_key_safe(key_name, default_value):
    try: return userdata.get(key_name)
    except: return default_value

try:
    GEMINI_API_KEY = get_key_safe('GEMINI_API_KEY', 'fake_gemini_key')
    OPENAI_API_KEY = get_key_safe('OPENAI_API_KEY', 'fake_openai_key')
    NGROK_TOKEN = get_key_safe('NGROK_TOKEN', '36eo7ryZNTqDusuTUZIwwIljH0H_4qtuCshpdjJqFgbLKw9MS')
    LS_API_KEY = get_key_safe('LANGCHAIN_API_KEY', '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')

    if GEMINI_API_KEY != 'fake_gemini_key':
        genai.configure(api_key=GEMINI_API_KEY)

    if LS_API_KEY and "lsv2" in LS_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = LS_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = "CPP_Code_Detective"
        print(" LangSmith Tracing: ON")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("ℹ LangSmith Tracing: OFF")
except Exception as e:
    print(f" Lỗi cấu hình: {e}")

# Paths
PATH_OOP = "/content/drive/MyDrive/My_AI_Models/C++_OOP_detection"
PATH_NORMAL = "/content/drive/MyDrive/My_AI_Models/C++_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = 0.5

# ==================================================================================
# 2. LOCAL LLM HANDLER (TÍCH HỢP PERPLEXITY METRIC)
# ==================================================================================
class LocalLLMHandler:
    def __init__(self, model_id="Qwen/Qwen2.5-Coder-7B-Instruct"):
        self.model = None
        self.tokenizer = None
        self.model_id = model_id
        self.is_ready = False

    def load_model(self):
        print(f"\n [LOCAL LLM] Loading {self.model_id} (Router & PPL Engine)...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, quantization_config=bnb_config, device_map="auto"
            )
            self.is_ready = True
            print(" [LOCAL LLM] Ready!")
        except Exception as e:
            print(f" [LOCAL LLM] Failed: {e}")

    def generate(self, prompt, max_new_tokens=250):
        if not self.is_ready: return None
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1)
            ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
        except Exception: return None

    def calculate_perplexity(self, text):
        if not self.is_ready: return 0.0
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            return float(perplexity)
        except Exception as e:
            print(f"⚠️ PPL Error: {e}")
            return 0.0

# ==================================================================================
# 3. MODEL MANAGER (ROBERTA ENGINE)
# ==================================================================================
class ModelManager:
    def __init__(self):
        self.oop_models = []
        self.normal_models = []
        self.tokenizer = None
        self.threshold = DEFAULT_THRESHOLD

    def load_resources(self):
        print("\n [SYSTEM] Loading Roberta Models to RAM (Sleep Mode)...")
        try:
            target = PATH_OOP if os.path.exists(PATH_OOP) else "microsoft/graphcodebert-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(target)
            thresh_file = os.path.join(PATH_OOP, "threshold.json")
            if os.path.exists(thresh_file):
                with open(thresh_file, 'r') as f:
                    data = json.load(f)
                    self.threshold = data.get("best_threshold", DEFAULT_THRESHOLD) if isinstance(data, dict) else data
        except:
            self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

        self.oop_models = self._load_from_disk(PATH_OOP, "OOP")
        self.normal_models = self._load_from_disk(PATH_NORMAL, "NORMAL")
        print(f" [SYSTEM] Ready. Loaded {len(self.oop_models)} OOP & {len(self.normal_models)} Normal models.")

    def _load_from_disk(self, root_path, label):
        loaded = []
        try:
            if not os.path.exists(root_path): return []
            folds = [os.path.join(root_path, d) for d in sorted(os.listdir(root_path)) if d.startswith("fold_")]
            for p in tqdm(folds, desc=f"Loading {label}"):
                model = RobertaForSequenceClassification.from_pretrained(p)
                model.eval()
                loaded.append(model)
        except Exception as e:
            print(f"❌ Error loading {label}: {e}")
        return loaded

    def get_models(self, type_name):
        return self.oop_models if type_name == "OOP" else self.normal_models

local_llm = LocalLLMHandler()
manager = ModelManager()

# ==================================================================================
# 4. HELPER FUNCTIONS & RENDERER
# ==================================================================================
def remove_cpp_comments(text):
    def replacer(match):
        s = match.group(0)
        return " " if s.startswith('/') else s
    pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE)
    return re.sub(pattern, replacer, text)

def heuristic_classify(code_snippet):
    clean_code = remove_cpp_comments(code_snippet)
    score = 0
    if re.search(r'\bclass\s+\w+', clean_code): score += 5
    if re.search(r'\bvirtual\b', clean_code): score += 3
    if re.search(r'\bpublic\s*:', clean_code): score += 2
    if re.search(r'\bprivate\s*:', clean_code): score += 2
    if re.search(r'\bprotected\s*:', clean_code): score += 2
    if re.search(r'\btemplate\s*<', clean_code): score += 2
    if re.search(r':\s*(public|private|protected)\s+\w+', clean_code): score += 4
    colons = len(re.findall(r'\w+::\w+', clean_code))
    if colons > 0: score += min(colons, 3)
    return "OOP" if score >= 4 else "NORMAL"

class ExpertExplainer:
    def analyze(self, tokens, scores, top_k=50):
        clean_tokens = [t.replace('Ġ', '').replace('Ċ', '').strip() for t in tokens]
        top_ai, top_hu = [], []
        indices = np.argsort(np.abs(scores))[-top_k:][::-1]
        seen_ai, seen_hu = set(), set()
        BLACKLIST = {'', '<s>', '</s>', '\n', ' ', '(', ')', '{', '}', ';', ',', '.'}
        for idx in indices:
            raw = clean_tokens[idx]
            s = scores[idx]
            if raw not in BLACKLIST and len(raw) > 1:
                if s > 0 and raw not in seen_ai and len(top_ai) < 10:
                    top_ai.append(f"'{raw}'"); seen_ai.add(raw)
                elif s < 0 and raw not in seen_hu and len(top_hu) < 10:
                    top_hu.append(f"'{raw}'"); seen_hu.add(raw)
        return top_ai, top_hu

explainer = ExpertExplainer()

def render_html_view(tokens, scores, top_k=50, title="CODE ANALYSIS"):
    aggregated_blocks = []
    current_tokens = []
    current_score_sum = 0.0
    token_count = 0
    decode = manager.tokenizer.convert_tokens_to_string

    for i, (token, score) in enumerate(zip(tokens, scores)):
        if token in ['<s>', '</s>', '<pad>']: continue
        is_start_new_word = token.startswith('Ġ') or token.startswith('Ċ')
        if is_start_new_word and current_tokens:
            full_word = decode(current_tokens)
            if full_word:
                avg_score = current_score_sum / token_count if token_count > 0 else 0
                aggregated_blocks.append((full_word, avg_score))
            current_tokens = []; current_score_sum = 0.0; token_count = 0

        current_tokens.append(token)
        current_score_sum += score
        token_count += 1

    if current_tokens:
        full_word = decode(current_tokens)
        if full_word:
            aggregated_blocks.append((full_word, current_score_sum / token_count))

    word_stats = defaultdict(list)
    for word, score in aggregated_blocks:
        clean_word = word.strip()
        if clean_word: word_stats[clean_word].append(score)

    unique_words_list = [(word, sum(score_list) / len(score_list)) for word, score_list in word_stats.items()]
    pos_words = sorted([(w, s) for w, s in unique_words_list if s > 0], key=lambda x: x[1], reverse=True)
    neg_words = sorted([(w, s) for w, s in unique_words_list if s < 0], key=lambda x: x[1])

    ai_list_html = [f"<tr><td style='color:#2ecc71; font-weight:bold; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{html.escape(w)}</td><td style='text-align:right; color:#555; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{s:.3f}</td></tr>" for w,s in pos_words[:top_k]]
    human_list_html = [f"<tr><td style='color:#e74c3c; font-weight:bold; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{html.escape(w)}</td><td style='text-align:right; color:#555; padding:4px 8px; border-bottom:1px solid #f0f0f0;'>{s:.3f}</td></tr>" for w,s in neg_words[:top_k]]

    code_html_parts = []
    all_scores = [abs(s) for w, s in aggregated_blocks]
    max_impact = np.max(all_scores) if len(all_scores) > 0 and np.max(all_scores) > 0 else 1.0

    for word, score in aggregated_blocks:
        safe_word = html.escape(word)
        relative_score = abs(score) / max_impact
        alpha = min(np.power(relative_score, 0.35), 1.0)
        bg_color = "transparent"
        if alpha >= 0.02:
            bg_color = f"rgba(46, 204, 113, {alpha:.2f})" if score > 0 else f"rgba(231, 76, 60, {alpha:.2f})"
        code_html_parts.append(f'<span style="background-color: {bg_color}; border-radius: 2px;">{safe_word}</span>')

    code_html = "".join(code_html_parts)

    return f"""
    <html>
    <head><meta charset="utf-8"></head>
    <body style="margin:0; padding:0; background-color: #f8f9fa;">
        <div style="display: flex; gap: 20px; font-family: 'Segoe UI', sans-serif; margin-bottom: 30px;">
            <div style="flex: 3; border: 1px solid #ccc; border-radius: 8px; background-color: #fff; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
                <div style="background-color: #2c3e50; padding: 12px 15px; border-bottom: 1px solid #000; font-weight: bold; color: #ecf0f1; font-size: 14px;">📑 {title}</div>
                <div style="padding: 15px; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.6; color: #222; height: auto; overflow-y: visible; white-space: pre-wrap;">{code_html}</div>
            </div>
            <div style="flex: 1.2; display: flex; flex-direction: column; gap: 15px; max-height: 800px; position: sticky; top: 0;">
                 <div style="border: 1px solid #f5c6cb; border-radius: 8px; background-color: #fff; overflow: hidden; display: flex; flex-direction: column; flex: 1;">
                    <div style="background-color: #f8d7da; padding: 10px; text-align: center; color: #721c24; font-weight: bold; font-size: 13px; border-bottom: 1px solid #f5c6cb;">👤 Human Signals (Negative)</div>
                    <div style="padding: 0; overflow-y: auto; flex: 1; max-height: 350px;">
                        <table style="width:100%; font-size:12px; border-collapse: collapse; margin: 0;"><tbody>{ "".join(human_list_html) if human_list_html else "<tr><td colspan='2' align='center' style='padding:10px; color:#999'>None</td></tr>" }</tbody></table>
                    </div>
                 </div>
                 <div style="border: 1px solid #c3e6cb; border-radius: 8px; background-color: #fff; overflow: hidden; display: flex; flex-direction: column; flex: 1;">
                    <div style="background-color: #d4edda; padding: 10px; text-align: center; color: #155724; font-weight: bold; font-size: 13px; border-bottom: 1px solid #c3e6cb;">🤖 AI Signals (Positive)</div>
                    <div style="padding: 0; overflow-y: auto; flex: 1; max-height: 350px;">
                        <table style="width:100%; font-size:12px; border-collapse: collapse; margin: 0;"><tbody>{ "".join(ai_list_html) if ai_list_html else "<tr><td colspan='2' align='center' style='padding:10px; color:#999'>None</td></tr>" }</tbody></table>
                    </div>
                 </div>
            </div>
        </div>
    </body>
    </html>
    """

# ==================================================================================
# 5. CORE ENGINE (BATCH PROCESSING + LIG FIX OOM)
# ==================================================================================
def run_roberta_engine(code_text, model_type_name):
    code_text = code_text.replace('\r\n', '\n').strip()
    all_tokens = manager.tokenizer.tokenize(code_text)
    total_tokens = len(all_tokens)

    STRIDE = 256; MAX_LEN = 510
    chunks_info = []
    if total_tokens <= MAX_LEN:
        input_ids = manager.tokenizer.convert_tokens_to_ids([manager.tokenizer.cls_token] + all_tokens + [manager.tokenizer.sep_token])
        chunks_info.append({ "ids": input_ids, "start_idx": 0, "end_idx": total_tokens })
    else:
        for i in range(0, total_tokens, STRIDE):
            chunk_toks = all_tokens[i : i + MAX_LEN]
            input_ids = manager.tokenizer.convert_tokens_to_ids([manager.tokenizer.cls_token] + chunk_toks + [manager.tokenizer.sep_token])
            chunks_info.append({ "ids": input_ids, "start_idx": i, "end_idx": i + len(chunk_toks) })
            if i + MAX_LEN >= total_tokens: break

    print(f"📄 Processing {total_tokens} tokens in {len(chunks_info)} chunks.")
    chunk_results_accum = [{"prob_sum": 0.0, "attrs_sum": None, "count": 0} for _ in chunks_info]

    target_type = "OOP" if "OOP" in model_type_name else "NORMAL"
    models_in_ram = manager.get_models(target_type)
    if not models_in_ram: return {"error": f"No preloaded models for {model_type_name}"}

    global_attrs = np.zeros(total_tokens); global_counts = np.zeros(total_tokens)
    total_folds = len(models_in_ram)

    BATCH_SIZE = 4

    for fold_idx, model_cpu in enumerate(tqdm(models_in_ram, desc=f"Running {model_type_name}")):
        print(f"🔄 [ENGINE] Running Fold {fold_idx + 1}/{total_folds}...")
        try:
            model = model_cpu.to(DEVICE)

            def predict_func(inputs, attention_mask=None):
                return model(inputs_embeds=inputs, attention_mask=attention_mask).logits

            lig = LayerIntegratedGradients(predict_func, model.roberta.embeddings)

            for batch_start in range(0, len(chunks_info), BATCH_SIZE):
                batch_chunks = chunks_info[batch_start : batch_start + BATCH_SIZE]
                max_len_in_batch = max(len(c["ids"]) for c in batch_chunks)

                padded_input_ids = []
                attention_masks = []
                padded_bases = []

                for c in batch_chunks:
                    pad_len = max_len_in_batch - len(c["ids"])
                    padded_input_ids.append(c["ids"] + [manager.tokenizer.pad_token_id] * pad_len)
                    attention_masks.append([1] * len(c["ids"]) + [0] * pad_len)
                    padded_bases.append([manager.tokenizer.pad_token_id] * max_len_in_batch)

                inp = torch.tensor(padded_input_ids).to(DEVICE)
                mask_tensor = torch.tensor(attention_masks).to(DEVICE)
                base = torch.tensor(padded_bases).to(DEVICE)

                with torch.no_grad():
                    logits = model(inp, attention_mask=mask_tensor).logits
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                embed_inp = model.roberta.embeddings(inp)
                embed_base = model.roberta.embeddings(base)

                attrs = lig.attribute(
                    inputs=embed_inp,
                    baselines=embed_base,
                    target=1,
                    n_steps=20,
                    internal_batch_size=4,
                    additional_forward_args=(mask_tensor,)
                )

                attrs_np = attrs.sum(dim=2).cpu().detach().numpy()

                for j, prob in enumerate(probs):
                    global_idx = batch_start + j
                    actual_len = len(batch_chunks[j]["ids"])
                    valid_attrs = attrs_np[j][:actual_len]
                    clean_prob = float(prob)

                    print(f"    ├── Chunk {global_idx+1}/{len(chunks_info)}: Score {clean_prob:.4f}")
                    chunk_results_accum[global_idx]["prob_sum"] += clean_prob
                    chunk_results_accum[global_idx]["count"] += 1

                    if chunk_results_accum[global_idx]["attrs_sum"] is None:
                        chunk_results_accum[global_idx]["attrs_sum"] = valid_attrs
                    else:
                        chunk_results_accum[global_idx]["attrs_sum"] += valid_attrs

                del inp, mask_tensor, base, embed_inp, embed_base, logits, attrs
                torch.cuda.empty_cache()
                gc.collect()

            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"⚠️ Error in Fold {fold_idx + 1}: {e}")
            try:
                model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
            except: pass

    chunk_results_data = []
    weighted_prob_sum = 0.0; total_weight = 0.0

    for i, res in enumerate(chunk_results_accum):
        if res["count"] == 0: continue
        avg_prob = float(res["prob_sum"] / res["count"])
        avg_attrs = res["attrs_sum"] / res["count"]
        chunk_len = len(chunks_info[i]["ids"]) - 2
        chunk_label = "AI" if avg_prob >= manager.threshold else "HUMAN"
        chunk_tokens = manager.tokenizer.convert_ids_to_tokens(chunks_info[i]["ids"])
        c_ai, c_hu = explainer.analyze(chunk_tokens, avg_attrs)
        snippet = manager.tokenizer.decode(chunks_info[i]["ids"], skip_special_tokens=True)
        chunk_html = render_html_view(chunk_tokens, avg_attrs, top_k=50, title=f"CHUNK {i+1} ({avg_prob:.4f})")
        chunk_results_data.append({"index": i+1, "score": avg_prob, "label": chunk_label, "top_ai": c_ai, "top_hu": c_hu, "snippet": snippet, "html": chunk_html})
        weighted_prob_sum += avg_prob * chunk_len; total_weight += chunk_len
        valid = avg_attrs[1:-1]; s_idx = chunks_info[i]["start_idx"]; l = min(len(valid), chunks_info[i]["end_idx"] - s_idx)
        global_attrs[s_idx : s_idx+l] += valid[:l]; global_counts[s_idx : s_idx+l] += 1

    final_attrs = np.divide(global_attrs, global_counts, out=np.zeros_like(global_attrs), where=global_counts!=0)
    final_avg = float(weighted_prob_sum / total_weight) if total_weight > 0 else 0.5
    final_pred = "AI GENERATED" if final_avg >= manager.threshold else "HUMAN WRITTEN"
    global_html = render_html_view(all_tokens, final_attrs, top_k=100, title=f"GLOBAL ({final_pred})")
    g_ai, g_hu = explainer.analyze(all_tokens, final_attrs, top_k=9999)

    return {
        "model_used": model_type_name, "final_pred": final_pred, "final_score": final_avg,
        "total_tokens": total_tokens, "total_chunks": len(chunks_info),
        "global_html": global_html, "chunks": chunk_results_data,
        "g_ai": g_ai, "g_hu": g_hu
    }

# ==================================================================================
# 6. LANGGRAPH AGENT (ASYNC + STRUCTURED OUTPUT + PERPLEXITY)
# ==================================================================================

class AgentState(TypedDict):
    code_input: str
    classification: Literal["OOP", "NORMAL", "ERROR"]
    final_output: dict
    retry_count: int
    is_ambiguous: bool

def router_node(state: AgentState):
    print("\n🤖 [AGENT] Router Node: Analyzing Code Structure...")
    if state.get("retry_count", 0) > 0:
        print(f"🔄 [RETRY LOOP] Skipping router check. Forced Class: {state['classification']}")
        return {}

    code = state["code_input"][:2000]
    router_prompt = ChatPromptTemplate.from_template("""
        You are a C++ Expert Classifier.
        Analyze the following code snippet.
        CRITERIA:
        - "OOP": Contains `class`, `inheritance`, `virtual`, `template`, `public/private` specifiers.
        - "NORMAL": Contains `struct`, procedural logic, simple main functions, competitive programming style.
        OUTPUT FORMAT: Return ONLY one word: "OOP" or "NORMAL".
        CODE: {code}
    """)
    final_prompt_str = router_prompt.format(code=code)

    decision = None
    if local_llm.is_ready:
        try:
            print(f"   ▶️  Using Model: Local Qwen ({local_llm.model_id})")
            res = local_llm.generate(final_prompt_str, max_new_tokens=10)
            if res:
                if "OOP" in res.upper(): decision = "OOP"
                elif "NORMAL" in res.upper(): decision = "NORMAL"
        except: pass

    if not decision:
        print("🔄 Router Fallback to Heuristic...")
        decision = heuristic_classify(state["code_input"])

    print(f"🎯 Router Decision: {decision}")
    return {"classification": decision, "retry_count": 0}

def analyzer_node(state: AgentState):
    decision = state["classification"]
    model_name = "C++ OOP Model" if decision == "OOP" else "C++ Normal Model"
    print(f"⚙️ [AGENT] Analyzer Node: Running {model_name}...")

    # 1. Chạy Classification Model (RoBERTa)
    print(f"   ▶️  Using Model: RoBERTa Ensemble (Microsoft/GraphCodeBERT Architecture)")
    result = run_roberta_engine(state["code_input"], model_name)

    # 2. Tính toán Perplexity (SOTA)
    print(f"   ▶️  Calculating Perplexity using Generative Model...")
    ppl = local_llm.calculate_perplexity(state["code_input"])
    result["perplexity"] = ppl
    print(f"   📊  Perplexity Score: {ppl:.2f}")

    return {"final_output": result}

def judge_node(state: AgentState):
    res = state["final_output"]
    score = res.get("final_score", 0.5)
    ppl = res.get("perplexity", 0.0)
    retry = state.get("retry_count", 0)
    print(f"⚖️ [JUDGE] Confidence Score: {score:.4f} | PPL: {ppl:.2f} | Retries: {retry}")

    is_suspicious_human = score > 0.6 and ppl > 5.0
    is_suspicious_ai = score < 0.4 and ppl < 1.5
    is_ambiguous_score = 0.40 <= score <= 0.60

    if (is_ambiguous_score or is_suspicious_human or is_suspicious_ai) and retry < 1:
        print("⚠️ Ambiguous or Conflicting Result. Self-Correction Activated!")
        print("🔄 Switching Model...")
        new_class = "NORMAL" if state["classification"] == "OOP" else "OOP"
        return {"classification": new_class, "retry_count": retry + 1, "is_ambiguous": True}

    print("✅ Result Accepted. Proceeding to Critique.")
    return {"is_ambiguous": False}

class ChunkAnalysis(BaseModel):
    summary: str = Field(description="A short critique paragraph (max 2 sentences) explaining the score.")
    key_features: List[str] = Field(description="A list of 3 to 5 key technical features found in the code.")
    style: str = Field(description="A short description of the coding style.")

async def critique_node(state: AgentState):
    res = state["final_output"]
    if "error" in res: return {}

    parser = PydanticOutputParser(pydantic_object=ChunkAnalysis)
    format_instructions = parser.get_format_instructions()

    map_prompt_template = ChatPromptTemplate.from_template("""
        Role: Code Security Auditor.
        Task: Analyze this C++ code snippet.

        Input Data:
        - AI Confidence Score: {score}
        - Top AI Signals: {signals}
        - Code Snippet: {snippet}

        {format_instructions}
    """)

    async def try_generate_chunk_analysis(inputs):
        prompt_str = map_prompt_template.format(
            score=inputs['score'],
            signals=inputs['signals'],
            snippet=inputs['snippet'],
            format_instructions=format_instructions
        )

        raw_text = "Analysis unavailable"
        parsed_data = {"summary": "Analysis unavailable", "key_features": [], "style": "Unknown"}

        if GEMINI_API_KEY and GEMINI_API_KEY != "fake_gemini_key":
            try:
                print(f"      ▶️  Using Model: Gemini 1.5 Flash (Async)")
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.3)
                response = await llm.ainvoke(prompt_str)
                raw_text = response.content
            except: pass
        elif OPENAI_API_KEY and OPENAI_API_KEY != "fake_openai_key":
            try:
                print(f"      ▶️  Using Model: GPT-4o (Async)")
                llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.3)
                response = await llm.ainvoke(prompt_str)
                raw_text = response.content
            except: pass
        elif local_llm.is_ready:
            try:
                print(f"      ▶️  Using Model: Local ({local_llm.model_id})")
                raw_text = local_llm.generate(prompt_str, 300)
            except: pass

        try:
            if raw_text:
                parsed_result = parser.parse(raw_text)
                parsed_data = {
                    "summary": parsed_result.summary,
                    "key_features": parsed_result.key_features,
                    "style": parsed_result.style
                }
        except Exception as e:
            print(f"⚠️ Parse Error/Fallback: {e}")
            if raw_text: parsed_data["summary"] = raw_text[:200]

        return raw_text, parsed_data

    print(f"\n📝 [AGENT] Critique Node: Starting MAP Phase (Analyzing {len(res['chunks'])} chunks)...")

    chunk_metadata_list = []

    for i, chunk in enumerate(res["chunks"]):
        print(f" 🔹 [MAP] Chunk {i+1} Analysis:")
        raw_text, parsed_data = await try_generate_chunk_analysis({
            "score": f"{chunk['score']:.2f}",
            "signals": f"{', '.join(chunk['top_ai'][:3])}",
            "snippet": chunk['snippet'][:400]
        })

        meta = {
            "chunk_id": i+1,
            "role": chunk['label'],
            "score": chunk['score'],
            "summary": parsed_data["summary"],
            "keywords": parsed_data["key_features"],
            "style": parsed_data["style"]
        }
        chunk_metadata_list.append(meta)
        chunk["critique"] = parsed_data["summary"]

        print(f"   - Extracted JSON: {json.dumps(meta, indent=2, ensure_ascii=False)}")
        print("-" * 40)

    print("\n📝 [AGENT] Starting REDUCE Phase (Global Analysis)...")

    reduce_prompt_template = ChatPromptTemplate.from_template("""
        Role: Senior Code Auditor.
        Task: Analyze the following Audit Logs (Summaries from code chunks) and provide a final verdict.

        INPUT CONTEXT (Audit Logs):
        {context_logs}

        INSTRUCTIONS:
        1. Analyze the trend of scores and keywords across chunks.
        2. Identify if the code is consistent or has mixed styles.
        3. Provide a final conclusion on the authorship (Pure AI, Pure Human, or Hybrid).
        4. Keep the output concise (under 3 sentences).
    """)

    context_logs_str = json.dumps(chunk_metadata_list, indent=1)
    print(f" 🔸 [REDUCE INPUT] Context sent to Global Model:\n{context_logs_str[:1000]}... (truncated)\n")

    async def generate_final_verdict(prompt_text):
        if GEMINI_API_KEY and GEMINI_API_KEY != "fake_gemini_key":
            try:
                print(f"   ▶️  Using Model: Gemini 1.5 Flash (Async)")
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
                res = await llm.ainvoke(prompt_text)
                return res.content
            except: pass
        if OPENAI_API_KEY and OPENAI_API_KEY != "fake_openai_key":
            try:
                print(f"   ▶️  Using Model: GPT-4o (Async)")
                llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.7)
                res = await llm.ainvoke(prompt_text)
                return res.content
            except: pass
        if local_llm.is_ready:
            print(f"   ▶️  Using Model: Local ({local_llm.model_id})")
            return local_llm.generate(prompt_text, 150)
        return "Global summary unavailable."

    final_prompt = reduce_prompt_template.format(context_logs=context_logs_str)
    res["global_critique"] = await generate_final_verdict(final_prompt)

    print(f"✅ [DONE] Global Critique: {res['global_critique']}")
    return {"final_output": res}

def should_continue(state: AgentState):
    if state.get("is_ambiguous", False): return "analyzer"
    return "critique"

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("judge", judge_node)
workflow.add_node("critique", critique_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "analyzer")
workflow.add_edge("analyzer", "judge")
workflow.add_conditional_edges("judge", should_continue, {"analyzer": "analyzer", "critique": "critique"})
workflow.add_edge("critique", END)
agent_app = workflow.compile()

# ==================================================================================
# 7. FASTAPI SERVER (LIFESPAN + ASYNC + SSE STREAMING)
# ==================================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("🚀 [STARTUP] ĐANG KHỞI ĐỘNG HỆ THỐNG AI TẠI GLOBAL STATE...")
    local_llm.load_model()
    manager.load_resources()
    print("✅ [STARTUP] HOÀN TẤT! SERVER ĐÃ SẴN SÀNG ĐÓN KHÁCH.")
    print("="*50 + "\n")

    yield

    print("\n" + "="*50)
    print("🛑 [SHUTDOWN] ĐANG TẮT SERVER, TIẾN HÀNH DỌN DẸP VRAM...")
    if local_llm.model:
        local_llm.model.cpu()
        del local_llm.model
    if manager.oop_models:
        del manager.oop_models
        del manager.normal_models
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ [SHUTDOWN] ĐÃ XẢ SẠCH VRAM. HẸN GẶP LẠI!")
    print("="*50 + "\n")

app = FastAPI(lifespan=lifespan)

class CodeRequest(BaseModel):
    code_base64: str

RESULT_CACHE = {}
MAX_CACHE_SIZE = 100

@app.get("/")
async def home(): return {"status": "Online", "mode": "Agentic Workflow SSE"}

# 7A. ENDPOINT BÌNH THƯỜNG (Giữ lại để tương thích ngược / test nhanh)
@app.post("/api/analyze")
async def analyze_code(request: CodeRequest):
    if not request.code_base64.strip(): return {"error": "Empty code"}
    try:
        decoded_str = base64.b64decode(request.code_base64).decode('utf-8').replace('\r\n', '\n').strip()
        print(f"\n📥 [REQUEST SYNC] Code Length: {len(decoded_str)}")

        code_hash = hashlib.md5(decoded_str.encode('utf-8')).hexdigest()
        if code_hash in RESULT_CACHE:
            return {"result": RESULT_CACHE[code_hash]}

        final_state = await agent_app.ainvoke({"code_input": decoded_str, "retry_count": 0, "is_ambiguous": False})
        result = final_state["final_output"]

        if "error" not in result:
            if len(RESULT_CACHE) >= MAX_CACHE_SIZE:
                oldest_key = next(iter(RESULT_CACHE))
                del RESULT_CACHE[oldest_key]
            RESULT_CACHE[code_hash] = result

        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# 7B. ENDPOINT STREAMING SSE MỚI (Dùng cho Giao diện "Dây thun")
@app.post("/api/analyze_stream")
async def analyze_code_stream(request: CodeRequest):
    async def event_generator():
        try:
            if not request.code_base64.strip():
                yield f"data: {json.dumps({'error': 'Empty code'})}\n\n"
                return

            decoded_str = base64.b64decode(request.code_base64).decode('utf-8').replace('\r\n', '\n').strip()
            print(f"\n🌊 [REQUEST STREAM] Code Length: {len(decoded_str)}")

            # Tính toán ước lượng thời gian chạy RoBERTa (để UI chuẩn bị thanh tiến trình dài)
            # Code càng dài -> Chạy càng lâu -> Khúc chờ từ 20% đến 60% càng tốn thời gian
            estimated_roberta_time = max(5, (len(decoded_str) // 1500) * 8)

            code_hash = hashlib.md5(decoded_str.encode('utf-8')).hexdigest()

            yield f"data: {json.dumps({'progress': 5, 'message': 'Đã nhận code, chuẩn bị khởi động LangGraph...'})}\n\n"
            await asyncio.sleep(0.1)

            if code_hash in RESULT_CACHE:
                print("⚡ [CACHE HIT] Đã từng phân tích code này! Stream ngay lập tức.")
                yield f"data: {json.dumps({'progress': 100, 'message': 'Lấy kết quả từ Cache tốc độ cao!', 'result': RESULT_CACHE[code_hash]})}\n\n"
                return

            final_output = None

            async for event in agent_app.astream({"code_input": decoded_str, "retry_count": 0, "is_ambiguous": False}):
                for node_name, node_state in event.items():
                    if node_name == "router":
                        decision = node_state.get('classification', 'Unknown')
                        # Gửi thêm tham số `estimated_time` để báo cho Streamlit biết khúc này sẽ kẹt lâu
                        yield f"data: {json.dumps({'progress': 20, 'message': f'Định tuyến xong ({decision}). Đang chạy RoBERTa Engine...', 'estimated_time': estimated_roberta_time})}\n\n"
                    elif node_name == "analyzer":
                        yield f"data: {json.dumps({'progress': 65, 'message': 'Hoàn tất RoBERTa & Tính toán Perplexity. Đang đánh giá...'})}\n\n"
                    elif node_name == "judge":
                        if node_state.get('is_ambiguous'):
                            yield f"data: {json.dumps({'progress': 70, 'message': 'Phát hiện nhập nhằng! Đang tự sửa sai đổi Model...'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'progress': 75, 'message': 'Đánh giá xong. Đang gọi LLM tổng hợp (Critique)...', 'estimated_time': 10})}\n\n"
                    elif node_name == "critique":
                        yield f"data: {json.dumps({'progress': 95, 'message': 'Hoàn tất phân tích Map-Reduce!'})}\n\n"

                    if "final_output" in node_state:
                        final_output = node_state["final_output"]

            if final_output:
                if len(RESULT_CACHE) >= MAX_CACHE_SIZE:
                    oldest_key = next(iter(RESULT_CACHE))
                    del RESULT_CACHE[oldest_key]
                RESULT_CACHE[code_hash] = final_output
                print(f"✅ [DONE STREAM] Score: {final_output['final_score']:.4f}")
                yield f"data: {json.dumps({'progress': 100, 'message': 'Xử lý thành công!', 'result': final_output})}\n\n"
            else:
                yield f"data: {json.dumps({'error': 'Không thu thập được kết quả từ Graph.'})}\n\n"

        except Exception as e:
            print(f"❌ Streaming Error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

if __name__ == "__main__":
    print("🔄 Đang chuẩn bị Ngrok...")
    time.sleep(2)
    if NGROK_TOKEN and NGROK_TOKEN != "fake_token": ngrok.set_auth_token(NGROK_TOKEN)
    else: print("⚠️ Ngrok Token invalid.")
    try:
        public_url = ngrok.connect(8000).public_url
        print(f"\n🚀 TÊN MIỀN NGROK (Copy vào Streamlit): {public_url}")
        nest_asyncio.apply()
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        while True: time.sleep(1)
    except Exception as e: print(f"❌ Ngrok Error: {e}")

