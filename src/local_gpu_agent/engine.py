"""
engine.py — Core AI Engine: ModelManager, RoBERTa Ensemble, LIG, Heuristic, HTML Renderer
Wrap từ extract_1.py §3-§5 (lines 165-470)
"""
import os
import gc
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients

from .config import (
    DEVICE, PATH_OOP, PATH_NORMAL, DEFAULT_THRESHOLD,
    MAX_LEN, STRIDE, LIG_N_STEPS, LIG_BATCH_SIZE, ENGINE_BATCH_SIZE,
)


# HTML rendering and ExpertExplainer logic has been moved to the Local Brain.
# Colab now returns raw tokens and attribution scores.


# ==================================================================================
# MODEL MANAGER — Load RoBERTa K-Fold Models
# ==================================================================================

class ModelManager:
    """Quản lý load/get RoBERTa models từ Google Drive"""

    def __init__(self):
        self.oop_models = []
        self.normal_models = []
        self.tokenizer = None
        self.threshold = DEFAULT_THRESHOLD

    def load_resources(self):
        """Load tokenizer, threshold, và tất cả fold models vào RAM"""
        print("\n🔧 [SYSTEM] Loading RoBERTa Models to RAM (Sleep Mode)...")
        try:
            target = PATH_OOP if os.path.exists(PATH_OOP) else "microsoft/graphcodebert-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(target)
            thresh_file = os.path.join(PATH_OOP, "threshold.json")
            if os.path.exists(thresh_file):
                with open(thresh_file, 'r') as f:
                    data = json.load(f)
                    self.threshold = (
                        data.get("best_threshold", DEFAULT_THRESHOLD)
                        if isinstance(data, dict)
                        else data
                    )
        except Exception:
            self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

        self.oop_models = self._load_from_disk(PATH_OOP, "OOP")
        self.normal_models = self._load_from_disk(PATH_NORMAL, "NORMAL")
        print(f"✅ [SYSTEM] Ready. Loaded {len(self.oop_models)} OOP & {len(self.normal_models)} Normal models.")

    def _load_from_disk(self, root_path, label):
        loaded = []
        try:
            if not os.path.exists(root_path):
                return []
            folds = [
                os.path.join(root_path, d)
                for d in sorted(os.listdir(root_path))
                if d.startswith("fold_")
            ]
            for p in tqdm(folds, desc=f"Loading {label}"):
                model = RobertaForSequenceClassification.from_pretrained(p)
                model.eval()
                loaded.append(model)
        except Exception as e:
            print(f"❌ Error loading {label}: {e}")
        return loaded

    def get_models(self, type_name):
        return self.oop_models if type_name == "OOP" else self.normal_models

    def cleanup(self):
        """Giải phóng VRAM"""
        del self.oop_models
        del self.normal_models
        self.oop_models = []
        self.normal_models = []
        torch.cuda.empty_cache()
        gc.collect()


# ==================================================================================
# CORE ENGINE — RoBERTa Ensemble + LIG Attribution
# ==================================================================================

def run_roberta_engine(code_text, model_type_name, manager):
    """
    Chạy RoBERTa Ensemble inference + LIG attribution trên code C++.

    Args:
        code_text: Code C++ đã decode
        model_type_name: "C++ OOP Model" hoặc "C++ Normal Model"
        manager: ModelManager instance

    Returns:
        dict với keys: model_used, final_pred, final_score, total_tokens,
                       total_chunks, tokens, attrs, chunks
    """
    code_text = code_text.replace('\r\n', '\n').strip()
    all_tokens = manager.tokenizer.tokenize(code_text)
    total_tokens = len(all_tokens)

    # --- Chunking ---
    chunks_info = []
    if total_tokens <= MAX_LEN:
        input_ids = manager.tokenizer.convert_tokens_to_ids(
            [manager.tokenizer.cls_token] + all_tokens + [manager.tokenizer.sep_token]
        )
        chunks_info.append({"ids": input_ids, "start_idx": 0, "end_idx": total_tokens})
    else:
        for i in range(0, total_tokens, STRIDE):
            chunk_toks = all_tokens[i : i + MAX_LEN]
            input_ids = manager.tokenizer.convert_tokens_to_ids(
                [manager.tokenizer.cls_token] + chunk_toks + [manager.tokenizer.sep_token]
            )
            chunks_info.append({"ids": input_ids, "start_idx": i, "end_idx": i + len(chunk_toks)})
            if i + MAX_LEN >= total_tokens:
                break

    print(f"📄 Processing {total_tokens} tokens in {len(chunks_info)} chunks.")
    chunk_results_accum = [{"prob_sum": 0.0, "attrs_sum": None, "count": 0} for _ in chunks_info]

    target_type = "OOP" if "OOP" in model_type_name else "NORMAL"
    models_in_ram = manager.get_models(target_type)
    if not models_in_ram:
        return {"error": f"No preloaded models for {model_type_name}"}

    global_attrs = np.zeros(total_tokens)
    global_counts = np.zeros(total_tokens)
    total_folds = len(models_in_ram)

    # --- Fold-level inference + LIG ---
    for fold_idx, model_cpu in enumerate(tqdm(models_in_ram, desc=f"Running {model_type_name}")):
        print(f"🔄 [ENGINE] Running Fold {fold_idx + 1}/{total_folds}...")
        try:
            model = model_cpu.to(DEVICE)

            def predict_func(inputs, attention_mask=None):
                return model(inputs_embeds=inputs, attention_mask=attention_mask).logits

            lig = LayerIntegratedGradients(predict_func, model.roberta.embeddings)

            for batch_start in range(0, len(chunks_info), ENGINE_BATCH_SIZE):
                batch_chunks = chunks_info[batch_start : batch_start + ENGINE_BATCH_SIZE]
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
                    n_steps=LIG_N_STEPS,
                    internal_batch_size=LIG_BATCH_SIZE,
                    additional_forward_args=(mask_tensor,),
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
            except Exception:
                pass

    # --- Aggregate results ---
    chunk_results_data = []
    weighted_prob_sum = 0.0
    total_weight = 0.0

    for i, res in enumerate(chunk_results_accum):
        if res["count"] == 0:
            continue
        avg_prob = float(res["prob_sum"] / res["count"])
        avg_attrs = res["attrs_sum"] / res["count"]
        chunk_len = len(chunks_info[i]["ids"]) - 2
        chunk_label = "AI" if avg_prob >= manager.threshold else "HUMAN"

        chunk_tokens = manager.tokenizer.convert_ids_to_tokens(chunks_info[i]["ids"])
        snippet = manager.tokenizer.decode(chunks_info[i]["ids"], skip_special_tokens=True)

        chunk_results_data.append({
            "index": i + 1,
            "score": avg_prob,
            "label": chunk_label,
            "tokens": chunk_tokens,
            "attrs": avg_attrs.tolist(), # Convert to list for JSON
            "snippet": snippet,
        })

        weighted_prob_sum += avg_prob * chunk_len
        total_weight += chunk_len

        valid = avg_attrs[1:-1]
        s_idx = chunks_info[i]["start_idx"]
        l = min(len(valid), chunks_info[i]["end_idx"] - s_idx)
        global_attrs[s_idx : s_idx + l] += valid[:l]
        global_counts[s_idx : s_idx + l] += 1

    final_attrs = np.divide(
        global_attrs, global_counts,
        out=np.zeros_like(global_attrs),
        where=global_counts != 0,
    )
    final_avg = float(weighted_prob_sum / total_weight) if total_weight > 0 else 0.5
    final_pred = "AI GENERATED" if final_avg >= manager.threshold else "HUMAN WRITTEN"

    return {
        "model_used": model_type_name,
        "final_pred": final_pred,
        "final_score": final_avg,
        "total_tokens": total_tokens,
        "total_chunks": len(chunks_info),
        "tokens": all_tokens,
        "attrs": final_attrs.tolist(), # Convert to list for JSON
        "chunks": chunk_results_data,
    }
