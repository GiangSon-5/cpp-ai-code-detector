"""
engine.py — Core AI Engine cho Local GPU Server (1-fold, 4GB VRAM)

Sao chép logic từ colab_runtime/scripts/engine.py với các điều chỉnh:
  1. Chỉ load 1 fold (fold đầu tiên tìm thấy trong PATH_OOP / PATH_NORMAL)
  2. ENGINE_BATCH_SIZE = 1 và LIG_BATCH_SIZE = 1 để an toàn VRAM
  3. Model được offload về CPU ngay sau mỗi batch (không giữ trên GPU)
  4. Không phụ thuộc vào Google Drive hay ngrok
"""

import os
import gc
import json
import time
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients

from .config import (
    DEVICE, PATH_OOP, PATH_NORMAL, DEFAULT_THRESHOLD,
    MAX_LEN, STRIDE, LIG_N_STEPS, LIG_BATCH_SIZE, ENGINE_BATCH_SIZE,
    gpu_log,
)


# ==================================================================================
# MODEL MANAGER — Load chỉ 1 fold RoBERTa (4GB VRAM safe)
# ==================================================================================

class LocalModelManager:
    """
    Quản lý load / get RoBERTa single-fold models từ đường dẫn local.

    Khác biệt với Colab ModelManager:
      - Chỉ load fold ĐẦU TIÊN tìm thấy (không load toàn bộ 5 folds)
      - Model không bị giữ liên tục trên GPU — chỉ move lên khi inference,
        sau đó move về CPU + empty_cache() ngay
    """

    def __init__(self):
        self.oop_models: list = []
        self.normal_models: list = []
        self.tokenizer: RobertaTokenizer | None = None
        self.threshold: float = DEFAULT_THRESHOLD
        self._loaded: bool = False

    # ------------------------------------------------------------------
    def load_resources(self) -> dict:
        """Load tokenizer và 1 fold mỗi loại vào CPU RAM."""
        # KHÔNG early-return nếu đã loaded — vì _loaded=True nhưng 1 type có thể vẫn rỗng
        # (load thất bại silently lần trước). Dùng ensure_type_loaded() để lazy-reload.
        if self._loaded and self.oop_models and self.normal_models:
            return {"status": "already_loaded"}

        t0 = time.perf_counter()
        print("\n🔧 [LOCAL GPU] Loading Local RoBERTa Models (1 fold each)...")
        gpu_log("info", "engine", "load_resources", "Starting model load")

        # --- Tokenizer (chỉ load lần đầu) ---
        if self.tokenizer is None:
            try:
                tok_src = PATH_OOP if os.path.isdir(PATH_OOP) else "microsoft/graphcodebert-base"
                self.tokenizer = RobertaTokenizer.from_pretrained(tok_src)
                gpu_log("info", "engine", "load_resources",
                        f"Tokenizer loaded from: {tok_src}")
            except Exception as exc:
                gpu_log("warning", "engine", "load_resources",
                        f"Fallback tokenizer: {exc}")
                self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

        # --- Threshold ---
        thresh_file = os.path.join(PATH_OOP, "threshold.json")
        if os.path.exists(thresh_file):
            try:
                with open(thresh_file) as f:
                    data = json.load(f)
                self.threshold = (
                    data.get("best_threshold", DEFAULT_THRESHOLD)
                    if isinstance(data, dict) else data
                )
                gpu_log("info", "engine", "load_resources",
                        f"Threshold loaded: {self.threshold}")
            except Exception:
                pass

        # --- Load single folds (chỉ load loại nào còn rỗng) ---
        if not self.oop_models:
            self.oop_models = self._load_single_fold(PATH_OOP, "OOP")
        if not self.normal_models:
            self.normal_models = self._load_single_fold(PATH_NORMAL, "NORMAL")

        latency_ms = (time.perf_counter() - t0) * 1000
        self._loaded = True
        msg = (f"Ready. OOP: {len(self.oop_models)} fold(s), "
               f"NORMAL: {len(self.normal_models)} fold(s).")
        print(f"✅ [LOCAL GPU] {msg} ({latency_ms:.0f}ms)")
        gpu_log("info", "engine", "load_resources", msg,
                latency_ms=round(latency_ms, 1))
        return {"status": "loaded", "oop_folds": len(self.oop_models),
                "normal_folds": len(self.normal_models)}

    # ------------------------------------------------------------------
    def _load_single_fold(self, root_path: str, label: str) -> list:
        """
        Load chỉ fold đầu tiên tìm thấy trong root_path.
        Nếu root_path chính nó là 1 fold (chứa config.json), load nó trực tiếp.
        """
        if not os.path.isdir(root_path):
            print(f"⚠️  [LOCAL GPU] Path không tồn tại: {root_path}")
            gpu_log("warning", "engine", "_load_single_fold",
                    f"Path not found: {root_path}")
            return []

        # Trường hợp 1: root_path là folder của 1 fold (chứa config.json)
        if os.path.exists(os.path.join(root_path, "config.json")):
            fold_path = root_path
        else:
            # Trường hợp 2: root_path chứa nhiều thư mục fold_1, fold_2 ...
            sub_folds = sorted(
                d for d in os.listdir(root_path)
                if d.startswith("fold_") and os.path.isdir(os.path.join(root_path, d))
            )
            if not sub_folds:
                print(f"⚠️  [LOCAL GPU] Không tìm thấy folder fold_* trong: {root_path}")
                gpu_log("warning", "engine", "_load_single_fold",
                        f"No fold_* subdirs in: {root_path}")
                return []
            fold_path = os.path.join(root_path, sub_folds[0])
            print(f"ℹ️  [LOCAL GPU] Sẽ load fold đầu tiên: {sub_folds[0]}")

        try:
            print(f"⏳ [LOCAL GPU] Loading {label} model from: {fold_path}")
            import traceback as _tb
            model = RobertaForSequenceClassification.from_pretrained(
                fold_path,
                ignore_mismatched_sizes=True,  # tránh lỗi nếu config khác
            )
            model.eval()
            # Giữ trên CPU, chỉ push lên GPU khi inference
            model.cpu()
            print(f"✅ [LOCAL GPU] Loaded {label} model OK ({fold_path})")
            gpu_log("info", "engine", "_load_single_fold",
                    f"Loaded 1 {label} fold from: {fold_path}")
            return [model]
        except Exception as exc:
            print(f"❌ [LOCAL GPU] Lỗi load {label}: {exc}")
            _tb.print_exc()   # In full traceback để debug
            gpu_log("error", "engine", "_load_single_fold",
                    f"Error loading {label}: {exc}")
            return []

    # ------------------------------------------------------------------
    def get_models(self, type_name: str) -> list:
        return self.oop_models if type_name == "OOP" else self.normal_models

    # ------------------------------------------------------------------
    def ensure_type_loaded(self, type_name: str) -> list:
        """
        Đảm bảo model của type_name đã được load.
        Nếu chưa (do startup load thất bại), tự động load lại.
        Trả về list models (rỗng nếu load thất bại).
        """
        models = self.get_models(type_name)
        if models:
            return models

        # Lazy-reload: tokenizer trước nếu chưa có
        if self.tokenizer is None:
            try:
                tok_src = PATH_OOP if os.path.isdir(PATH_OOP) else "microsoft/graphcodebert-base"
                self.tokenizer = RobertaTokenizer.from_pretrained(tok_src)
            except Exception:
                self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

        path = PATH_OOP if type_name == "OOP" else PATH_NORMAL
        print(f"⚠️  [LOCAL GPU] {type_name} models empty — lazy-loading now from: {path}")
        gpu_log("warning", "engine", "ensure_type_loaded",
                f"{type_name} empty, lazy-loading", path=path)

        loaded = self._load_single_fold(path, type_name)
        if type_name == "OOP":
            self.oop_models = loaded
        else:
            self.normal_models = loaded

        if loaded:
            print(f"✅ [LOCAL GPU] Lazy-load {type_name} OK ({len(loaded)} fold)")
        else:
            print(f"❌ [LOCAL GPU] Lazy-load {type_name} FAILED — xem traceback bên trên")
        return loaded

    # ------------------------------------------------------------------
    def cleanup(self):
        """Giải phóng VRAM và CPU RAM."""
        gpu_log("info", "engine", "cleanup", "Releasing model memory")
        del self.oop_models
        del self.normal_models
        self.oop_models = []
        self.normal_models = []
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ [LOCAL GPU] VRAM released.")


# ==================================================================================
# CORE ENGINE — RoBERTa Single-Fold + LIG (4GB VRAM safe)
# ==================================================================================

def run_roberta_engine_local(code_text: str, model_type_name: str,
                              manager: LocalModelManager) -> dict:
    """
    Chạy RoBERTa inference + LIG attribution cho Local GPU Server.

    Giống engine Colab nhưng:
      - Chỉ chạy 1 fold
      - ENGINE_BATCH_SIZE = 1, LIG_BATCH_SIZE = 1
      - Model được move về CPU + GPU cache cleared sau MỖI batch

    Args:
        code_text: Code C++ (đã decode từ base64, đã strip)
        model_type_name: "C++ OOP Model" | "C++ Normal Model"
        manager: LocalModelManager instance (đã load_resources)

    Returns:
        dict: {model_used, final_pred, final_score, total_tokens,
               total_chunks, tokens, attrs, chunks}
    """
    t0 = time.perf_counter()
    gpu_log("info", "engine", "run_roberta_engine_local",
            f"Starting inference: {model_type_name}", code_len=len(code_text))

    code_text = code_text.replace('\r\n', '\n').strip()
    all_tokens = manager.tokenizer.tokenize(code_text)
    total_tokens = len(all_tokens)

    # --- Chunking (Sliding Window) ---
    chunks_info = []
    if total_tokens <= MAX_LEN:
        ids = manager.tokenizer.convert_tokens_to_ids(
            [manager.tokenizer.cls_token] + all_tokens + [manager.tokenizer.sep_token]
        )
        chunks_info.append({"ids": ids, "start_idx": 0, "end_idx": total_tokens})
    else:
        for i in range(0, total_tokens, STRIDE):
            chunk_toks = all_tokens[i: i + MAX_LEN]
            ids = manager.tokenizer.convert_tokens_to_ids(
                [manager.tokenizer.cls_token] + chunk_toks + [manager.tokenizer.sep_token]
            )
            chunks_info.append({"ids": ids, "start_idx": i,
                                 "end_idx": i + len(chunk_toks)})
            if i + MAX_LEN >= total_tokens:
                break

    print(f"📄 [LOCAL GPU] {total_tokens} tokens → {len(chunks_info)} chunk(s).")
    gpu_log("info", "engine", "run_roberta_engine_local",
            f"Tokenized", total_tokens=total_tokens, chunks=len(chunks_info))

    chunk_results_accum = [
        {"prob_sum": 0.0, "attrs_sum": None, "count": 0}
        for _ in chunks_info
    ]

    target_type = "OOP" if "OOP" in model_type_name else "NORMAL"
    # Dùng ensure_type_loaded() thay vì get_models() để tự lazy-load nếu rỗng
    models_in_ram = manager.ensure_type_loaded(target_type)
    if not models_in_ram:
        gpu_log("error", "engine", "run_roberta_engine_local",
                f"No models loaded for {model_type_name} after lazy-load attempt")
        return {"error": (
            f"Không thể load model {model_type_name}. "
            f"Kiểm tra đường dẫn {'LOCAL_MODEL_PATH_OOP' if target_type == 'OOP' else 'LOCAL_MODEL_PATH_NORMAL'} "
            f"trong .env và xem traceback trong terminal Local GPU Server."
        )}

    global_attrs  = np.zeros(total_tokens)
    global_counts = np.zeros(total_tokens)
    total_folds   = len(models_in_ram)

    # --- Fold-level inference ---
    for fold_idx, model_cpu in enumerate(models_in_ram):
        fold_t0 = time.perf_counter()
        print(f"🔄 [LOCAL GPU] Running Fold {fold_idx + 1}/{total_folds}...")

        try:
            # Push lên GPU chỉ trong block này
            model = model_cpu.to(DEVICE)

            def predict_func(inputs, attention_mask=None):
                return model(inputs_embeds=inputs,
                             attention_mask=attention_mask).logits

            lig = LayerIntegratedGradients(predict_func, model.roberta.embeddings)

            # ENGINE_BATCH_SIZE = 1 → an toàn nhất cho 4GB VRAM
            for batch_start in range(0, len(chunks_info), ENGINE_BATCH_SIZE):
                batch_chunks = chunks_info[batch_start: batch_start + ENGINE_BATCH_SIZE]
                max_len_in_batch = max(len(c["ids"]) for c in batch_chunks)

                padded_input_ids = []
                attention_masks  = []
                padded_bases     = []

                for c in batch_chunks:
                    pad_len = max_len_in_batch - len(c["ids"])
                    padded_input_ids.append(
                        c["ids"] + [manager.tokenizer.pad_token_id] * pad_len
                    )
                    attention_masks.append(
                        [1] * len(c["ids"]) + [0] * pad_len
                    )
                    padded_bases.append(
                        [manager.tokenizer.pad_token_id] * max_len_in_batch
                    )

                inp         = torch.tensor(padded_input_ids).to(DEVICE)
                mask_tensor = torch.tensor(attention_masks).to(DEVICE)
                base        = torch.tensor(padded_bases).to(DEVICE)

                # Forward pass
                with torch.no_grad():
                    logits = model(inp, attention_mask=mask_tensor).logits
                    probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                # LIG Attribution
                embed_inp  = model.roberta.embeddings(inp)
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
                    g_idx      = batch_start + j
                    actual_len = len(batch_chunks[j]["ids"])
                    valid_attrs = attrs_np[j][:actual_len]
                    clean_prob  = float(prob)

                    print(f"    ├── Chunk {g_idx + 1}/{len(chunks_info)}: "
                          f"Score {clean_prob:.4f}")
                    chunk_results_accum[g_idx]["prob_sum"] += clean_prob
                    chunk_results_accum[g_idx]["count"]    += 1

                    if chunk_results_accum[g_idx]["attrs_sum"] is None:
                        chunk_results_accum[g_idx]["attrs_sum"] = valid_attrs
                    else:
                        chunk_results_accum[g_idx]["attrs_sum"] += valid_attrs

                # Giải phóng VRAM ngay sau mỗi batch
                del inp, mask_tensor, base, embed_inp, embed_base, logits, attrs
                torch.cuda.empty_cache()
                gc.collect()

            # Đưa model về CPU để giải phóng GPU memory
            model.cpu()
            torch.cuda.empty_cache()
            gc.collect()

            fold_ms = (time.perf_counter() - fold_t0) * 1000
            gpu_log("info", "engine", "run_roberta_engine_local",
                    f"Fold {fold_idx + 1} complete",
                    fold_idx=fold_idx + 1, latency_ms=round(fold_ms, 1))

        except Exception as exc:
            print(f"⚠️  [LOCAL GPU] Error in Fold {fold_idx + 1}: {exc}")
            gpu_log("error", "engine", "run_roberta_engine_local",
                    f"Fold {fold_idx + 1} failed: {exc}")
            try:
                model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass

    # --- Aggregate results ---
    chunk_results_data = []
    weighted_prob_sum  = 0.0
    total_weight       = 0.0

    for i, res in enumerate(chunk_results_accum):
        if res["count"] == 0:
            continue
        avg_prob   = float(res["prob_sum"] / res["count"])
        avg_attrs  = res["attrs_sum"] / res["count"]
        chunk_len  = len(chunks_info[i]["ids"]) - 2
        chunk_label = "AI" if avg_prob >= manager.threshold else "HUMAN"

        chunk_tokens = manager.tokenizer.convert_ids_to_tokens(chunks_info[i]["ids"])
        snippet      = manager.tokenizer.decode(
            chunks_info[i]["ids"], skip_special_tokens=True
        )

        chunk_results_data.append({
            "index":   i + 1,
            "score":   avg_prob,
            "label":   chunk_label,
            "tokens":  chunk_tokens,
            "attrs":   avg_attrs.tolist(),
            "snippet": snippet,
        })

        weighted_prob_sum += avg_prob * chunk_len
        total_weight      += chunk_len

        valid = avg_attrs[1:-1]
        s_idx = chunks_info[i]["start_idx"]
        ll    = min(len(valid), chunks_info[i]["end_idx"] - s_idx)
        global_attrs[s_idx: s_idx + ll]  += valid[:ll]
        global_counts[s_idx: s_idx + ll] += 1

    final_attrs = np.divide(
        global_attrs, global_counts,
        out=np.zeros_like(global_attrs),
        where=global_counts != 0,
    )
    final_avg  = (float(weighted_prob_sum / total_weight)
                  if total_weight > 0 else 0.5)
    final_pred = "AI GENERATED" if final_avg >= manager.threshold else "HUMAN WRITTEN"

    total_ms = (time.perf_counter() - t0) * 1000
    gpu_log("info", "engine", "run_roberta_engine_local",
            f"Complete: {final_pred} ({final_avg:.4f})",
            final_score=final_avg, prediction=final_pred,
            total_tokens=total_tokens, total_chunks=len(chunks_info),
            latency_ms=round(total_ms, 1))

    print(f"✅ [LOCAL GPU] Done: {final_pred} | Score: {final_avg:.4f} | "
          f"{total_ms:.0f}ms")

    return {
        "model_used":    model_type_name,
        "final_pred":    final_pred,
        "final_score":   final_avg,
        "total_tokens":  total_tokens,
        "total_chunks":  len(chunks_info),
        "tokens":        all_tokens,
        "attrs":         final_attrs.tolist(),
        "chunks":        chunk_results_data,
    }
