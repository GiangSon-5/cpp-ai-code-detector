import os
import json
import re
import random
import math
import warnings
import numpy as np
import pandas as pd
import lizard
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel

warnings.filterwarnings('ignore')

# CPU Core Control
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, CPU_COUNT // 2)
print(f"[SYSTEM] CPU cores detected: {CPU_COUNT}")
print(f"[SYSTEM] Cap for parallel execution: {N_JOBS} cores")

# Paths
WORKSPACE = r"C:\Users\2\Desktop\WORKSPACE\LVTN-Main"
TRAIN_DATASET_PATH = os.path.join(WORKSPACE, "data_splits", "train_set", "train_data_with_oop.jsonl")
SAVE_DIR = os.path.join(WORKSPACE, "retrain ML models", "saved_models_44_features_improved")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# DEFINE 44-FEATURE EXTRACTOR
# -------------------------------------------------------------------------
class CppFeatureExtractorV8:
    def __init__(self):
        self.string_pattern = re.compile(r'".*?(?<!\\)"|\'.*?(?<!\\)\'')
        self.comment_line_pattern = re.compile(r'//.*')
        self.comment_block_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)

        self.include_pattern = re.compile(r'#include\s*[<"].*[>"]')
        self.macro_pattern = re.compile(r'#define\s+')
        self.bits_stdc_pattern = re.compile(r'bits/stdc\+\+\.h')
        self.fast_io_pattern = re.compile(r'ios_base::sync_with_stdio|cin\.tie')

        self.modern_cpp_pattern = re.compile(r'\b(auto|nullptr|unique_ptr|shared_ptr|constexpr|lambda)\b')
        self.const_pattern = re.compile(r'\b(const|constexpr)\b')
        self.exception_pattern = re.compile(r'\b(try|catch|throw)\b')
        self.single_char_var_pattern = re.compile(r'\b[a-zA-Z]\b')
        
        self.emote_tokens = [
            ":)", ":-)", ":D", ":-D", ";)", ";-)", "<3",
            "^_^", "T_T", "xD", "XD", "UwU", "uwu", "O_O", ":3",
            "¯\\_(ツ)_/¯"
        ]

        self.operators_pattern = re.compile(r'(\+|-|\*|/|%|=|==|!=|<|>|<=|>=|&&|\|\||!|&|\||\^|~|<<|>>|\+\+|--|->|\.|::|\?|:)')
        self.keywords = {'int', 'void', 'if', 'else', 'while', 'for', 'return', 'class', 'public', 'private', 'struct', 'bool', 'char', 'float', 'double', 'std', 'cout', 'cin', 'endl', 'break', 'continue'}

        # New OOP and advanced styling patterns
        self.class_pattern = re.compile(r'\bclass\s+[a-zA-Z_]\w*')
        self.struct_pattern = re.compile(r'\bstruct\s+[a-zA-Z_]\w*')
        self.inheritance_pattern = re.compile(r'\bclass\s+[a-zA-Z_]\w*\s*:\s*(public|private|protected)\b')
        self.access_specifier_pattern = re.compile(r'\b(public|private|protected)\s*:')
        self.virtual_override_pattern = re.compile(r'\b(virtual|override)\b')
        self.using_std_pattern = re.compile(r'\busing\s+namespace\s+std\s*;')
        self.try_catch_pattern = re.compile(r'\b(try|catch)\b')
        self.raw_pointer_pattern = re.compile(r'\b(int|float|double|char|void|[a-zA-Z_]\w*)\s*\*+\s*[a-zA-Z_]\w*')
        self.getter_setter_pattern = re.compile(r'\b(get|set)[A-Z]\w*\b')
        self.cpp_cast_pattern = re.compile(r'\b(static_cast|dynamic_cast|reinterpret_cast|const_cast)\b')

    def calculate_entropy(self, text):
        if not text: return 0.0
        prob = [float(c) / len(text) for c in dict(Counter(text)).values()]
        return -sum(p * math.log2(p) for p in prob)

    def calculate_bigram_entropy(self, text):
        if len(text) < 2: return 0.0
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        prob = [float(c) / len(bigrams) for c in dict(Counter(bigrams)).values()]
        return -sum(p * math.log2(p) for p in prob)

    def count_emoji_markers(self, text):
        count = 0
        for char in text:
            codepoint = ord(char)
            if (0x1F300 <= codepoint <= 0x1FAFF) or (0x2600 <= codepoint <= 0x27BF):
                count += 1
        for token in self.emote_tokens:
            count += text.count(token)
        return count

    def score_emoji_markers(self, marker_count):
        if marker_count <= 0: return 0.0
        if marker_count == 1: return 0.65
        if marker_count == 2: return 0.9
        return 1.0

    def calculate_halstead_metrics(self, pure_code, identifiers):
        ops = self.operators_pattern.findall(pure_code)
        kw_found = [w for w in re.findall(r'\b[a-zA-Z_]\w*\b', pure_code) if w in self.keywords]
        all_operators = ops + kw_found

        N1 = len(all_operators)
        n1 = len(set(all_operators))

        strings = self.string_pattern.findall(pure_code)
        numbers = re.findall(r'\b\d+(\.\d+)?\b', pure_code)
        all_operands = identifiers + strings + numbers

        N2 = len(all_operands)
        n2 = len(set(all_operands))

        n = n1 + n2
        N = N1 + N2

        Volume = N * math.log2(n) if n > 0 else 0
        Difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        Effort = Difficulty * Volume
        Bugs = Volume / 3000

        return Volume, Difficulty, Effort, Bugs

    def extract(self, code_raw):
        features = {}
        lines = code_raw.split('\n')
        total_chars = len(code_raw)
        pure_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]
        total_loc = len(pure_lines) if len(pure_lines) > 0 else 1

        line_comments = self.comment_line_pattern.findall(code_raw)
        block_comments = self.comment_block_pattern.findall(code_raw)
        comments_text = "\n".join(line_comments + block_comments)
        string_literals = self.string_pattern.findall(code_raw)

        pure_code = self.string_pattern.sub('', code_raw)
        pure_code = self.comment_block_pattern.sub('', pure_code)
        pure_code = self.comment_line_pattern.sub('', pure_code)

        # --- [A] LAYOUT & FORMATTING ---
        features['comment_ratio'] = len(comments_text) / total_chars if total_chars > 0 else 0
        features['empty_line_ratio'] = sum(1 for line in lines if not line.strip()) / max(1, len(lines))
        marker_text = comments_text + "\n" + "\n".join(string_literals)
        emoji_marker_count = self.count_emoji_markers(marker_text)
        features['emoji_marker_score'] = self.score_emoji_markers(emoji_marker_count)

        line_lengths = [len(l.strip()) for l in pure_lines]
        features['avg_line_length'] = np.mean(line_lengths) if line_lengths else 0
        features['max_line_length'] = np.max(line_lengths) if line_lengths else 0

        spaces = code_raw.count(' ')
        tabs = code_raw.count('\t')
        features['tab_vs_space_ratio'] = tabs / (spaces + tabs) if (spaces + tabs) > 0 else 0
        features['trailing_space_ratio'] = sum(1 for l in lines if l.endswith(' ') or l.endswith('\t')) / max(1, len(lines))

        kr_brace = len(re.findall(r'\S\s*\{', pure_code))
        allman_brace = len(re.findall(r'^\s*\{', pure_code, re.MULTILINE))
        total_braces = kr_brace + allman_brace
        allman_ratio = allman_brace / total_braces if total_braces > 0 else 0.5
        features['brace_style_consistency'] = abs(allman_ratio - 0.5) * 2

        # --- [B] NAMING CONVENTIONS ---
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', pure_code)
        custom_ids = [w for w in identifiers if w not in self.keywords]

        if custom_ids:
            id_lengths = [len(w) for w in custom_ids]
            features['avg_identifier_length'] = np.mean(id_lengths)
            features['identifier_length_variance'] = np.var(id_lengths)
            features['single_char_var_ratio'] = len(self.single_char_var_pattern.findall(pure_code)) / len(custom_ids)
            features['unique_identifier_ratio'] = len(set(custom_ids)) / len(custom_ids)
        else:
            features['avg_identifier_length'] = 0
            features['identifier_length_variance'] = 0
            features['single_char_var_ratio'] = 0
            features['unique_identifier_ratio'] = 0

        features['keyword_to_identifier_ratio'] = len([w for w in identifiers if w in self.keywords]) / max(1, len(custom_ids))

        # --- [C] STRUCTURAL COMPLEXITY ---
        try:
            analysis = lizard.analyze_source_code("test.cpp", code_raw)
            if analysis.function_list:
                cc_list = [f.cyclomatic_complexity for f in analysis.function_list]
                features['avg_cyclomatic_complexity'] = np.mean(cc_list)
                features['num_functions'] = len(analysis.function_list)
                features['avg_function_loc'] = np.mean([f.end_line - f.start_line for f in analysis.function_list])
            else:
                features['avg_cyclomatic_complexity'] = 1.0
                features['num_functions'] = 0
                features['avg_function_loc'] = 0
        except Exception:
            features['avg_cyclomatic_complexity'] = 1.0
            features['num_functions'] = 0
            features['avg_function_loc'] = 0

        V, D, E, B = self.calculate_halstead_metrics(pure_code, custom_ids)
        features['halstead_volume'] = V
        features['halstead_difficulty'] = D
        features['halstead_effort'] = E
        features['halstead_bugs'] = B

        MI = 171 - 5.2 * math.log(max(1, V)) - 0.23 * features['avg_cyclomatic_complexity'] - 16.2 * math.log(max(1, total_loc))
        features['maintainability_index'] = max(0, MI)
        features['code_to_comment_ratio'] = total_loc / max(1, len(comments_text.split('\n')))

        depth, max_depth = 0, 0
        for char in pure_code:
            if char == '{': depth += 1; max_depth = max(max_depth, depth)
            elif char == '}': depth = max(0, depth - 1)
        features['max_nesting_depth'] = max_depth

        # --- [D] CODING HABITS & IDIOMS ---
        features['total_includes'] = len(self.include_pattern.findall(code_raw))
        features['has_bits_stdc'] = 1 if self.bits_stdc_pattern.search(code_raw) else 0
        features['macro_count'] = len(self.macro_pattern.findall(code_raw))

        total_words = len(re.findall(r'\b\w+\b', pure_code))
        features['modern_cpp_ratio'] = len(self.modern_cpp_pattern.findall(pure_code)) / max(1, total_words)
        features['const_usage_ratio'] = len(self.const_pattern.findall(pure_code)) / max(1, total_words)

        features['has_fast_io'] = 1 if self.fast_io_pattern.search(pure_code) else 0

        count_endl = pure_code.count('endl')
        count_n = pure_code.count('\\n')
        features['newline_style_ratio'] = count_n / max(1, (count_n + count_endl))

        # --- [E] INFORMATION THEORY ---
        features['shannon_entropy'] = self.calculate_entropy(pure_code)
        features['bigram_entropy'] = self.calculate_bigram_entropy(pure_code)

        spaces_pattern = "".join(['S' if c == ' ' else 'T' if c == '\t' else 'N' if c == '\n' else '' for c in code_raw])
        features['whitespace_entropy'] = self.calculate_entropy(spaces_pattern)

        # --- [F] NEW OOP & ADVANCED CODING HABITS ---
        class_cnt = len(self.class_pattern.findall(pure_code))
        features['class_count'] = class_cnt
        features['struct_count'] = len(self.struct_pattern.findall(pure_code))
        features['has_inheritance'] = 1 if self.inheritance_pattern.search(pure_code) else 0
        
        access_spec_cnt = len(self.access_specifier_pattern.findall(pure_code))
        features['access_specifier_ratio'] = access_spec_cnt / max(1, class_cnt)
        
        virtual_override_cnt = len(self.virtual_override_pattern.findall(pure_code))
        features['virtual_override_ratio'] = virtual_override_cnt / max(1, features['num_functions'])
        
        features['using_std_ratio'] = 1 if self.using_std_pattern.search(code_raw) else 0
        
        try_catch_cnt = len(self.try_catch_pattern.findall(pure_code))
        features['try_catch_ratio'] = try_catch_cnt / max(1, total_loc)
        
        raw_ptr_cnt = len(self.raw_pointer_pattern.findall(pure_code))
        features['raw_pointer_ratio'] = raw_ptr_cnt / max(1, total_words)
        
        getter_setter_cnt = len(self.getter_setter_pattern.findall(pure_code))
        features['getter_setter_ratio'] = getter_setter_cnt / max(1, len(custom_ids) if custom_ids else 1)
        
        std_prefix_cnt = pure_code.count('std::')
        features['std_prefix_ratio'] = std_prefix_cnt / max(1, total_words)
        
        cpp_cast_cnt = len(self.cpp_cast_pattern.findall(pure_code))
        features['cpp_cast_ratio'] = cpp_cast_cnt / max(1, total_words)

        return features

def strip_metadata_headers(code_raw):
    lines = code_raw.split('\n')
    cleaned_lines = []
    header_passed = False
    for line in lines:
        if not header_passed:
            if line.strip().startswith('//') and any(kw in line.upper() for kw in ['DATASET', 'MODEL', 'GEMINI', 'GPT', 'CATEGORY']):
                continue
            if line.strip() != '':
                header_passed = True
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

# -------------------------------------------------------------------------
# EXECUTION FLOW
# -------------------------------------------------------------------------
def main():
    # STEP 1: LOAD NEW DATASET
    print("\n[STEP 1] LOADING NEW BALANCED MERGED TRAIN SET...")
    with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    print(f"Loaded {len(train_data)} records for training.")

    # STEP 2: EXTRACT FEATURES
    print("\n[STEP 2] EXTRACTING 44 FEATURES FROM DATASET...")
    extractor = CppFeatureExtractorV8()
    extracted_data = []
    
    for item in tqdm(train_data, desc="Extracting 44 C++ features"):
        clean_code = strip_metadata_headers(item['code'])
        features = extractor.extract(clean_code)
        features['label'] = 1 if item['label'] == 'AI' else 0
        extracted_data.append(features)
        
    df_all = pd.DataFrame(extracted_data).fillna(0)
    
    # Train/Test splits (aligned with original logic)
    X = df_all.drop(columns=['label'])
    y = df_all['label']
    
    # Save the order of features to align scaler during inference
    feature_order = list(X.columns)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"Feature extraction completed. Base features count: {X_train.shape[1]}")

    # STEP 3: SAVE BERT TEST SET (For hybrid fusion pipeline compatibility)
    print("\n[STEP 3] SAVING TEST SET FOR BERT...")
    test_indices = X_test.index
    test_records = [train_data[i] for i in test_indices]
    
    test_bert_path = os.path.join(SAVE_DIR, 'test_set_for_bert.jsonl')
    with open(test_bert_path, 'w', encoding='utf-8') as f:
        for record in test_records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved {len(test_records)} test samples to {test_bert_path}")

    # STEP 4: FEATURE SELECTION PIPELINE
    print("\n[STEP 4] RUNNING 3-STAGE FEATURE SELECTION...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # --- Stage 1: Correlation Filter ---
    corr_matrix = X_train_scaled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
    X_train_t1 = X_train_scaled.drop(columns=to_drop_corr)
    print(f"  Stage 1: Dropped {len(to_drop_corr)} multi-collinear features (r > 0.85). Remaining: {X_train_t1.shape[1]}")
    
    # --- Stage 2: Select K-Best (Mutual Information) ---
    selector_kbest = SelectKBest(score_func=mutual_info_classif, k=min(20, X_train_t1.shape[1]))
    X_train_t2_arr = selector_kbest.fit_transform(X_train_t1, y_train)
    selected_kbest_cols = X_train_t1.columns[selector_kbest.get_support()]
    X_train_t2 = pd.DataFrame(X_train_t2_arr, columns=selected_kbest_cols)
    print(f"  Stage 2: Kept top {len(selected_kbest_cols)} features using Mutual Information.")
    
    # --- Stage 3: Lasso L1 Regularization ---
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, C=0.5)
    selector_l1 = SelectFromModel(lasso)
    selector_l1.fit(X_train_t2, y_train)
    final_features = X_train_t2.columns[selector_l1.get_support()]
    print(f"  Stage 3: Chose final {len(final_features)} most important features using Lasso L1.")
    print(f"  Selected Features: {list(final_features)}")
    
    X_train_final = X_train_scaled[final_features]
    X_val_final = X_val_scaled[final_features]
    X_test_final = X_test_scaled[final_features]

    # STEP 5: MODEL TRAINING
    print("\n[STEP 5] TRAINING REGULATED ML MODELS...")
    imbalance_ratio = float((y_train == 0).sum()) / (y_train == 1).sum()
    
    models = {
        "XGBoost_Regulated": xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42,
            scale_pos_weight=imbalance_ratio,
            max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, n_estimators=300, early_stopping_rounds=30,
            n_jobs=N_JOBS
        ),
        "LightGBM_Regulated": lgb.LGBMClassifier(
            random_state=42, verbose=-1, class_weight='balanced',
            max_depth=5, num_leaves=20, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, n_estimators=300, n_jobs=N_JOBS
        ),
        "RandomForest_Regulated": RandomForestClassifier(
            random_state=42, class_weight='balanced',
            max_depth=6, min_samples_split=10, min_samples_leaf=5, n_estimators=300,
            n_jobs=N_JOBS
        )
    }
    
    best_val_score = 0
    best_model_name = ""
    final_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "XGBoost_Regulated":
            model.fit(X_train_final, y_train, eval_set=[(X_val_final, y_val)], verbose=False)
        elif name == "LightGBM_Regulated":
            model.fit(X_train_final, y_train, eval_set=[(X_val_final, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        else:
            model.fit(X_train_final, y_train)
            
        y_val_pred = model.predict(X_val_final)
        val_f1 = f1_score(y_val, y_val_pred)
        train_acc = accuracy_score(y_train, model.predict(X_train_final))
        print(f"  [{name}] Train Acc: {train_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_score:
            best_val_score = val_f1
            best_model_name = name
            final_model = model
            
    print(f"\n=> WINNING MODEL (based on Val F1): {best_model_name}")

    # STEP 6: EVALUATE ON TEST SET
    print("\n[STEP 6] EVALUATING ON TEST SET (20%)...")
    y_train_pred = final_model.predict(X_train_final)
    y_test_pred  = final_model.predict(X_test_final)
    
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test  = accuracy_score(y_test, y_test_pred)
    
    print(f"Accuracy Train Set: {acc_train*100:.2f}%")
    print(f"Accuracy Test Set : {acc_test*100:.2f}%")
    print(classification_report(y_test, y_test_pred, target_names=["Human (0)", "AI (1)"]))

    # STEP 7: SERIALIZE ASSETS TO saved_models_44_features_improved
    print("\n[STEP 7] SERIALIZING ASSETS TO saved_models_44_features_improved...")
    
    # Save Feature Order inside Scaler for downstream alignment
    scaler.feature_names_in_ = np.array(feature_order, dtype=object)
    
    # 1. Models
    joblib.dump(models["LightGBM_Regulated"], os.path.join(SAVE_DIR, "LightGBM_Regulated.pkl"))
    joblib.dump(models["XGBoost_Regulated"], os.path.join(SAVE_DIR, "XGBoost_Regulated.pkl"))
    joblib.dump(models["RandomForest_Regulated"], os.path.join(SAVE_DIR, "RandomForest_Regulated.pkl"))
    
    # 2. Scaler
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
    
    # 3. Features
    joblib.dump(final_features, os.path.join(SAVE_DIR, "feature_names.pkl"))
    joblib.dump(final_features, os.path.join(SAVE_DIR, "final_features.pkl"))
    print("Saved all models, scaler, and features.")

    # STEP 8: GENERATE SHAP AND PREDICTIONS FOR FUSION
    print("\n[STEP 8] GENERATING SHAP AND PREDICTIONS FOR HYBRID FUSION...")
    lgbm_model = models["LightGBM_Regulated"]
    
    # Process test records features
    test_records_features = []
    for item in tqdm(test_records, desc="Processing fusion test features"):
        clean_code = strip_metadata_headers(item['code'])
        features = extractor.extract(clean_code)
        features['label'] = 1 if item['label'] == 'AI' else 0
        features['sample_id'] = item.get('file_name', item.get('relative_path', 'unknown_id'))
        test_records_features.append(features)
        
    df_test_rec = pd.DataFrame(test_records_features).fillna(0)
    sample_ids = df_test_rec['sample_id'].values
    y_true = df_test_rec['label'].values
    X_test_raw = df_test_rec.drop(columns=['label', 'sample_id'])
    
    # Align columns
    X_test_raw_aligned = X_test_raw.reindex(columns=feature_order, fill_value=0)
    
    # Scale and filter
    X_test_scaled_all = scaler.transform(X_test_raw_aligned)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_all, columns=feature_order)
    X_test_final_infer = X_test_scaled_df[final_features]
    
    # Probs & SHAP values
    lgbm_probs = lgbm_model.predict_proba(X_test_final_infer)[:, 1]
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_raw = explainer.shap_values(X_test_final_infer)
    shap_values = shap_raw[1] if isinstance(shap_raw, list) else shap_raw
    
    # Save
    np.save(os.path.join(SAVE_DIR, 'lgbm_probs_test.npy'), lgbm_probs)
    np.save(os.path.join(SAVE_DIR, 'lgbm_shap_test.npy'), shap_values)
    np.save(os.path.join(SAVE_DIR, 'y_true_test.npy'), y_true)
    np.save(os.path.join(SAVE_DIR, 'sample_ids_test.npy'), sample_ids, allow_pickle=True)
    
    print("ALL TRAINING AND ASSET EXPORT STEPS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
