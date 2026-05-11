"""
feature_extractor.py — Trích xuất 32 đặc trưng tĩnh từ mã nguồn C++

Sử dụng regex + thư viện lizard để đánh giá complexity.
Deep logging cho extraction timing.
"""

import re
import math
import time

import lizard

from .config import colab_log


def strip_metadata_headers(code_text):
    """Xóa các khối header/metadata (ví dụ: JSON metadata) ở đầu file nếu có."""
    code_text = re.sub(r'^\s*\{.*?"code"\s*:\s*".*?"\s*\}\s*', '', code_text, flags=re.DOTALL)
    return code_text.strip()


class CppFeatureExtractorV8:
    """
    Trích xuất 32 đặc trưng tĩnh từ mã nguồn C++.
    Sử dụng regex và thư viện lizard để đánh giá complexity.
    """
    def __init__(self):
        # 32 features (20 được chọn cuối cùng bởi LightGBM)
        self.feature_names = [
            'empty_line_ratio', 'avg_line_length', 'max_line_length', 'tab_vs_space_ratio',
            'brace_style_consistency', 'avg_identifier_length', 'identifier_length_variance',
            'single_char_var_ratio', 'unique_identifier_ratio', 'keyword_to_identifier_ratio',
            'avg_cyclomatic_complexity', 'num_functions', 'avg_function_loc',
            'halstead_volume', 'halstead_difficulty', 'halstead_effort', 'halstead_bugs',
            'maintainability_index', 'code_to_comment_ratio', 'max_nesting_depth',
            'total_includes', 'has_bits_stdc', 'macro_count', 'modern_cpp_ratio',
            'const_usage_ratio', 'has_fast_io', 'newline_style_ratio',
            'shannon_entropy', 'bigram_entropy', 'whitespace_entropy',
            'comment_ratio', 'trailing_space_ratio'
        ]

    def extract(self, code_text):
        """Trả về dictionary chứa 32 features."""
        t0 = time.perf_counter()
        code_text = strip_metadata_headers(code_text)
        features = {}

        lines = code_text.split('\n')
        num_lines = len(lines) if len(lines) > 0 else 1

        # 1. Layout & Format (1-5)
        empty_lines = sum(1 for line in lines if line.strip() == '')
        features['empty_line_ratio'] = empty_lines / num_lines

        line_lengths = [len(line) for line in lines]
        features['avg_line_length'] = sum(line_lengths) / num_lines
        features['max_line_length'] = max(line_lengths) if line_lengths else 0

        tabs = code_text.count('\t')
        spaces = code_text.count(' ')
        features['tab_vs_space_ratio'] = tabs / (spaces + 1)

        # brace style: K&R vs Allman
        kr_braces = len(re.findall(r'\{\s*$', code_text, re.MULTILINE))
        allman_braces = len(re.findall(r'^\s*\{', code_text, re.MULTILINE))
        total_braces = kr_braces + allman_braces
        features['brace_style_consistency'] = max(kr_braces, allman_braces) / total_braces if total_braces > 0 else 1.0

        # 2. Naming (6-10)
        identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code_text)
        keywords = {
            'int', 'float', 'double', 'char', 'void', 'if', 'else', 'for',
            'while', 'return', 'class', 'struct', 'public', 'private',
            'protected', 'virtual', 'template', 'namespace', 'using', 'std',
            'cout', 'cin', 'endl', 'include',
        }
        vars_only = [w for w in identifiers if w not in keywords]

        var_lengths = [len(v) for v in vars_only]
        avg_id_len = sum(var_lengths) / len(var_lengths) if var_lengths else 0
        features['avg_identifier_length'] = avg_id_len

        variance = sum((x - avg_id_len)**2 for x in var_lengths) / len(var_lengths) if var_lengths else 0
        features['identifier_length_variance'] = variance

        single_chars = sum(1 for v in vars_only if len(v) == 1)
        features['single_char_var_ratio'] = single_chars / len(vars_only) if vars_only else 0

        unique_vars = len(set(vars_only))
        features['unique_identifier_ratio'] = unique_vars / len(vars_only) if vars_only else 0

        features['keyword_to_identifier_ratio'] = (
            len([w for w in identifiers if w in keywords]) / len(identifiers)
            if identifiers else 0
        )

        # 3. Complexity & Halstead via lizard (11-18)
        try:
            lizard_res = lizard.analyze_file.analyze_source_code("temp.cpp", code_text)
        except Exception:
            lizard_res = type("obj", (object,), {"function_list": []})()

        if lizard_res.function_list:
            features['avg_cyclomatic_complexity'] = (
                sum(f.cyclomatic_complexity for f in lizard_res.function_list)
                / len(lizard_res.function_list)
            )
            features['num_functions'] = len(lizard_res.function_list)
            features['avg_function_loc'] = (
                sum(f.length for f in lizard_res.function_list)
                / len(lizard_res.function_list)
            )
        else:
            features['avg_cyclomatic_complexity'] = 1.0
            features['num_functions'] = 0.0
            features['avg_function_loc'] = 0.0

        # Halstead Approximation
        operators = re.findall(r'[\+\-\*/%=\<\>\!\&\|\^\~]+', code_text)
        operands = vars_only + re.findall(r'\b\d+\b', code_text)

        N1 = len(operators)
        N2 = len(operands)
        n1 = len(set(operators))
        n2 = len(set(operands))

        vocabulary = n1 + n2
        length = N1 + N2

        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = volume * difficulty

        features['halstead_volume'] = volume
        features['halstead_difficulty'] = difficulty
        features['halstead_effort'] = effort
        features['halstead_bugs'] = volume / 3000

        # Maintainability Index (MI) approximation
        mi = (
            171
            - 5.2 * math.log(volume if volume > 0 else 1)
            - 0.23 * features['avg_cyclomatic_complexity']
            - 16.2 * math.log(num_lines if num_lines > 0 else 1)
        )
        features['maintainability_index'] = max(0, mi * 100 / 171)

        # 4. Style & Includes (19-27)
        comments = re.findall(r'//.*|/\*.*?\*/', code_text, re.DOTALL)
        comment_loc = sum(len(c.split('\n')) for c in comments)

        features['comment_ratio'] = comment_loc / num_lines
        features['code_to_comment_ratio'] = (
            (num_lines - comment_loc) / comment_loc if comment_loc > 0 else num_lines
        )

        # Max nesting depth approximation
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        features['max_nesting_depth'] = max(indents) / 4 if indents else 0

        includes = re.findall(r'#include', code_text)
        features['total_includes'] = len(includes)
        features['has_bits_stdc'] = 1.0 if 'bits/stdc++.h' in code_text else 0.0
        features['macro_count'] = len(re.findall(r'#define', code_text))

        features['modern_cpp_ratio'] = (
            len(re.findall(r'auto\s+|constexpr\s+|nullptr', code_text)) / len(identifiers)
            if identifiers else 0
        )
        features['const_usage_ratio'] = (
            len(re.findall(r'\bconst\b', code_text)) / len(vars_only) if vars_only else 0
        )

        fast_io = 1.0 if 'ios_base::sync_with_stdio' in code_text or 'cin.tie' in code_text else 0.0
        features['has_fast_io'] = fast_io

        newlines = code_text.count('\n')
        endls = code_text.count('endl')
        features['newline_style_ratio'] = newlines / (endls + 1)

        # 5. Entropy (28-32)
        def calc_entropy(text):
            if not text:
                return 0
            prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
            return -sum(p * math.log2(p) for p in prob if p > 0)

        features['shannon_entropy'] = calc_entropy(code_text)

        bigrams = [code_text[i:i+2] for i in range(len(code_text)-1)]
        features['bigram_entropy'] = calc_entropy(bigrams)

        whitespace_only = re.sub(r'\S', '', code_text)
        features['whitespace_entropy'] = calc_entropy(whitespace_only)

        trailing_spaces = sum(1 for line in lines if line != line.rstrip())
        features['trailing_space_ratio'] = trailing_spaces / num_lines

        # Log extraction timing
        latency = (time.perf_counter() - t0) * 1000
        colab_log("info", "feature_extractor", "extract",
                  f"Extracted {len(self.feature_names)} features",
                  num_features=len(self.feature_names),
                  num_lines=num_lines,
                  latency_ms=round(latency, 2))

        return {k: features.get(k, 0.0) for k in self.feature_names}
