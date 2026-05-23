import re
import math
import numpy as np
from collections import Counter

try:
    import lizard
except ImportError:
    lizard = None

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

        # Regex hỗ trợ tính Halstead (Toán tử & Toán hạng)
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
        if marker_count <= 0:
            return 0.0
        if marker_count == 1:
            return 0.65
        if marker_count == 2:
            return 0.9
        return 1.0

    def calculate_halstead_metrics(self, pure_code, identifiers):
        # N1, n1: Operators (Toán tử và từ khóa)
        ops = self.operators_pattern.findall(pure_code)
        kw_found = [w for w in re.findall(r'\b[a-zA-Z_]\w*\b', pure_code) if w in self.keywords]
        all_operators = ops + kw_found

        N1 = len(all_operators)
        n1 = len(set(all_operators))

        # N2, n2: Operands (Biến, hàm, chuỗi, số)
        strings = self.string_pattern.findall(pure_code)
        numbers = re.findall(r'\b\d+(\.\d+)?\b', pure_code)
        all_operands = identifiers + strings + numbers

        N2 = len(all_operands)
        n2 = len(set(all_operands))

        # Tính toán Halstead
        n = n1 + n2 # Vocabulary
        N = N1 + N2 # Length

        Volume = N * math.log2(n) if n > 0 else 0
        Difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        Effort = Difficulty * Volume
        Bugs = Volume / 3000

        return Volume, Difficulty, Effort, Bugs

    def analyze_structure_fallback(self, code_raw):
        function_pattern = re.compile(
            r'\b[a-zA-Z_][\w:<>*&\s]*\s+[a-zA-Z_]\w*\s*\([^;{}]*\)\s*\{',
            re.MULTILINE
        )
        functions = list(function_pattern.finditer(code_raw))
        if not functions:
            return 1.0, 0, 0.0

        complexity_tokens = re.compile(r'\b(if|for|while|case|catch)\b|&&|\|\||\?')
        line_positions = [0]
        for idx, char in enumerate(code_raw):
            if char == '\n':
                line_positions.append(idx + 1)

        function_locs = []
        function_complexities = []
        for match in functions:
            start_idx = match.start()
            next_start = len(code_raw)
            for other in functions:
                if other.start() > start_idx:
                    next_start = min(next_start, other.start())

            chunk = code_raw[start_idx:next_start]
            loc = max(1, chunk.count('\n') + 1)
            cc = 1 + len(complexity_tokens.findall(chunk))
            function_locs.append(loc)
            function_complexities.append(cc)

        return (
            float(np.mean(function_complexities)),
            len(functions),
            float(np.mean(function_locs)),
        )

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
        features['brace_style_consistency'] = abs(allman_ratio - 0.5) * 2 # 0: mixed, 1: consistent

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

        # --- [C] STRUCTURAL COMPLEXITY (Halstead + Cyclomatic) ---
        if lizard is not None:
            analysis = lizard.analyze_file.analyze_source_code("test.cpp", code_raw)
            if analysis.function_list:
                cc_list = [f.cyclomatic_complexity for f in analysis.function_list]
                features['avg_cyclomatic_complexity'] = np.mean(cc_list)
                features['num_functions'] = len(analysis.function_list)
                features['avg_function_loc'] = np.mean([f.end_line - f.start_line for f in analysis.function_list])
            else:
                features['avg_cyclomatic_complexity'] = 1.0
                features['num_functions'] = 0
                features['avg_function_loc'] = 0
        else:
            avg_cc, num_funcs, avg_loc = self.analyze_structure_fallback(code_raw)
            features['avg_cyclomatic_complexity'] = avg_cc
            features['num_functions'] = num_funcs
            features['avg_function_loc'] = avg_loc

        # Tính toán Halstead Metrics
        V, D, E, B = self.calculate_halstead_metrics(pure_code, custom_ids)
        features['halstead_volume'] = V
        features['halstead_difficulty'] = D
        features['halstead_effort'] = E
        features['halstead_bugs'] = B

        # Maintainability Index
        MI = 171 - 5.2 * math.log(max(1, V)) - 0.23 * features['avg_cyclomatic_complexity'] - 16.2 * math.log(max(1, total_loc))
        features['maintainability_index'] = max(0, MI)
        features['code_to_comment_ratio'] = total_loc / max(1, len(comments_text.split('\n')))

        # Độ sâu lồng nhau (Nesting depth heuristic)
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

        # Whitespace pattern entropy
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
