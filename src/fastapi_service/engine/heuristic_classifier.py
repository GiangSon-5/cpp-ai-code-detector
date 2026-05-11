"""
engine/heuristic_classifier.py — OOP/Normal heuristic classification.

Ported from extract_1.py heuristic_classify().
Checks for OOP keywords (class, virtual, public, protected, etc.)
and returns a confidence score 0.0-1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.shared.logger import AppLogger


@dataclass
class HeuristicResult:
    classification: str   # "OOP" | "NORMAL"
    score: float          # 0.0 = definitely NORMAL, 1.0 = definitely OOP
    reasons: list[str]


# OOP indicators with weights
_OOP_PATTERNS: list[tuple[str, float, str]] = [
    (r"\bclass\s+\w+", 0.25, "class declaration"),
    (r"\bvirtual\b", 0.15, "virtual keyword"),
    (r"\bpublic\s*:", 0.10, "public access specifier"),
    (r"\bprotected\s*:", 0.08, "protected access specifier"),
    (r"\bprivate\s*:", 0.08, "private access specifier"),
    (r"\binherit|:\s*public\s+\w+", 0.12, "inheritance"),
    (r"\boverride\b", 0.08, "override keyword"),
    (r"\btemplate\s*<", 0.06, "template"),
    (r"\boperator\s*[+\-*/=<>!]+", 0.05, "operator overload"),
    (r"\bnamespace\b", 0.03, "namespace"),
]


@AppLogger.log_function(module="heuristic_classifier")
def heuristic_classify(code: str) -> HeuristicResult:
    """Classify C++ code as OOP or NORMAL based on syntax heuristics.

    Returns:
        HeuristicResult with classification, score, and reasons.
    """
    total_score = 0.0
    reasons: list[str] = []

    for pattern, weight, desc in _OOP_PATTERNS:
        matches = re.findall(pattern, code, re.MULTILINE)
        if matches:
            total_score += weight
            reasons.append(f"{desc} (×{len(matches)})")

    # Clamp to [0, 1]
    total_score = min(1.0, total_score)

    # Threshold: score >= 0.25 → OOP
    classification = "OOP" if total_score >= 0.25 else "NORMAL"

    return HeuristicResult(
        classification=classification,
        score=round(total_score, 4),
        reasons=reasons,
    )
