"""
shared/hashing.py — Deterministic code hashing utility.

Used across all layers to generate the code_hash (SHA-256) that
serves as the primary key linking Bronze → Silver → Gold.
"""

from __future__ import annotations

import hashlib

from src.shared.logger import AppLogger


@AppLogger.log_function(module="hashing")
def compute_code_hash(raw_code: str) -> str:
    """Return a 64-char hex SHA-256 digest of the normalised source code.

    Normalisation:
        1. Strip leading/trailing whitespace.
        2. Normalise line endings to \\n.
        3. Encode as UTF-8.

    This ensures the same logical code always maps to the same hash,
    regardless of OS line-ending differences.
    """
    normalised = raw_code.strip().replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
