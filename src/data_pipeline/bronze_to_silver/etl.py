"""
bronze_to_silver/etl.py — ETL pipeline: Bronze → Silver (ML + DL).

Reads raw code from Bronze layer (S3/DB), extracts features,
and writes Silver parquet to DagsHub S3.

Deep logging 2-session on all operations.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.shared.logger import AppLogger
from src.shared.s3_client import DagsHubS3Client

logger = AppLogger()


class BronzeToSilverETL:
    """ETL pipeline from Bronze raw code to Silver feature layer."""

    def __init__(self):
        self._s3 = DagsHubS3Client()

    @AppLogger.log_function(module="bronze_to_silver")
    def extract_ml_features(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Extract 32 ML features from a batch of Bronze records.

        Args:
            records: list of dicts with keys 'code_hash', 'raw_code', optional 'label'

        Returns:
            DataFrame with 32 feature columns + code_hash + label
        """
        from src.colab_runtime.scripts.feature_extractor import CppFeatureExtractorV8

        extractor = CppFeatureExtractorV8()
        rows = []

        for rec in records:
            code_hash = rec.get("code_hash", "")
            raw_code = rec.get("raw_code", "")
            label = rec.get("label")

            try:
                features = extractor.extract(raw_code)
                features["code_hash"] = code_hash
                features["label"] = label
                rows.append(features)
            except Exception as exc:
                logger.warning(
                    module="bronze_to_silver",
                    function="extract_ml_features",
                    message=f"Feature extraction failed for {code_hash[:12]}: {exc}",
                )
                # Return zero features for malformed code
                zero_features = {name: 0.0 for name in extractor.feature_names}
                zero_features["code_hash"] = code_hash
                zero_features["label"] = label
                rows.append(zero_features)

        return pd.DataFrame(rows)

    @AppLogger.log_function(module="bronze_to_silver")
    def extract_dl_tokens(self, records: list[dict[str, Any]], max_length: int = 512) -> pd.DataFrame:
        """Tokenize Bronze records for DL model input.

        Args:
            records: list of dicts with 'code_hash' and 'raw_code'
            max_length: max token length

        Returns:
            DataFrame with input_ids[512], attention_mask[512], code_hash, label
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        except Exception as exc:
            logger.error(
                module="bronze_to_silver",
                function="extract_dl_tokens",
                error=f"Tokenizer load failed: {exc}",
            )
            return pd.DataFrame()

        rows = []
        for rec in records:
            code_hash = rec.get("code_hash", "")
            raw_code = rec.get("raw_code", "")
            label = rec.get("label")

            try:
                encoding = tokenizer(
                    raw_code,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None,
                )
                rows.append({
                    "code_hash": code_hash,
                    "input_ids": encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "label": label,
                })
            except Exception as exc:
                logger.warning(
                    module="bronze_to_silver",
                    function="extract_dl_tokens",
                    message=f"Tokenization failed for {code_hash[:12]}: {exc}",
                )

        return pd.DataFrame(rows)

    @AppLogger.log_function(module="bronze_to_silver")
    def push_silver_to_s3(self, df: pd.DataFrame, layer: str = "ml") -> str:
        """Push Silver DataFrame to DagsHub S3.

        Args:
            df: Feature DataFrame
            layer: "ml" or "dl"

        Returns:
            S3 key of the uploaded file
        """
        if df.empty:
            logger.warning(
                module="bronze_to_silver",
                function="push_silver_to_s3",
                message="Empty DataFrame — skipping push",
            )
            return ""

        now = datetime.utcnow()
        key = f"silver/{layer}/features_{now.strftime('%Y%m%d_%H%M%S')}.parquet"

        # Convert to parquet bytes
        buffer = df.to_parquet(index=False)
        if isinstance(buffer, bytes):
            data = buffer
        else:
            from io import BytesIO
            bio = BytesIO()
            df.to_parquet(bio, index=False)
            data = bio.getvalue()

        self._s3.upload_bytes(key=key, data=data, content_type="application/parquet")

        logger.info(
            module="bronze_to_silver",
            function="push_silver_to_s3",
            message=f"Silver pushed: {key} ({len(data)} bytes, {len(df)} rows)",
        )
        return key

    @AppLogger.log_function(module="bronze_to_silver")
    def run_full_etl(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full Bronze→Silver ETL for a batch of records.

        Returns:
            dict with ml_key, dl_key, record_count
        """
        t0 = time.perf_counter()

        # ML features
        ml_df = self.extract_ml_features(records)
        ml_key = self.push_silver_to_s3(ml_df, layer="ml")

        # DL tokens
        dl_df = self.extract_dl_tokens(records)
        dl_key = self.push_silver_to_s3(dl_df, layer="dl")

        latency = (time.perf_counter() - t0) * 1000
        return {
            "ml_key": ml_key,
            "dl_key": dl_key,
            "record_count": len(records),
            "ml_features": len(ml_df),
            "dl_tokens": len(dl_df),
            "latency_ms": round(latency, 1),
        }
