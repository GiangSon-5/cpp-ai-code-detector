"""
shared/s3_client.py — DagsHub S3 client for Data Lake operations.

Handles upload/download to Bronze, Silver-ML, Silver-DL, and Gold
buckets on the DagsHub S3-compatible endpoint.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import boto3
import orjson
import pandas as pd
from botocore.config import Config as BotoConfig

from src.shared.config import settings
from src.shared.logger import AppLogger

logger = AppLogger()

# Bucket / prefix constants (all in the same DagsHub repo "bucket")
BUCKET_NAME = "data-lake"
PREFIX_BRONZE = "bronze/"
PREFIX_SILVER_ML = "silver/ml/"
PREFIX_SILVER_DL = "silver/dl/"
PREFIX_GOLD = "gold/"


def _is_local_mode() -> bool:
    """Check if we should use local filesystem instead of S3."""
    key = settings.AWS_ACCESS_KEY_ID
    return not key or key == "your_access_key" or "dummy" in key

def _get_local_path(key: str) -> Path:
    """Get local path for a given S3 key."""
    local_path = settings.PROJECT_ROOT / "data_lake" / key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return local_path

def _get_s3_client() -> Any:
    """Create a boto3 S3 client configured for DagsHub."""
    if _is_local_mode():
        return None
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
    )


# =====================================================================
# Upload helpers
# =====================================================================
@AppLogger.log_function(module="s3_client")
def upload_bronze_jsonl(code_hash: str, record: dict[str, Any]) -> str:
    """Upload a single Bronze record as JSONL to DagsHub S3."""
    key = f"{PREFIX_BRONZE}{code_hash}.jsonl"
    body = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
    s3 = _get_s3_client()
    if s3 is None:
        _get_local_path(key).write_bytes(body)
    else:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=body, ContentType="application/jsonl")
    return key


@AppLogger.log_function(module="s3_client")
def upload_silver_ml_parquet(code_hash: str, df: pd.DataFrame) -> str:
    """Upload Silver-ML features as Parquet."""
    key = f"{PREFIX_SILVER_ML}{code_hash}.parquet"
    s3 = _get_s3_client()
    if s3 is None:
        df.to_parquet(_get_local_path(key), engine="pyarrow", index=False)
    else:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue(), ContentType="application/octet-stream")
    return key


@AppLogger.log_function(module="s3_client")
def upload_silver_dl_parquet(code_hash: str, df: pd.DataFrame) -> str:
    """Upload Silver-DL token IDs as Parquet."""
    key = f"{PREFIX_SILVER_DL}{code_hash}.parquet"
    s3 = _get_s3_client()
    if s3 is None:
        df.to_parquet(_get_local_path(key), engine="pyarrow", index=False)
    else:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue(), ContentType="application/octet-stream")
    return key


@AppLogger.log_function(module="s3_client")
def upload_gold_parquet(code_hash: str, df: pd.DataFrame) -> str:
    """Upload Gold prediction data as Parquet."""
    key = f"{PREFIX_GOLD}{code_hash}.parquet"
    s3 = _get_s3_client()
    if s3 is None:
        df.to_parquet(_get_local_path(key), engine="pyarrow", index=False)
    else:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", index=False)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue(), ContentType="application/octet-stream")
    return key


# =====================================================================
# Download helpers
# =====================================================================
@AppLogger.log_function(module="s3_client")
def download_parquet(prefix: str, code_hash: str) -> pd.DataFrame:
    """Download a Parquet file and return as DataFrame."""
    key = f"{prefix}{code_hash}.parquet"
    s3 = _get_s3_client()
    if s3 is None:
        return pd.read_parquet(_get_local_path(key), engine="pyarrow")
    else:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        buf = io.BytesIO(response["Body"].read())
        return pd.read_parquet(buf, engine="pyarrow")


@AppLogger.log_function(module="s3_client")
def list_objects(prefix: str, max_keys: int = 1000) -> list[str]:
    """List object keys under a given prefix."""
    s3 = _get_s3_client()
    if s3 is None:
        local_dir = settings.PROJECT_ROOT / "data_lake" / prefix
        if not local_dir.exists():
            return []
        keys = []
        for file in local_dir.glob("**/*"):
            if file.is_file():
                # Get path relative to data_lake
                rel_path = file.relative_to(settings.PROJECT_ROOT / "data_lake")
                keys.append(str(rel_path).replace("\\", "/"))
        return keys[:max_keys]
    
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, MaxKeys=max_keys)
    return [obj["Key"] for obj in response.get("Contents", [])]

class DagsHubS3Client:
    """Class wrapper for DagsHub S3 compatible operations used by Celery tasks."""
    
    def __init__(self):
        self.s3 = _get_s3_client()
        
    def upload_bytes(self, key: str, data: bytes, content_type: str) -> None:
        """Always save locally, then attempt to upload to S3."""
        # 1. Always save a local copy for user inspection (as requested)
        local_path = _get_local_path(key)
        local_path.write_bytes(data)
        
        # 2. Attempt S3 upload if configured
        if self.s3 is not None:
            try:
                self.s3.put_object(
                    Bucket=BUCKET_NAME,
                    Key=key,
                    Body=data,
                    ContentType=content_type
                )
            except Exception as e:
                logger.warning(
                    module="s3_client",
                    function="upload_bytes",
                    message=f"S3 Upload failed, but local copy is saved: {e}"
                )
