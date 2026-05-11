"""
shared/logger.py — Deep Logging (2-Session Rotation)

CORE RULES:
    1. All log output is JSON (one object per line → JSONL).
    2. Only TWO log files are ever kept:
         • logs/current_run.log.json   ← the active session
         • logs/previous_run.log.json  ← the immediately preceding session
    3. On application boot (.rotate_on_startup()):
         - delete previous_run (if exists)
         - rename current_run → previous_run
         - create a fresh current_run
    4. Every log entry contains at minimum:
         timestamp, level, module, function, input, output, error, latency_ms
    5. The logger exposes a decorator @AppLogger.log_function() that
       auto-wraps any sync/async function to capture all of the above.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import orjson  # 3-5× faster than stdlib json

from src.shared.config import settings


class AppLogger:
    """Project-wide JSON logger with 2-session file rotation."""

    _instance: "AppLogger | None" = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "AppLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Path | None = None) -> None:
        if hasattr(self, "_initialised"):
            return
        self._initialised = True

        self._log_dir = log_dir or settings.LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._current_path = self._log_dir / "current_run.log.json"
        self._previous_path = self._log_dir / "previous_run.log.json"

        # File handle (opened on first write or explicit rotate)
        self._fh: Any | None = None

    # ------------------------------------------------------------------
    # Session rotation
    # ------------------------------------------------------------------
    def rotate_on_startup(self) -> None:
        """Call ONCE when the application boots.

        Lifecycle:
            previous_run.log.json  → deleted
            current_run.log.json   → renamed to previous_run.log.json
            (new) current_run.log.json created (empty)
        """
        # Close any open handle from a prior import
        self._close()

        # Step 1 — delete old previous
        if self._previous_path.exists():
            self._previous_path.unlink()

        # Step 2 — current → previous
        if self._current_path.exists():
            self._current_path.rename(self._previous_path)

        # Step 3 — open fresh current
        self._open()

        self.info(
            module="logger",
            function="rotate_on_startup",
            message="Session started — log rotation complete.",
            input_data=None,
            output_data={
                "current": str(self._current_path),
                "previous": str(self._previous_path),
            },
        )

    # ------------------------------------------------------------------
    # Low-level file I/O
    # ------------------------------------------------------------------
    def _open(self) -> None:
        self._fh = open(self._current_path, "ab")  # append-binary for orjson

    def _close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
        self._fh = None

    def _ensure_open(self) -> None:
        if self._fh is None or self._fh.closed:
            self._open()

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------
    def _write(self, record: dict[str, Any]) -> None:
        self._ensure_open()
        line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
        try:
            self._fh.write(line)  # type: ignore[union-attr]
            self._fh.flush()  # type: ignore[union-attr]
        except Exception:
            # Fallback: reopen and retry once
            self._close()
            self._open()
            self._fh.write(line)  # type: ignore[union-attr]
            self._fh.flush()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API — structured log levels
    # ------------------------------------------------------------------
    def _log(
        self,
        level: str,
        *,
        module: str,
        function: str,
        message: str = "",
        input_data: Any = None,
        output_data: Any = None,
        error: str | None = None,
        latency_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "module": module,
            "function": function,
            "message": message,
            "input": self._safe_serialise(input_data),
            "output": self._safe_serialise(output_data),
            "error": error,
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
        }
        if extra:
            record["extra"] = extra
        self._write(record)

    def info(self, **kwargs: Any) -> None:
        self._log("INFO", **kwargs)

    def warning(self, **kwargs: Any) -> None:
        self._log("WARNING", **kwargs)

    def error(self, **kwargs: Any) -> None:
        self._log("ERROR", **kwargs)

    def debug(self, **kwargs: Any) -> None:
        self._log("DEBUG", **kwargs)

    # ------------------------------------------------------------------
    # Decorator — auto-instrument any function (sync or async)
    # ------------------------------------------------------------------
    @staticmethod
    def log_function(module: str | None = None) -> Callable:
        """Decorator that wraps a function to auto-log input, output,
        error, and latency_ms to `current_run.log.json`.

        Usage::

            @AppLogger.log_function(module="engine")
            def predict(code: str) -> dict: ...

            @AppLogger.log_function()
            async def fetch_data(url: str) -> bytes: ...
        """

        def decorator(fn: Callable) -> Callable:
            _module = module or fn.__module__
            _fname = fn.__qualname__

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    logger = AppLogger()
                    input_snapshot = _snapshot_args(args, kwargs)
                    t0 = time.perf_counter()
                    try:
                        result = await fn(*args, **kwargs)
                        latency = (time.perf_counter() - t0) * 1000
                        logger.info(
                            module=_module,
                            function=_fname,
                            input_data=input_snapshot,
                            output_data=result,
                            latency_ms=latency,
                        )
                        return result
                    except Exception as exc:
                        latency = (time.perf_counter() - t0) * 1000
                        logger.error(
                            module=_module,
                            function=_fname,
                            input_data=input_snapshot,
                            output_data=None,
                            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                            latency_ms=latency,
                        )
                        raise

                return async_wrapper
            else:
                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    logger = AppLogger()
                    input_snapshot = _snapshot_args(args, kwargs)
                    t0 = time.perf_counter()
                    try:
                        result = fn(*args, **kwargs)
                        latency = (time.perf_counter() - t0) * 1000
                        logger.info(
                            module=_module,
                            function=_fname,
                            input_data=input_snapshot,
                            output_data=result,
                            latency_ms=latency,
                        )
                        return result
                    except Exception as exc:
                        latency = (time.perf_counter() - t0) * 1000
                        logger.error(
                            module=_module,
                            function=_fname,
                            input_data=input_snapshot,
                            output_data=None,
                            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                            latency_ms=latency,
                        )
                        raise

                return sync_wrapper

        return decorator

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_serialise(obj: Any) -> Any:
        """Return a JSON-safe representation (best effort)."""
        if obj is None:
            return None
        try:
            orjson.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return repr(obj)[:2000]


def _snapshot_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Create a JSON-safe snapshot of function arguments."""
    safe_args = []
    for a in args:
        try:
            orjson.dumps(a)
            safe_args.append(a)
        except (TypeError, ValueError):
            safe_args.append(repr(a)[:500])

    safe_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        try:
            orjson.dumps(v)
            safe_kwargs[k] = v
        except (TypeError, ValueError):
            safe_kwargs[k] = repr(v)[:500]

    return {"args": safe_args, "kwargs": safe_kwargs}
