"""JSON logs with privacy-oriented field redaction."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

_REDACT = {"text", "transcript", "api_key", "authorization", "audio"}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.10 tests
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "latency_seconds", "status_code", "model_version"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        for key in _REDACT:
            payload.pop(key, None)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel((level or os.getenv("TTS_LOG_LEVEL") or "INFO").upper())
