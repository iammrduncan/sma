"""JSON logs on stderr and, once a run exists, in that run's logs directory."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from sma.scrub import scrub_text

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": scrub_text(record.getMessage()),
        }
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        if record.exc_info:
            payload["exc"] = scrub_text(self.formatException(record.exc_info))
        return json.dumps(payload, sort_keys=True)


def get_logger(name: str = "sma") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    return logger


def add_run_log(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    logging.getLogger("sma").addHandler(handler)
    return handler


def log_event(event: str, **fields: object) -> None:
    logger = get_logger()
    parts = [event] + [f"{key}={scrub_text(str(value))}" for key, value in fields.items()]
    logger.info(" ".join(parts), extra={"event": event})
