"""Readable lines on stderr, and JSON in the run's logs/sma.log once a run exists."""

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


# How an event reads on the terminal. Fields fill the braces. An event without an entry here
# prints as "event key=value ...". sma.log always keeps the full JSON record.
_CONSOLE = {
    "run_created": "Run {run_id}: {repo} ({quant})",
    "step_start": "▶ {step}",
    "step_done": "✓ {step}",
    "step_skip": "· {step} was already done",
    "step_failed": "✗ {step} failed: {error}",
    "retry": "↻ {what}: attempt {attempt} failed, retrying in {sleep_s}s. {error}",
    "warning": "! {message}",
    "image_pull_start": "Pulling {image}",
    "image_pull_progress": "  {line}",
    "fetched": "Downloaded {repo} at commit {revision_sha}",
    "docker_run": "Starting the {mode} server ({name})",
    "server_log": "  Live server log: {path}",
    "server_wait": "  Waiting for the {mode} server, {elapsed_s}s: {line}",
    "server_ready": "✓ {mode} server ready after {elapsed_s}s",
}


class _Fields(dict):  # type: ignore[type-arg]
    def __missing__(self, key: str) -> str:
        return "?"


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stamp = datetime.now().strftime("%H:%M:%S")
        event = getattr(record, "event", None)
        fields = getattr(record, "fields", None)
        template = _CONSOLE.get(event or "")
        if template is not None and isinstance(fields, dict):
            text = template.format_map(_Fields(fields))
            if event == "step_start" and str(fields.get("attempt", "1")) != "1":
                text += f" (attempt {fields['attempt']})"
        else:
            text = scrub_text(record.getMessage())
        if record.exc_info:
            text += "\n" + scrub_text(self.formatException(record.exc_info))
        return f"{stamp}  {text}"


def get_logger(name: str = "sma") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_ConsoleFormatter())
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
    cleaned = {key: scrub_text(str(value)) for key, value in fields.items()}
    parts = [event] + [f"{key}={value}" for key, value in cleaned.items()]
    logger.info(" ".join(parts), extra={"event": event, "fields": cleaned})
