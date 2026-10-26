"""Atomic file writes: temp file, fsync, rename.

A crash leaves the previous file intact. JSON is encoded with strict rules
(no NaN or Infinity) so a reader never sees a half-written document.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def dumps_json(payload: Any) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    return (text + "\n").encode("utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(Path(path), dumps_json(payload))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))
