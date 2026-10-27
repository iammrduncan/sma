"""Atomic file writes: temp file, fsync, rename.

A crash leaves the previous file intact. JSON is encoded with strict rules
(no NaN or Infinity) so a reader never sees a half-written document.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterable
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


def atomic_write_json_stream(
    path: Path, head: dict[str, Any], key: str, items: Iterable[Any], tail: Callable[[], dict[str, Any]]
) -> None:
    """Write {head..., key: [items...], tail()...} one item at a time, with the same temp-file rename.

    For large records (the logprob dump): only one item is in memory at a time. `tail` runs after the
    last item, so it can hold totals computed while the items were written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")

    def _encode(value: Any) -> str:
        return json.dumps(value, sort_keys=True, allow_nan=False)

    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write("{")
            for name, value in head.items():
                handle.write(f"{_encode(name)}: {_encode(value)}, ")
            handle.write(f"{_encode(key)}: [")
            for index, item in enumerate(items):
                handle.write((",\n" if index else "\n") + _encode(item))
            handle.write("]")
            for name, value in tail().items():
                handle.write(f", {_encode(name)}: {_encode(value)}")
            handle.write("}\n")
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
