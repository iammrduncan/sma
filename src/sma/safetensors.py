"""Read safetensors headers without loading tensors or unpickling anything."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

_MAX_HEADER = 64 * 1024 * 1024


def read_header(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too small to be safetensors")
        (length,) = struct.unpack("<Q", prefix)
        if length > _MAX_HEADER:
            raise ValueError(f"{path} header is {length} bytes, over the {_MAX_HEADER} limit")
        blob = handle.read(length)
        if len(blob) != length:
            raise ValueError(f"{path} header was truncated")
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} header is not JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} header is not an object")
    return parsed
