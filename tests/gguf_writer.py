"""Write small, valid GGUF v3 files for tests: header, metadata, tensor infos, and optional zero data."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sma.gguf import GGML_BLOCK_SIZES

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
_FORMATS = {UINT8: "B", INT8: "b", UINT16: "H", INT16: "h", UINT32: "I", INT32: "i", FLOAT32: "f", BOOL: "?",
            UINT64: "Q", INT64: "q", FLOAT64: "d"}


@dataclass(frozen=True)
class Typed:
    """A metadata value with an explicit GGUF type. item_type sets an array's element type."""

    type_id: int
    value: Any
    item_type: int | None = None


def _type_of(value: Any) -> int:
    if isinstance(value, Typed):
        return value.type_id
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return UINT32 if 0 <= value < 2**32 else INT64
    if isinstance(value, float):
        return FLOAT32
    if isinstance(value, str):
        return STRING
    if isinstance(value, list):
        return ARRAY
    raise TypeError(f"no GGUF type for {value!r}")


def _payload(type_id: int, value: Any, item_type: int | None = None) -> bytes:
    if isinstance(value, Typed):
        return _payload(value.type_id, value.value, value.item_type)
    if type_id == STRING:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw
    if type_id == ARRAY:
        if item_type is None:
            item_type = _type_of(value[0]) if value else UINT32
        return struct.pack("<IQ", item_type, len(value)) + b"".join(_payload(item_type, item) for item in value)
    return struct.pack("<" + _FORMATS[type_id], value)


def _tensor_bytes(shape: tuple[int, ...], type_id: int) -> int:
    block, size = GGML_BLOCK_SIZES.get(type_id, (1, 0))
    count = 1
    for dim in shape:
        count *= dim
    return count // block * size


def gguf_bytes(metadata: dict[str, Any], tensors: list[tuple[str, tuple[int, ...], int]] = (), *,
               version: int = 3, data: bool = False) -> bytes:
    """GGUF bytes. tensors are (name, shape, ggml type id). data=True appends aligned zero tensor data."""
    alignment = metadata.get("general.alignment", 32)
    alignment = alignment.value if isinstance(alignment, Typed) else alignment
    out = bytearray(b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(metadata)))
    for key, value in metadata.items():
        out += _payload(STRING, key) + struct.pack("<I", _type_of(value)) + _payload(_type_of(value), value)
    offset = 0
    sizes = []
    for name, shape, type_id in tensors:
        out += _payload(STRING, name) + struct.pack("<I", len(shape))
        out += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", type_id, offset)
        size = _tensor_bytes(shape, type_id)
        sizes.append(size)
        offset += -(-size // alignment) * alignment
    if data:
        out += bytes(-len(out) % alignment)
        for size in sizes:
            out += bytes(size + -size % alignment)
    return bytes(out)


def write_gguf(path: Path, metadata: dict[str, Any], tensors: list[tuple[str, tuple[int, ...], int]] = (),
               **kwargs: Any) -> Path:
    path.write_bytes(gguf_bytes(metadata, tensors, **kwargs))
    return path
