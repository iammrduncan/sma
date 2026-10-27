"""Read a GGUF file's own metadata: architecture, quant type, tensors, and KV-cache layout.

SMA never converts a GGUF. It reads what the file says about itself. The reader needs only a
read(n) function, so it works on a local file or on a ranged HTTP read of a remote file, and it
never touches the tensor data.

Type sizes and enum values come from llama.cpp master (ggml.h, the block size asserts in
ggml-common.h, llama.h), checked on 2026-10-27. Layer rules follow llama.cpp's own loaders
(src/models/*.cpp and llama-hparams.cpp).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sma.errors import SmaError
from sma.memplan import KvLayout

# ggml type id: (name, values per block, bytes per block).
# gguf-py's GGML_QUANT_SIZES agrees with every entry. Gaps (4, 5, 31-33, 36-38) are removed types.
_GGML = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
GGML_TYPES = {type_id: name for type_id, (name, _, _) in _GGML.items()}
GGML_BLOCK_SIZES = {type_id: (block, size) for type_id, (_, block, size) in _GGML.items()}

# enum llama_ftype in include/llama.h, without the LLAMA_FTYPE_ prefix. Gaps are removed types.
FILE_TYPES = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS", 24: "MOSTLY_IQ1_S",
    25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
    36: "MOSTLY_TQ1_0", 37: "MOSTLY_TQ2_0", 38: "MOSTLY_MXFP4_MOE", 39: "MOSTLY_NVFP4",
    40: "MOSTLY_Q1_0", 41: "MOSTLY_Q2_0", 1024: "GUESSED",
}

# GGUF metadata value types. 8 is a string and 9 an array; the rest are fixed-size numbers.
_NUMBER_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_MAX_HEADER = 64 * 1024 * 1024
_TOO_BIG = "GGUF header is over the 64 MiB limit SMA reads. The file is damaged or not a model file."
_MAX_ARRAY_DEPTH = 8
_MAX_DIMS = 4  # GGML_MAX_DIMS
_STRETCHING = {"linear", "yarn", "longrope"}
# llama.cpp's loaders use this interval when a hybrid Qwen file has no full_attention_interval key.
_DEFAULT_FULL_ATTENTION_INTERVAL = {"qwen3next": 4, "qwen35": 4, "qwen35moe": 4}
# llama.cpp keeps recurrent (conv and SSM) state in F32.
_STATE_BYTES_PER_VALUE = 4


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    type_id: int
    type_name: str
    n_elements: int
    # None when the ggml type's size is unknown or the shape does not fill whole blocks.
    n_bytes: int | None
    offset: int


@dataclass
class GgufHeader:
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    alignment: int


@dataclass(frozen=True)
class GgufInfo:
    architecture: str
    name: str | None
    file_type: str | None
    file_type_id: int | None
    tensor_types: dict[str, int]
    tensor_bytes: dict[str, int]
    # None when only some files of a split set were read.
    params_total: int | None
    split_count: int
    is_moe: bool
    expert_count: int | None
    expert_used_count: int | None
    tokenizer_fingerprint: str | None
    layout: KvLayout
    notes: tuple[str, ...] = ()

    @property
    def native_context(self) -> int | None:
        return self.layout.native_context

    @property
    def native_source(self) -> str | None:
        return self.layout.native_source

    @property
    def rope_stretched(self) -> bool:
        return self.layout.rope_stretched


class _Stream:
    """Exact reads over read(n), with a cap so a hostile header cannot exhaust memory."""

    def __init__(self, read: Callable[[int], bytes]) -> None:
        self._read = read
        self.used = 0

    def take(self, n: int) -> bytes:
        if self.used + n > _MAX_HEADER:
            raise SmaError(_TOO_BIG)
        parts, missing = [], n
        while missing:
            chunk = self._read(missing)
            if not chunk:
                raise SmaError("GGUF header is truncated. Download the file again.")
            parts.append(chunk)
            missing -= len(chunk)
        self.used += n
        return b"".join(parts)

    def number(self, fmt: str) -> Any:
        fmt = "<" + fmt
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.number("Q")).decode("utf-8", "replace")

    def value(self, value_type: int, key: str, depth: int = 0) -> Any:
        if value_type in _NUMBER_FORMATS:
            return self.number(_NUMBER_FORMATS[value_type])
        if value_type == _STRING:
            return self.string()
        if value_type != _ARRAY:
            raise SmaError(f"GGUF metadata key {key!r} has unknown value type {value_type}. "
                           "The file is damaged or newer than SMA reads.")
        if depth >= _MAX_ARRAY_DEPTH:
            raise SmaError(f"GGUF metadata key {key!r} nests arrays too deep. The file is damaged.")
        item_type, count = self.number("I"), self.number("Q")
        if count > _MAX_HEADER:
            raise SmaError(_TOO_BIG)
        if item_type in _NUMBER_FORMATS:
            fmt = f"<{count}{_NUMBER_FORMATS[item_type]}"
            return list(struct.unpack(fmt, self.take(struct.calcsize(fmt))))
        return [self.value(item_type, key, depth + 1) for _ in range(count)]


def read_header(read: Callable[[int], bytes]) -> GgufHeader:
    """Parse the header, metadata, and tensor infos. Reads nothing past the last tensor info."""
    stream = _Stream(read)
    if stream.take(4) != b"GGUF":
        raise SmaError("not a GGUF file: it does not start with the GGUF magic bytes. "
                       "Check that the download is a .gguf file.")
    version = stream.number("I")
    if version not in (2, 3):
        raise SmaError(f"GGUF version {version} is not supported. SMA reads versions 2 and 3.")
    tensor_count, kv_count = stream.number("Q"), stream.number("Q")
    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = stream.string()
        if key in metadata:
            raise SmaError(f"GGUF metadata has key {key!r} twice. The file is damaged.")
        metadata[key] = stream.value(stream.number("I"), key)
    tensors = [_tensor_info(stream) for _ in range(tensor_count)]
    alignment = metadata.get("general.alignment", 32)
    if not _is_count(alignment) or alignment == 0:
        raise SmaError(f"GGUF general.alignment is {alignment!r}. The file is damaged.")
    return GgufHeader(version=version, metadata=metadata, tensors=tensors, alignment=alignment)


def read_header_file(path: str | Path) -> GgufHeader:
    path = Path(path)
    with path.open("rb") as handle:
        try:
            return read_header(handle.read)
        except SmaError as exc:
            raise SmaError(f"{path.name}: {exc}") from exc


def _tensor_info(stream: _Stream) -> TensorInfo:
    name = stream.string()
    n_dims = stream.number("I")
    if n_dims > _MAX_DIMS:
        raise SmaError(f"GGUF tensor {name!r} has {n_dims} dimensions; ggml allows {_MAX_DIMS}. The file is damaged.")
    shape = tuple(stream.number("Q") for _ in range(n_dims))
    type_id, offset = stream.number("I"), stream.number("Q")
    n_elements = math.prod(shape)
    n_bytes = None
    if type_id in GGML_BLOCK_SIZES:
        block, size = GGML_BLOCK_SIZES[type_id]
        if not shape or shape[0] % block == 0:
            n_bytes = n_elements // block * size
    type_name = GGML_TYPES.get(type_id, f"type {type_id}")
    return TensorInfo(name, shape, type_id, type_name, n_elements, n_bytes, offset)


def summarize(header: GgufHeader, *other_splits: GgufHeader) -> GgufInfo:
    """What SMA needs from one GGUF model. Pass the other files of a split set for whole-model tensor totals."""
    meta = header.metadata
    arch = meta.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise SmaError("GGUF file has no general.architecture. It is not a model file SMA can serve.")
    headers = (header, *other_splits)
    split_count = _count(meta, "split.count") or 1
    split_numbers = [_count(h.metadata, "split.no") or 0 for h in headers]
    if split_numbers[0] != 0 or sorted(split_numbers) != list(range(len(headers))) or len(headers) > split_count:
        raise SmaError("These GGUF headers are not one split set. Pass the first split file, then the others.")
    notes = []
    complete = len(headers) == split_count
    if not complete:
        notes.append(f"Tensor totals cover {len(headers)} of {split_count} split files.")

    tensors = [tensor for h in headers for tensor in h.tensors]
    tensor_bytes: dict[str, int] = {}
    for tensor in tensors:
        if tensor.n_bytes is not None:
            tensor_bytes[tensor.type_name] = tensor_bytes.get(tensor.type_name, 0) + tensor.n_bytes
    unsized = sorted({t.type_name for t in tensors if t.n_bytes is None})
    if unsized:
        notes.append(f"Tensor bytes leave out tensors whose size is unknown: {', '.join(unsized)}.")

    file_type_id = _count(meta, "general.file_type")
    file_type = None if file_type_id is None else FILE_TYPES.get(file_type_id, f"ftype {file_type_id}")
    expert_count = _max_count(meta, f"{arch}.expert_count")
    name = meta.get("general.name")
    return GgufInfo(
        architecture=arch,
        name=name if isinstance(name, str) else None,
        file_type=file_type,
        file_type_id=file_type_id,
        tensor_types=dict(Counter(t.type_name for t in tensors)),
        tensor_bytes=tensor_bytes,
        params_total=sum(t.n_elements for t in tensors) if complete else None,
        split_count=split_count,
        is_moe=bool(expert_count),
        expert_count=expert_count,
        expert_used_count=_max_count(meta, f"{arch}.expert_used_count"),
        tokenizer_fingerprint=_tokenizer_fingerprint(meta),
        layout=_layout(meta, arch),
        notes=tuple(notes),
    )


def _tokenizer_fingerprint(meta: dict[str, Any]) -> str | None:
    tokenizer = {key: value for key, value in meta.items() if key.startswith("tokenizer.")}
    if not tokenizer:
        return None
    blob = json.dumps(tokenizer, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _layout(meta: dict[str, Any], arch: str) -> KvLayout:
    def key(name: str) -> str:
        return f"{arch}.{name}"

    notes: list[str] = []
    block_count = _count(meta, key("block_count"))
    if not block_count:
        raise SmaError(f"GGUF metadata has no {key('block_count')}, so SMA cannot plan the KV cache.")
    # MTP layers sit after the trunk. llama.cpp's main context keeps no KV or state for them.
    mtp_layers = _count(meta, key("nextn_predict_layers")) or 0
    if mtp_layers >= block_count:
        raise SmaError(f"GGUF {key('nextn_predict_layers')} is not below {key('block_count')}. The file is damaged.")
    if mtp_layers:
        notes.append(f"{mtp_layers} MTP layer(s) after the trunk are not counted.")
    n = block_count - mtp_layers

    heads = _per_layer(meta, key("attention.head_count"), n) or [0] * n
    kv_heads = _per_layer(meta, key("attention.head_count_kv"), n) or heads
    embedding = _count(meta, key("embedding_length"))
    ssm = {part: _count(meta, key(f"ssm.{part}")) for part in ("conv_kernel", "inner_size", "state_size", "group_count")}
    has_ssm = any(value is not None for value in ssm.values())
    conv_cache = _count(meta, key("shortconv.l_cache"))

    # Which layers attend and which keep a fixed recurrent state (linear attention, conv, or Mamba).
    linear = _flags(meta, key("attention.recurrent_layers"), n)
    interval = _count(meta, key("full_attention_interval"))
    if linear is None and not interval and arch in _DEFAULT_FULL_ATTENTION_INTERVAL:
        interval = _DEFAULT_FULL_ATTENTION_INTERVAL[arch]
        notes.append(f"The file has no {key('full_attention_interval')}. llama.cpp's default of {interval} is used.")
    if linear is None and interval:
        # Qwen3-Next and Qwen3.5: every interval-th layer is full attention, the rest linear.
        linear = [(i + 1) % interval != 0 for i in range(n)]
    if linear is not None:
        recurrent = linear
    elif (has_ssm or conv_cache) and isinstance(meta.get(key("attention.head_count_kv")), list):
        # A layer with no KV heads is recurrent (LFM2 convolutions, Granite-H Mamba). Nemotron-H also lists
        # a feed-forward size per layer: there a layer with no KV heads but a feed-forward size is an MLP.
        per_layer_ff = isinstance(meta.get(key("feed_forward_length")), list)
        ff = (_per_layer(meta, key("feed_forward_length"), n) if per_layer_ff else None) or [0] * n
        recurrent = [kv == 0 and size == 0 for kv, size in zip(kv_heads, ff)]
    elif has_ssm:
        # Mamba keeps state in every layer. Falcon-H1 runs attention and Mamba side by side in every layer.
        recurrent = [True] * n
    else:
        recurrent = [False] * n
    attends = [kv > 0 and not (linear and linear[i]) for i, kv in enumerate(kv_heads)]
    # Gemma 4: the last shared_kv_layers layers reuse KV from earlier layers.
    shared = _count(meta, key("attention.shared_kv_layers")) or 0
    if shared:
        attends = [attend and i < block_count - shared for i, attend in enumerate(attends)]
        notes.append(f"The last {shared} layers reuse earlier KV and are not counted.")
    if not any(attends) and not any(recurrent):
        raise SmaError(f"SMA cannot find attention or recurrent layers in this {arch} GGUF, so it cannot plan the KV cache.")

    window = _count(meta, key("attention.sliding_window")) or 0
    sliding = [False] * n
    if window:
        pattern = meta.get(key("attention.sliding_window_pattern"))
        if isinstance(pattern, list):
            sliding = _flags(meta, key("attention.sliding_window_pattern"), n) or sliding
        elif _is_count(pattern):
            # llama.cpp's set_swa_pattern: the last layer of each group of `pattern` is full. 0 means all slide.
            sliding = [pattern == 0 or i % pattern < pattern - 1 for i in range(n)]
        else:
            notes.append(f"{key('attention.sliding_window')} is {window}, but the file does not say which layers slide. "
                         "Every attention layer is counted as full, which overcounts KV.")

    full_values = sliding_values = full_layers = sliding_layers = 0
    if any(attends):
        first_heads = next((h for h in heads if h), 0)
        head_size = embedding // first_heads if embedding and first_heads else None
        key_size = _count(meta, key("attention.key_length")) or head_size
        value_size = _count(meta, key("attention.value_length")) or head_size
        if not key_size or not value_size:
            raise SmaError(f"GGUF metadata has no {key('attention.key_length')} or {key('embedding_length')}, "
                           "so SMA cannot size the KV cache.")
        sliding_size = (_count(meta, key("attention.key_length_swa")) or key_size) + (
            _count(meta, key("attention.value_length_swa")) or value_size)
        for i, attend in enumerate(attends):
            if attend and sliding[i]:
                sliding_layers += 1
                sliding_values += kv_heads[i] * sliding_size
            elif attend:
                full_layers += 1
                full_values += kv_heads[i] * (key_size + value_size)
    if meta.get(key("attention.kv_lora_rank")):
        notes.append("MLA model: KV is counted from key_length and value_length. llama.cpp may store less.")

    state_values = 0
    if any(recurrent):
        if conv_cache and embedding:
            state_values = embedding * (conv_cache - 1)
        elif ssm["conv_kernel"] and ssm["inner_size"] and ssm["state_size"]:
            # llama.cpp's n_embd_r (conv state) plus n_embd_s (SSM state) per layer.
            conv, inner, state = ssm["conv_kernel"], ssm["inner_size"], ssm["state_size"]
            state_values = (conv - 1) * (inner + 2 * (ssm["group_count"] or 0) * state) + state * inner
        else:
            notes.append("Recurrent state size is not computable from this file. state_bytes_per_stream is 0.")

    native, source, stretched = _native_context(meta, arch, notes)
    return KvLayout(
        total_layers=n,
        full_attention_layers=full_layers,
        full_values_per_token=full_values,
        sliding_attention_layers=sliding_layers,
        sliding_values_per_token=sliding_values,
        sliding_window=window if sliding_layers else None,
        state_bytes_per_stream=sum(recurrent) * state_values * _STATE_BYTES_PER_VALUE,
        native_context=native,
        native_source=source,
        rope_stretched=stretched,
        notes=tuple(notes),
    )


def _native_context(meta: dict[str, Any], arch: str, notes: list[str]) -> tuple[int | None, str | None, bool]:
    context_key = f"{arch}.context_length"
    original_key = f"{arch}.rope.scaling.original_context_length"
    scaling = meta.get(f"{arch}.rope.scaling.type")
    factor = meta.get(f"{arch}.rope.scaling.factor")
    stretched = isinstance(scaling, str) and scaling.lower() in _STRETCHING
    if isinstance(factor, (int, float)) and factor <= 1:
        stretched = False
    if stretched:
        original = _count(meta, original_key)
        if original:
            return original, original_key, True
        notes.append(f"RoPE {scaling} stretching is on but the file has no {original_key}. The native context is unknown.")
        return None, None, True
    context = _count(meta, context_key)
    if context is None:
        notes.append(f"The file has no {context_key}.")
        return None, None, False
    return context, context_key, False


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _count(meta: dict[str, Any], key: str) -> int | None:
    value = meta.get(key)
    if value is not None and not _is_count(value):
        raise SmaError(f"GGUF metadata {key} is {value!r}, not a count. The file is damaged.")
    return value


def _max_count(meta: dict[str, Any], key: str) -> int | None:
    """A count that newer files may store per layer."""
    value = meta.get(key)
    if isinstance(value, list) and all(_is_count(v) for v in value):
        return max(value, default=0)
    return _count(meta, key)


def _per_layer(meta: dict[str, Any], key: str, n: int) -> list[int] | None:
    value = meta.get(key)
    if value is None:
        return None
    values = value if isinstance(value, list) else [value] * n
    if len(values) < n or not all(_is_count(v) for v in values):
        raise SmaError(f"GGUF metadata {key} is not a count or a list of {n} counts. The file is damaged.")
    return values[:n]


def _flags(meta: dict[str, Any], key: str, n: int) -> list[bool] | None:
    value = meta.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or len(value) < n:
        raise SmaError(f"GGUF metadata {key} is not a list of {n} per-layer flags. The file is damaged.")
    return [bool(v) for v in value[:n]]


_SPLIT_SUFFIX = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
# Quant tags such as Q8_0, Q4_K_M, IQ2_XXS, UD-Q4_K_XL, MXFP4, BF16, as one token between - . / or the ends.
_QUANT_TAG = re.compile(r"(?<![^-./])((?:UD-)?(?:I?Q\d|TQ\d|MXFP4|NVFP4|BF16|F16|F32|F64)[A-Za-z0-9_]*)(?![^-./])",
                        re.IGNORECASE)


def _model_files(files: list[str]) -> list[str]:
    """GGUF model files. Vision projectors (mmproj) and MTP draft heads (mtp-*, Gemma 4) are not models."""
    names = {f: f.rsplit("/", 1)[-1].lower() for f in files}
    return sorted(f for f, name in names.items()
                  if name.endswith(".gguf") and "mmproj" not in name and not name.startswith("mtp-"))


def match_gguf_files(files: list[str], selector: str) -> list[str]:
    """The one file, or the one complete split set, whose path holds the selector as a whole token.

    Tokens are split by '-', '.', and '/'. An underscore is part of the token, so Q4_K does not match
    Q4_K_M or Q4_K_XL. The match ignores case.
    """
    available = ", ".join(gguf_quant_tags(files)) or "none"
    if not selector.strip():
        raise SmaError(f"The GGUF quant selector is empty. Available quant selectors: {available}.")
    token = re.compile(rf"(?<![^-./]){re.escape(selector)}(?![^-./])", re.IGNORECASE)
    models: dict[str, list[str]] = {}
    for path in _model_files(files):
        if token.search(path):
            models.setdefault(_SPLIT_SUFFIX.sub("", path), []).append(path)
    if not models:
        raise SmaError(f"No GGUF model file in this repo matches {selector!r}. Available quant selectors: {available}.")
    if len(models) > 1:
        raise SmaError(f"{selector!r} matches more than one GGUF model: {', '.join(sorted(models))}. "
                       f"Pass a more specific selector. Available quant selectors: {available}.")
    (paths,) = models.values()
    parts = [_SPLIT_SUFFIX.search(path) for path in paths]
    if parts[0] is None:
        return paths
    totals = {int(part.group(2)) for part in parts if part}
    numbers = [int(part.group(1)) for part in parts if part]
    if len(totals) != 1 or numbers != list(range(1, max(totals) + 1)):
        raise SmaError(f"The split set for {selector!r} is incomplete: found parts {numbers} of {max(totals)}. "
                       "Check the repo revision.")
    return paths


def gguf_quant_tags(files: list[str]) -> list[str]:
    """Best-effort quant selectors in a repo listing, from file and folder names."""
    tags = set()
    for path in _model_files(files):
        found = _QUANT_TAG.findall(_SPLIT_SUFFIX.sub("", path))
        if found:
            tags.add(found[-1])
    return sorted(tags)
