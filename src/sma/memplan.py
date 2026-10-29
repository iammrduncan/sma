"""Plan a run's memory before launch: weights, KV pool, and engine overhead under one budget.

See docs/serving-design.md. The KV cache is always 8-bit. Context starts at 256K, or the model's
native maximum if smaller, and steps down by halves until it fits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from sma.errors import SmaError

# Bytes per stored K or V value. FP8 on vLLM and SGLang. llama.cpp q8_0 packs 32 int8 values with
# one fp16 scale, 34 bytes per 32 values.
KV_TYPES = {"fp8": 1.0, "q8_0": 34 / 32}

LADDER_START = 262_144

# The deployment notes also say how many streams fit at this context.
SHORT_CONTEXT = 32_768

# RoPE types that stretch a model past the length it was trained at. SMA never turns one on.
_STRETCHING_ROPE = {"yarn", "linear", "longrope", "dynamic"}

# What each layer keeps per stream. Names come from `layer_types` (Qwen3.5 and 3.6, LFM2, Gemma,
# Granite) and from Nemotron-H's `layers_block_type`. MoE and MLP blocks keep nothing.
_LAYER_KINDS = {
    "full_attention": "full",
    "attention": "full",
    "sliding_attention": "sliding",
    "linear_attention": "linear",
    "conv": "conv",
    "mamba": "mamba",
    "moe": None,
    "mlp": None,
}
# Nemotron-H's older `hybrid_override_pattern`: one character per layer.
_PATTERN_KINDS = {"*": "full", "M": "mamba", "-": None, "E": None}

_DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}


@dataclass(frozen=True)
class KvLayout:
    """How much KV a model keeps, read from its config (safetensors) or its GGUF metadata."""

    total_layers: int
    # Layers that keep K and V for every token, and the K plus V values they store per token, summed.
    full_attention_layers: int
    full_values_per_token: int
    # Layers that keep K and V only for the last sliding_window tokens.
    sliding_attention_layers: int
    sliding_values_per_token: int
    sliding_window: int | None
    # Fixed state per stream of linear-attention, convolution, or Mamba layers. An estimate.
    state_bytes_per_stream: int
    # The longest context the model supports without RoPE stretching. None when the config has none.
    native_context: int | None
    native_source: str | None
    # The config itself already turns on RoPE stretching (YaRN or similar). SMA never adds it.
    rope_stretched: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Need:
    """One benchmark's demand: streams it runs at once, and the least context each stream needs."""

    name: str
    streams: int
    min_context: int

    def __post_init__(self) -> None:
        if self.streams < 1 or self.min_context < 1:
            raise SmaError(
                f"Benchmark {self.name!r} asks for {self.streams} streams of {self.min_context} tokens. "
                "Set at least 1 stream and at least 1 token."
            )


@dataclass(frozen=True)
class MemoryPlan:
    """The memory plan for one run. Every value is computed before launch, none is measured."""

    kv_type: str
    budget_bytes: int
    weights_bytes: int
    overhead_bytes: int
    ladder: tuple[int, ...]
    # Context per stream: the largest ladder step where one stream fits next to the weights.
    context: int
    context_reason: str
    # Bytes one token adds in the full-attention layers. The sliding layers stand in when the model
    # has no full-attention layers. None when it has no attention layers at all.
    kv_bytes_per_token: int | None
    # One stream at `context`, including the fixed state of linear-attention, conv, or Mamba layers.
    stream_bytes: int
    # Everything left after weights, overhead, and buffers becomes the KV pool, unless the engine runs a
    # fixed number of streams (llama.cpp's slots): then it holds those streams at `context` and no more.
    pool_bytes: int
    # Tokens the pool holds, divided by kv_bytes_per_token. It ignores sliding layers and fixed
    # state, so it is an upper bound. None when kv_bytes_per_token is None.
    pool_tokens: int | None
    streams_full_context: int
    streams_32k: int
    footprint_bytes: int
    # Benchmark name -> context per stream: the largest step that fits its streams at once.
    benchmarks: dict[str, int]
    notes: tuple[str, ...] = ()
    # Engine memory per token the pool holds (llama.cpp sizes its compute buffers by its whole cache),
    # and that memory for the whole pool. Inside footprint_bytes.
    buffer_bytes_per_token: int = 0
    buffer_bytes: int = 0
    # The streams the pool is capped at, at `context` (llama.cpp's slots). None: the pool is the rest
    # of the budget.
    pool_streams: int | None = None

    def to_json(self) -> dict:
        """Plain JSON: every field, plus a rounded GB value next to each byte count."""
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        data.update(ladder=list(self.ladder), benchmarks=dict(self.benchmarks), notes=list(self.notes))
        for name in ("budget", "weights", "overhead", "stream", "pool", "buffer", "footprint"):
            data[f"{name}_gb"] = round(data[f"{name}_bytes"] / 1e9, 2)
        return data


def kv_layout_from_hf_config(config: dict) -> KvLayout:
    """Read the KV layout from a Hugging Face config.json. The config is untrusted input."""
    if not isinstance(config, dict):
        raise SmaError("config.json is not a JSON object. Check the checkpoint's config.")
    cfg = _text_config(config)
    layers = _count(cfg, "num_hidden_layers")
    if layers is None:
        raise SmaError(
            "config.json gives no num_hidden_layers, so SMA cannot plan KV memory. "
            "Check that the repo is a text-generation checkpoint."
        )
    notes: list[str] = []
    kinds = _layer_kinds(cfg, layers)

    window = _sliding_window(cfg)
    if "sliding" in kinds and window is None:
        notes.append("config.json marks sliding layers but turns no window on. They are counted as full attention.")
        kinds = ["full" if kind == "sliding" else kind for kind in kinds]
    elif "sliding" not in kinds and window is not None:
        notes.append(f"config.json sets a {window:,}-token sliding window but marks no sliding layers. "
                     "Every attention layer is counted as full attention.")
        window = None

    full_layers = kinds.count("full")
    sliding_layers = kinds.count("sliding")
    per_layer = _attention_values_per_layer(cfg, notes) if full_layers + sliding_layers else 0
    native, source, stretched = _native_context(cfg, notes)
    return KvLayout(
        total_layers=layers,
        full_attention_layers=full_layers,
        full_values_per_token=full_layers * per_layer,
        sliding_attention_layers=sliding_layers,
        sliding_values_per_token=sliding_layers * per_layer,
        sliding_window=window,
        state_bytes_per_stream=_state_bytes(cfg, kinds, notes),
        native_context=native,
        native_source=source,
        rope_stretched=stretched,
        notes=tuple(notes),
    )


def stream_kv_bytes(layout: KvLayout, context: int, kv_type: str) -> int:
    """Bytes one stream needs at this context: its K and V, plus any fixed per-stream state."""
    bytes_per_value = _bytes_per_value(kv_type)
    window_tokens = min(context, layout.sliding_window or 0)
    values = layout.full_values_per_token * context + layout.sliding_values_per_token * window_tokens
    return math.ceil(values * bytes_per_value) + layout.state_bytes_per_stream


def context_ladder(native_context: int | None, start: int = LADDER_START, floor: int = 1024) -> list[int]:
    """The contexts to try, largest first: start or the native maximum, then halves of start."""
    if native_context is None:
        raise SmaError(
            "The model's config.json gives no native context length, so SMA cannot choose a context. "
            "Check max_position_embeddings and any RoPE scaling settings in config.json."
        )
    assert floor >= 1, "the ladder floor must be at least 1 token"
    top = min(start, native_context)
    ladder = [top]
    step = start
    while step >= floor:
        if step < top:
            ladder.append(step)
        step //= 2
    return ladder


def plan_memory(
    *,
    layout: KvLayout,
    weights_bytes: int,
    kv_type: str,
    budget_bytes: int,
    overhead_bytes: int,
    needs: list[Need],
    start: int = LADDER_START,
    buffer_bytes_per_token: int = 0,
    max_streams: int | None = None,
) -> MemoryPlan:
    """Pick the context per stream, size the KV pool, and give each benchmark its context.

    `buffer_bytes_per_token` is engine memory for each token the pool holds: llama.cpp sizes its
    compute buffers by its whole cache. The pool shrinks so that pool plus buffers fit the budget.
    `max_streams` caps the pool at that many streams at the chosen context, for an engine that never
    runs more at once (llama.cpp's slots). A bigger pool would hold tokens no stream can reach.
    """
    bytes_per_value = _bytes_per_value(kv_type)
    fixed_bytes = weights_bytes + overhead_bytes
    if fixed_bytes > budget_bytes:
        raise SmaError(
            f"The weights ({_gb(weights_bytes)}) and engine overhead ({_gb(overhead_bytes)}) need "
            f"{_gb(fixed_bytes)}, more than the {_gb(budget_bytes)} memory budget. "
            "Choose a smaller checkpoint or quant."
        )
    token_values = layout.full_values_per_token or layout.sliding_values_per_token
    kv_bytes_per_token = math.ceil(token_values * bytes_per_value) if token_values else None
    buffer_share = buffer_bytes_per_token / kv_bytes_per_token if kv_bytes_per_token else 0.0
    pool_bytes = int((budget_bytes - fixed_bytes) / (1 + buffer_share))
    ladder = context_ladder(layout.native_context, start)

    def stream(context: int) -> int:
        return stream_kv_bytes(layout, context, kv_type)

    if stream(ladder[0]) == 0:
        raise SmaError(
            "config.json gives no KV cache or state size for this model, so SMA cannot plan its streams. "
            "Check the layer settings in config.json."
        )
    fitting = [step for step in ladder if stream(step) <= pool_bytes]
    if not fitting:
        raise SmaError(
            f"One stream at {ladder[-1]:,} tokens needs {_gb(stream(ladder[-1]))} of KV cache, but only "
            f"{_gb(pool_bytes)} is left after {_gb(weights_bytes)} of weights and {_gb(overhead_bytes)} "
            f"of engine overhead in the {_gb(budget_bytes)} budget. Choose a smaller checkpoint or quant."
        )
    context = fitting[0]
    if context < ladder[0]:
        reason = f"stepped down from {ladder[0]:,} to {context:,} tokens to fit the {_gb(budget_bytes)} budget"
    elif ladder[0] < start:
        reason = f"native maximum {context:,} tokens fits"
    else:
        reason = f"{context:,} tokens fits"

    stream_bytes = stream(context)
    capped = max_streams is not None and max_streams * stream_bytes < pool_bytes
    if capped:
        pool_bytes = max_streams * stream_bytes
    pool_tokens = pool_bytes // kv_bytes_per_token if kv_bytes_per_token else None
    buffer_bytes = (pool_tokens or 0) * buffer_bytes_per_token
    benchmarks: dict[str, int] = {}
    for need in needs:
        steps = [step for step in ladder if step <= context]
        best = next((step for step in steps if need.streams * stream(step) <= pool_bytes), None)
        if best is None or best < need.min_context:
            raise SmaError(_need_refusal(need, steps, context, reason, pool_bytes, stream))
        benchmarks[need.name] = best

    return MemoryPlan(
        kv_type=kv_type,
        budget_bytes=budget_bytes,
        weights_bytes=weights_bytes,
        overhead_bytes=overhead_bytes,
        ladder=tuple(ladder),
        context=context,
        context_reason=reason,
        kv_bytes_per_token=kv_bytes_per_token,
        stream_bytes=stream_bytes,
        pool_bytes=pool_bytes,
        pool_tokens=pool_tokens,
        streams_full_context=pool_bytes // stream_bytes,
        streams_32k=pool_bytes // stream(min(SHORT_CONTEXT, context)),
        footprint_bytes=fixed_bytes + pool_bytes + buffer_bytes,
        benchmarks=benchmarks,
        notes=layout.notes,
        buffer_bytes_per_token=buffer_bytes_per_token,
        buffer_bytes=buffer_bytes,
        pool_streams=max_streams if capped else None,
    )


def _need_refusal(need: Need, steps: list[int], context: int, reason: str, pool_bytes: int, stream) -> str:
    head = f"Benchmark {need.name!r} runs {need.streams} streams and needs at least {need.min_context:,} tokens each."
    if need.min_context > context:
        return f"{head} The model's context is {context:,} tokens ({reason}). Lower the benchmark's minimum context."
    smallest = min(step for step in steps if step >= need.min_context)
    needed = need.streams * stream(smallest)
    return (
        f"{head} {need.streams} streams at {smallest:,} tokens need {_gb(needed)} of KV cache, but the pool "
        f"holds {_gb(pool_bytes)}. Run fewer streams at once, or choose a smaller checkpoint or quant."
    )


def _bytes_per_value(kv_type: str) -> float:
    if kv_type not in KV_TYPES:
        raise SmaError(f"Unknown KV cache type {kv_type!r}. Use one of: {', '.join(KV_TYPES)}.")
    return KV_TYPES[kv_type]


def _gb(n: float) -> str:
    return f"{round(n / 1e9, 2):g} GB"


def _text_config(config: dict) -> dict:
    # Multimodal configs keep the language model in text_config (Qwen3.5/3.6, Gemma) or llm_config
    # (InternVL). Its keys win. Top-level keys such as dtype fill the gaps.
    for key in ("text_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            return {**config, **nested}
    return config


def _count(cfg: dict, key: str) -> int | None:
    """A positive whole number from the config, or None when the key is missing or null."""
    value = cfg.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SmaError(f"config.json sets {key} to {value!r}, not a positive whole number. Check the checkpoint's config.")
    return value


def _first_count(cfg: dict, *keys: str) -> int | None:
    for key in keys:
        value = _count(cfg, key)
        if value is not None:
            return value
    return None


def _layer_kinds(cfg: dict, layers: int) -> list[str | None]:
    """What each layer keeps: "full", "sliding", "linear", "conv", "mamba", or None for nothing."""
    for key in ("layer_types", "layers_block_type"):
        names = cfg.get(key)
        if names is None:
            continue
        if not isinstance(names, list) or len(names) != layers:
            raise SmaError(f"config.json {key} does not list one entry for each of the {layers} layers. "
                           "Check the checkpoint's config.")
        unknown = sorted({repr(name) for name in names if not isinstance(name, str) or name not in _LAYER_KINDS})
        if unknown:
            raise SmaError(f"config.json {key} names layer types SMA cannot size yet: {', '.join(unknown)}. "
                           "The memory planner needs support for them before this model can run.")
        return [_LAYER_KINDS[name] for name in names]
    pattern = cfg.get("hybrid_override_pattern")
    if pattern is not None:
        if not isinstance(pattern, str) or len(pattern) != layers or set(pattern) - set(_PATTERN_KINDS):
            raise SmaError(f"config.json hybrid_override_pattern is not one of {''.join(_PATTERN_KINDS)} "
                           f"for each of the {layers} layers. Check the checkpoint's config.")
        return [_PATTERN_KINDS[char] for char in pattern]
    # Qwen3-Next style: every interval-th layer is full attention, the rest linear attention.
    interval = _count(cfg, "full_attention_interval")
    if interval:
        return ["full" if (index + 1) % interval == 0 else "linear" for index in range(layers)]
    return ["full"] * layers


def _sliding_window(cfg: dict) -> int | None:
    window = cfg.get("sliding_window")
    # Many configs carry a window that is switched off (Qwen2 and Qwen3 set use_sliding_window false).
    if cfg.get("use_sliding_window") is False or isinstance(window, bool) or not isinstance(window, int):
        return None
    return window if window > 0 else None


def _attention_values_per_layer(cfg: dict, notes: list[str]) -> int:
    """K plus V values one attention layer stores per token."""
    rank = _count(cfg, "kv_lora_rank")
    if rank is not None:
        rope_dim = _count(cfg, "qk_rope_head_dim")
        if rope_dim is None:
            raise SmaError("config.json sets kv_lora_rank but no qk_rope_head_dim, so SMA cannot size the MLA cache. "
                           "Check the checkpoint's config.")
        notes.append("MLA attention: each layer keeps one compressed latent of kv_lora_rank + qk_rope_head_dim "
                     "values per token, not separate K and V. Engines may add scale bytes.")
        return rank + rope_dim
    heads = _count(cfg, "num_attention_heads")
    kv_heads = _count(cfg, "num_key_value_heads") or heads
    head_dim = _count(cfg, "head_dim")
    hidden = _count(cfg, "hidden_size")
    if head_dim is None and heads and hidden:
        head_dim = hidden // heads
    if not kv_heads or not head_dim:
        raise SmaError("config.json gives no attention head count or head size, so SMA cannot plan KV memory. "
                       "Check that the repo is a text-generation checkpoint.")
    return 2 * kv_heads * head_dim


def _native_context(cfg: dict, notes: list[str]) -> tuple[int | None, str | None, bool]:
    longest = _count(cfg, "max_position_embeddings")
    stretch = _rope_stretching(cfg)
    if stretch is None:
        if longest is None:
            notes.append("config.json gives no max_position_embeddings.")
            return None, None, False
        return longest, "max_position_embeddings", False
    key, kind, params = stretch
    original, source = _count(params, "original_max_position_embeddings"), f"{key}.original_max_position_embeddings"
    if original is None:
        # Phi-3 style configs keep the original length at the top level.
        original, source = _count(cfg, "original_max_position_embeddings"), "original_max_position_embeddings"
    if original is None:
        # Gemma 3 ships linear scaling as part of the model and names no original length. SMA adds
        # nothing; the config's own maximum is the longest context the model was released for.
        notes.append(f"config.json turns on {kind} RoPE scaling but gives no original length. The native "
                     "context is the config's max_position_embeddings; SMA adds no stretching.")
        if longest is None:
            return None, None, True
        return longest, "max_position_embeddings", True
    stretched_to = f", not the stretched {longest:,}" if longest else ""
    notes.append(f"config.json turns on {kind} RoPE scaling. SMA never stretches RoPE, so the native "
                 f"context is the original {original:,} tokens{stretched_to}.")
    return original, source, True


def _rope_stretching(cfg: dict) -> tuple[str, str, dict] | None:
    """The first RoPE settings that stretch context, as (config key, rope type, parameters)."""
    for key in ("rope_scaling", "rope_parameters"):
        rope = cfg.get(key)
        if not isinstance(rope, dict):
            continue
        # transformers 5 may key rope_parameters by layer type, for example {"full_attention": {...}}.
        for params in [rope, *(value for value in rope.values() if isinstance(value, dict))]:
            kind = params.get("rope_type") or params.get("type")
            if isinstance(kind, str) and kind in _STRETCHING_ROPE:
                return key, kind, params
    return None


def _state_bytes(cfg: dict, kinds: list[str | None], notes: list[str]) -> int:
    """Estimate the fixed state per stream of linear-attention, Mamba, and conv layers."""
    estimators = (
        ("linear", "linear-attention", _linear_attention_state),
        ("mamba", "Mamba", _mamba_state),
        ("conv", "convolution", _conv_state),
    )
    if not any(kind in kinds for kind, _, _ in estimators):
        return 0
    dtype = cfg.get("dtype") or cfg.get("torch_dtype")
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        notes.append("config.json gives no known model dtype. State is sized at 2 bytes per value.")
        dtype = "bfloat16"
    # The recurrent (SSM) state often runs in float32 even when the weights are 16-bit.
    ssm_dtype = cfg.get("mamba_ssm_cache_dtype") or cfg.get("mamba_ssm_dtype") or dtype
    if not isinstance(ssm_dtype, str) or ssm_dtype not in _DTYPE_BYTES:
        notes.append(f"config.json gives no known recurrent state dtype. It is sized as {dtype}.")
        ssm_dtype = dtype

    total = 0
    for kind, label, estimate in estimators:
        layers = kinds.count(kind)
        if not layers:
            continue
        per_layer = estimate(cfg, _DTYPE_BYTES[dtype], _DTYPE_BYTES[ssm_dtype])
        if per_layer is None:
            notes.append(f"{layers} {label} layers keep a fixed state per stream, but config.json does not give "
                         "its size. It is counted as 0 bytes.")
            continue
        total += layers * per_layer
        recurrent = f", recurrent state in {ssm_dtype}" if kind != "conv" else ""
        notes.append(f"{layers} {label} layers keep about {(layers * per_layer) / 1e6:.1f} MB of state per "
                     f"stream. An estimate from config.json{recurrent}.")
    return total


def _linear_attention_state(cfg: dict, dtype_bytes: int, ssm_bytes: int) -> int | None:
    # Gated DeltaNet (Qwen3-Next, Qwen3.5, Qwen3.6): one key x value matrix per value head, plus a
    # short convolution over the q, k, and v projections.
    key_heads = _count(cfg, "linear_num_key_heads")
    value_heads = _count(cfg, "linear_num_value_heads")
    key_dim = _count(cfg, "linear_key_head_dim")
    value_dim = _count(cfg, "linear_value_head_dim")
    kernel = _count(cfg, "linear_conv_kernel_dim")
    if None in (key_heads, value_heads, key_dim, value_dim, kernel):
        return None
    recurrent = value_heads * key_dim * value_dim * ssm_bytes
    conv = (kernel - 1) * (2 * key_heads * key_dim + value_heads * value_dim) * dtype_bytes
    return recurrent + conv


def _mamba_state(cfg: dict, dtype_bytes: int, ssm_bytes: int) -> int | None:
    # Mamba2. Nemotron-H and Granite 4 spell the same settings differently.
    heads = _first_count(cfg, "mamba_num_heads", "mamba_n_heads")
    head_dim = _first_count(cfg, "mamba_head_dim", "mamba_d_head")
    state = _first_count(cfg, "ssm_state_size", "mamba_d_state")
    kernel = _first_count(cfg, "conv_kernel", "mamba_d_conv")
    groups = _first_count(cfg, "n_groups", "mamba_n_groups") or 1
    if None in (heads, head_dim, state, kernel):
        return None
    recurrent = heads * head_dim * state * ssm_bytes
    conv = (kernel - 1) * (heads * head_dim + 2 * groups * state) * dtype_bytes
    return recurrent + conv


def _conv_state(cfg: dict, dtype_bytes: int, ssm_bytes: int) -> int | None:
    # LFM2 short convolution: the last conv_L_cache - 1 inputs of width hidden_size.
    kernel = _count(cfg, "conv_L_cache")
    hidden = _count(cfg, "hidden_size")
    if kernel is None or hidden is None:
        return None
    return (kernel - 1) * hidden * dtype_bytes
