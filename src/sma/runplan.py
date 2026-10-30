"""SMA's own inputs to the memory plan: its benchmarks as needs, and each engine's KV type and overhead.

memplan.py does the arithmetic. This module knows what SMA runs. See docs/serving-design.md.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from sma.backends import ServePlan, get_backend
from sma.config import Config
from sma.memplan import SHORT_CONTEXT, KvLayout, MemoryPlan, Need, plan_memory

# Determinism prompts are a few tokens plus determinism.max_tokens. This is the ladder's floor.
_DETERMINISM_MIN_CONTEXT = 1024

# llama-server's micro-batch (-ub), its default. SMA does not set it.
_LLAMACPP_UBATCH = 512


def needs(cfg: Config) -> list[Need]:
    """Each benchmark SMA runs: the streams it runs at once, and the least context each stream needs."""
    shapes = (cfg.speed.prefill, cfg.speed.decode, cfg.speed.sweep)
    found = [
        Need("determinism", max(cfg.determinism.batch_sizes), _DETERMINISM_MIN_CONTEXT),
        Need("logprobs", 1, cfg.logprobs.window_tokens),
        Need("quality", cfg.quality.concurrency, cfg.memory.quality_min_context),
        Need("speed", max(cfg.speed.sweep_concurrency), max(shape.input_len + shape.output_len for shape in shapes)),
    ]
    if cfg.kernelbench.enabled:
        # A longer answer than its step still runs; the server may then queue it behind the others.
        found.append(Need("kernelbench", cfg.kernelbench.concurrency, cfg.memory.quality_min_context))
    if cfg.swebench.enabled:
        found.append(Need("swebench", cfg.swebench.workers, cfg.swebench.min_context))
    if cfg.browsecomp.enabled:
        found.append(Need("browsecomp", cfg.browsecomp.workers, cfg.browsecomp.min_context))
    return found


def slots(cfg: Config) -> int:
    """llama.cpp's parallel slots: the most requests SMA sends at once, outside the determinism probe."""
    agents = [cfg.kernelbench.concurrency] if cfg.kernelbench.enabled else []
    agents += [cfg.swebench.workers] if cfg.swebench.enabled else []
    agents += [cfg.browsecomp.workers] if cfg.browsecomp.enabled else []
    return max(cfg.quality.concurrency, *cfg.speed.sweep_concurrency, *agents)


def llamacpp_buffer_bytes_per_token(layout: KvLayout) -> int:
    """llama-server's compute buffers per token of its KV cache. It sizes them by the whole cache (-c):
    the 16-bit attention mask for one micro-batch, on the GPU and again on the host, and one attention
    layer's K and V converted to 16 bits for flash attention.

    Measured on a Spark (llama.cpp's memory breakdown, GPU plus host compute buffers over cells):
    Qwen3-0.6B 6,164 bytes per token (this gives 6,144), LFM3-1.7B BF16 and QAD-Q4_0 4,100 (this gives
    4,096).
    """
    layers = layout.full_attention_layers or layout.sliding_attention_layers
    values = (layout.full_values_per_token or layout.sliding_values_per_token) // layers if layers else 0
    return 2 * 2 * _LLAMACPP_UBATCH + 2 * values


def plan_run(cfg: Config, *, engine: str, layout: KvLayout, weights_bytes: int) -> MemoryPlan:
    """The plan for one run on one engine. Raises SmaError when it does not fit the budget."""
    overhead = int(cfg.memory.overhead_gb[engine] * 1e9)
    buffer_bytes_per_token, max_streams = 0, None
    if engine == "sglang":
        layout = sglang_layout(layout)
    if engine == "llamacpp":
        # llama-server's prompt cache lives in host RAM beside the pool (--cache-ram).
        overhead += cfg.engine.llamacpp_cache_ram_mib * 2**20
        # Its compute buffers grow with the whole cache, and its slots never use more than slots x
        # context. A pool of the rest of the budget held 11.6M tokens for LFM3-1.7B, 5.5 times what 8
        # slots use, and its buffers took 47 GB beyond the plan.
        buffer_bytes_per_token, max_streams = llamacpp_buffer_bytes_per_token(layout), slots(cfg)
    return plan_memory(
        layout=layout,
        weights_bytes=weights_bytes,
        kv_type=get_backend(engine).kv_cache_type,
        budget_bytes=int(cfg.memory.budget_gb * 1e9),
        overhead_bytes=overhead,
        needs=needs(cfg),
        start=cfg.memory.start_context,
        buffer_bytes_per_token=buffer_bytes_per_token,
        max_streams=max_streams,
    )


def sglang_layout(layout: KvLayout) -> KvLayout:
    """The layout SGLang allocates: its pool gives sliding-window layers K and V for every pooled token,
    not only the window. On a Spark (dev-lfm3, 2026-10-30) it gave LFM3-1.7B's 5 sliding layers full length
    beside its 6 full-attention layers: 11,264 bytes a token, where the plan counted 6,144, so 9.39M
    tokens took 98.5 GB and left the Spark 9 GB. Where SGLang runs its hybrid sliding-window pool, those
    layers still hold 0.8 of the pooled tokens (swa_full_tokens_ratio); counting them in full stays safe."""
    if not layout.sliding_attention_layers:
        return layout
    return replace(
        layout,
        full_attention_layers=layout.full_attention_layers + layout.sliding_attention_layers,
        full_values_per_token=layout.full_values_per_token + layout.sliding_values_per_token,
        sliding_attention_layers=0,
        sliding_values_per_token=0,
        sliding_window=None,
        notes=(*layout.notes, "SGLang keeps the sliding-window layers at full length, so they count as full attention."),
    )


def serve_plan(cfg: Config, plan: dict[str, Any] | None, *, model_file: str | None = None) -> ServePlan:
    """The serve command's view of a recorded plan (MemoryPlan.to_json). Placeholders when it is missing."""
    if not isinstance(plan, dict):
        return ServePlan("<context>", "<kv-pool-bytes>", "<kv-pool-tokens>", model_file or "<file.gguf>", slots(cfg))
    tokens = plan.get("pool_tokens")
    if not isinstance(tokens, int):
        # A model with no attention layers: the pool is counted in whole streams.
        tokens = int(plan["context"]) * max(1, int(plan.get("streams_full_context") or 1))
    return ServePlan(
        context=int(plan["context"]),
        pool_bytes=int(plan["pool_bytes"]),
        pool_tokens=tokens,
        model_file=model_file,
        slots=slots(cfg),
        weights_bytes=int(plan["weights_bytes"]) if isinstance(plan.get("weights_bytes"), int) else None,
    )


def describe(plan: dict[str, Any]) -> list[str]:
    """Plain lines for the dry run and the log."""
    return [
        f"KV cache: {plan['kv_type']}, {plan['kv_bytes_per_token'] or 0:,} bytes per token in the full-attention layers",
        f"context per stream: {plan['context']:,} tokens ({plan['context_reason']})",
        f"memory: {plan['weights_gb']} GB weights + {plan['overhead_gb']} GB engine overhead + "
        f"{plan['pool_gb']} GB KV pool"
        + (f" + {plan['buffer_gb']} GB engine buffers sized by it" if plan.get("buffer_bytes") else "")
        + f" = {plan['footprint_gb']} GB of the {plan['budget_gb']} GB budget",
        *(
            [f"KV pool: {plan['pool_streams']} slots at {plan['context']:,} tokens, the most the server runs at once"]
            if plan.get("pool_streams")
            else []
        ),
        f"streams that fit: {plan['streams_full_context']} at {plan['context']:,} tokens, "
        f"{plan['streams_32k']} at {min(SHORT_CONTEXT, plan['context']):,}",
        "per benchmark: "
        + ", ".join(f"{name} {context:,}" for name, context in sorted((plan.get("benchmarks") or {}).items())),
    ]
