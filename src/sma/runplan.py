"""SMA's own inputs to the memory plan: its benchmarks as needs, and each engine's KV type and overhead.

memplan.py does the arithmetic. This module knows what SMA runs. See docs/serving-design.md.
"""

from __future__ import annotations

from typing import Any

from sma.backends import ServePlan, get_backend
from sma.config import Config
from sma.memplan import SHORT_CONTEXT, KvLayout, MemoryPlan, Need, plan_memory

# Determinism prompts are a few tokens plus determinism.max_tokens. This is the ladder's floor.
_DETERMINISM_MIN_CONTEXT = 1024


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
    return found


def slots(cfg: Config) -> int:
    """llama.cpp's parallel slots: the most requests SMA sends at once, outside the determinism probe."""
    agents = [cfg.kernelbench.concurrency] if cfg.kernelbench.enabled else []
    agents += [cfg.swebench.workers] if cfg.swebench.enabled else []
    return max(cfg.quality.concurrency, *cfg.speed.sweep_concurrency, *agents)


def plan_run(cfg: Config, *, engine: str, layout: KvLayout, weights_bytes: int) -> MemoryPlan:
    """The plan for one run on one engine. Raises SmaError when it does not fit the budget."""
    overhead = int(cfg.memory.overhead_gb[engine] * 1e9)
    if engine == "llamacpp":
        # llama-server's prompt cache lives in host RAM beside the pool (--cache-ram).
        overhead += cfg.engine.llamacpp_cache_ram_mib * 2**20
    return plan_memory(
        layout=layout,
        weights_bytes=weights_bytes,
        kv_type=get_backend(engine).kv_cache_type,
        budget_bytes=int(cfg.memory.budget_gb * 1e9),
        overhead_bytes=overhead,
        needs=needs(cfg),
        start=cfg.memory.start_context,
        pool_overhead_fraction=float(cfg.memory.pool_overhead_fraction[engine]),
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
    )


def describe(plan: dict[str, Any]) -> list[str]:
    """Plain lines for the dry run and the log."""
    return [
        f"KV cache: {plan['kv_type']}, {plan['kv_bytes_per_token'] or 0:,} bytes per token in the full-attention layers",
        f"context per stream: {plan['context']:,} tokens ({plan['context_reason']})",
        f"memory: {plan['weights_gb']} GB weights + {plan['overhead_gb']} GB engine overhead + "
        f"{plan['pool_gb']} GB KV pool"
        + (f" (+{round(plan['pool_overhead_fraction'] * 100)}% engine buffers sized by it)" if plan.get("pool_overhead_fraction") else "")
        + f" = {plan['footprint_gb']} GB of the {plan['budget_gb']} GB budget",
        f"streams that fit: {plan['streams_full_context']} at {plan['context']:,} tokens, "
        f"{plan['streams_32k']} at {min(SHORT_CONTEXT, plan['context']):,}",
        "per benchmark: "
        + ", ".join(f"{name} {context:,}" for name, context in sorted((plan.get("benchmarks") or {}).items())),
    ]
