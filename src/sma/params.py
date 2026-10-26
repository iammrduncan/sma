"""Parameter counts from safetensors shapes, plus the arena size class."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from sma.safetensors import read_header

_EXPERT_TOKEN = ".experts."


def is_routed_expert(name: str) -> bool:
    if ".shared_expert" in name:
        return False
    return _EXPERT_TOKEN in name or name.startswith("experts.")


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def count_header(header: dict[str, Any]) -> tuple[int, int, int]:
    """Return (total, routed_expert, other) parameter counts."""
    total = 0
    expert = 0
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        shape = meta.get("shape")
        if not isinstance(shape, list):
            continue
        count = _numel(shape)
        total += count
        if is_routed_expert(str(name)):
            expert += count
    return total, expert, total - expert


def count_shards(paths: list[Path]) -> tuple[int, int, int]:
    total = expert = other = 0
    for path in paths:
        shard_total, shard_expert, shard_other = count_header(read_header(path))
        total += shard_total
        expert += shard_expert
        other += shard_other
    return total, expert, other


def active_params(
    *,
    total: int,
    expert: int,
    other: int,
    num_experts: int | None,
    experts_per_tok: int | None,
    is_moe: bool,
) -> tuple[int | None, str]:
    if not is_moe:
        return total, "dense: active = total, embeddings included, from safetensors shapes"
    if expert == 0 or not num_experts or not experts_per_tok:
        return None, (
            "moe: active params unavailable because routed expert tensors or "
            "num_experts / num_experts_per_tok were missing. total is still recorded"
        )
    if experts_per_tok > num_experts:
        return None, "moe: num_experts_per_tok is greater than num_experts; active params not guessed"
    active = other + int(round(expert * (experts_per_tok / num_experts)))
    method = (
        "moe: active = non_expert + expert * (num_experts_per_tok / num_experts); "
        "shared experts stay in non_expert; source = safetensors shapes"
    )
    return active, method


def billions(count: int) -> float:
    """Round to 0.1B with half-up, which is the size-class input."""
    quantized = (Decimal(count) / Decimal(1_000_000_000)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    return float(quantized)


def size_class(total_params: int) -> str:
    """Class by total params. Boundaries include their upper bound."""
    rounded = billions(total_params)
    if rounded < 1.0:
        return "<1B"
    if rounded <= 3.0:
        return "1-3B"
    if rounded <= 8.0:
        return "3-8B"
    if rounded <= 15.0:
        return "8-15B"
    if rounded <= 20.0:
        return "15-20B"
    if rounded <= 34.0:
        return "20-34B"
    return ">34B"
