"""Parse Prometheus text and diff numeric samples."""

from __future__ import annotations

from typing import Any

# Names vLLM documents. Missing names are stored as null, not invented.
WATCHED = (
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:request_time_per_output_token_seconds_sum",
    "vllm:request_time_per_output_token_seconds_count",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_sum",
)


def parse_prometheus(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if " " not in stripped:
            continue
        key, raw = stripped.rsplit(" ", 1)
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        samples[key] = value
    return samples


def diff_samples(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    changes: dict[str, float | None] = {}
    for key in keys:
        if key in before and key in after:
            changes[key] = after[key] - before[key]
        else:
            changes[key] = None
    watched = {name: changes.get(name) for name in WATCHED}
    return {"watched": watched, "counters": changes}
