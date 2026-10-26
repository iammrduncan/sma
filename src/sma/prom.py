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


def metric_name(key: str) -> str:
    """Drop Prometheus labels. vLLM emits model_name, engine, and similar."""
    return key.split("{", 1)[0].strip()


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


def _by_name(samples: dict[str, float]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for key, value in samples.items():
        name = metric_name(key)
        grouped[name] = grouped.get(name, 0.0) + value
    return grouped


def diff_samples(
    before: dict[str, float],
    after: dict[str, float],
    *,
    watched: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    before_named = _by_name(before)
    after_named = _by_name(after)
    names = sorted(set(before_named) | set(after_named))
    changes: dict[str, float | None] = {}
    for name in names:
        if name in before_named and name in after_named:
            changes[name] = after_named[name] - before_named[name]
        else:
            changes[name] = None
    wanted = watched if watched is not None else WATCHED
    watched_values = {name: changes.get(name) for name in wanted}
    missing = [name for name, value in watched_values.items() if value is None]
    reason = None
    if missing:
        reason = (
            "These counters were missing from /metrics and are null: "
            + ", ".join(missing)
            + ". Nothing was substituted."
        )
    return {"watched": watched_values, "counters": changes, "missing_reason": reason}
