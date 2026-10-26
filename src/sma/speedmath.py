"""Derived speed numbers. Null when the bench JSON does not contain the stat."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _as_mapping(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[-1], dict):
        return payload[-1]
    return None


def _last_json_object(text: str) -> dict[str, Any] | None:
    """SGLang appends one JSON object per line. A retry must not break the parse."""
    last: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        mapped = _as_mapping(parsed)
        if mapped is not None:
            last = mapped
    return last


def load_bench(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    payload: dict[str, Any] | None
    try:
        payload = _as_mapping(json.loads(stripped))
    except json.JSONDecodeError:
        payload = _last_json_object(stripped)
    if payload is None:
        return None
    return normalize_bench(payload)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _seconds_to_ms(values: Any) -> list[float]:
    """SGLang stores per-request ttft/itl in seconds. See bench_serving.py, which multiplies by 1000 for *_ms."""
    if not isinstance(values, list):
        return []
    numbers: list[float] = []
    for item in values:
        if isinstance(item, list):
            numbers.extend(_seconds_to_ms(item))
            continue
        number = finite(item)
        if number is not None:
            numbers.append(number * 1000.0)
    return numbers


def _per_request_tpot_ms(itls: Any) -> list[float]:
    """TPOT is the mean inter-token gap of one request. SGLang uses (latency - ttft) / (output_len - 1), in seconds."""
    if not isinstance(itls, list):
        return []
    means: list[float] = []
    for item in itls:
        if isinstance(item, list):
            gaps = [finite(value) for value in item]
            usable = [value * 1000.0 for value in gaps if value is not None]
        else:
            number = finite(item)
            usable = [] if number is None else [number * 1000.0]
        if usable:
            means.append(sum(usable) / len(usable))
    return means


def normalize_bench(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy SGLang's result keys onto the p50/p90/p99 names SMA already reads.

    Checked against sglang tag v0.5.10.post1, python/sglang/bench_serving.py
    (the result dict written as JSONL). That build emits median and p99 for
    TTFT and TPOT, median/p95/p99 for ITL, and median/p90/p99 for e2e latency
    (`median_e2e_latency_ms`, `p90_e2e_latency_ms`, `p99_e2e_latency_ms`).
    It does not emit p90 TTFT, TPOT, or ITL. Those are computed from
    `--output-details` arrays when present, and otherwise stay null.
    """
    out = dict(payload)
    notes: list[str] = []
    aliases = {
        "p50_ttft_ms": "median_ttft_ms",
        "p99_ttft_ms": "p99_ttft_ms",
        "p50_tpot_ms": "median_tpot_ms",
        "p99_tpot_ms": "p99_tpot_ms",
        "p50_itl_ms": "median_itl_ms",
        "p99_itl_ms": "p99_itl_ms",
        "p50_e2el_ms": "median_e2e_latency_ms",
        "p90_e2el_ms": "p90_e2e_latency_ms",
        "p99_e2el_ms": "p99_e2e_latency_ms",
    }
    for canonical, source in aliases.items():
        if finite(out.get(canonical)) is None and finite(out.get(source)) is not None:
            out[canonical] = out[source]
    derived = {
        "p90_ttft_ms": _percentile(_seconds_to_ms(out.get("ttfts")), 90),
        "p90_tpot_ms": _percentile(_per_request_tpot_ms(out.get("itls")), 90),
        "p90_itl_ms": _percentile(_seconds_to_ms(out.get("itls")), 90),
    }
    for key, value in derived.items():
        if finite(out.get(key)) is None:
            if value is None:
                notes.append(
                    f"{key} is not in the SGLang summary and per-request data was absent, so it is null."
                )
            else:
                out[key] = value
                notes.append(f"{key} was computed from per-request arrays, not copied from a summary field.")
    if notes:
        out["percentile_notes"] = notes
    return out


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def percentile_ms(payload: dict[str, Any], metric: str, percentile: int) -> float | None:
    """Find TTFT/TPOT/ITL/E2EL in the shapes vLLM's save-result JSON has used."""
    flat_keys = [
        f"p{percentile}_{metric}_ms",
        f"{metric}_p{percentile}_ms",
    ]
    if percentile == 50:
        flat_keys.extend([f"median_{metric}_ms", f"{metric}_median_ms"])
    for key in flat_keys:
        value = payload.get(key)
        number = finite(value)
        if number is not None:
            return number
    list_key = f"percentiles_{metric}_ms"
    entries = payload.get(list_key)
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, (list, tuple)) and len(item) == 2 and int(item[0]) == percentile:
                number = finite(item[1])
                if number is not None:
                    return number
            if isinstance(item, dict) and int(item.get("percentile", -1)) == percentile:
                value = item.get("ms", item.get("value"))
                number = finite(value)
                if number is not None:
                    return number
    nested = payload.get(metric)
    if isinstance(nested, dict):
        for key in (f"p{percentile}", str(percentile), "median" if percentile == 50 else ""):
            number = finite(nested.get(key)) if key else None
            if number is not None:
                return number
    return None


def output_tok_s(payload: dict[str, Any]) -> float | None:
    for key in ("output_throughput", "output_tok_s", "output_tokens_per_second"):
        number = finite(payload.get(key))
        if number is not None:
            return number
    return None


def prefill_tok_s(input_len: int, ttft_ms: float | None) -> float | None:
    """Input tokens / TTFT. TTFT is in milliseconds in the bench JSON."""
    if ttft_ms is None or ttft_ms <= 0 or input_len <= 0:
        return None
    return input_len / (ttft_ms / 1000.0)


def decode_tok_s(tpot_ms: float | None) -> float | None:
    """1 / TPOT. TPOT is in milliseconds."""
    if tpot_ms is None or tpot_ms <= 0:
        return None
    return 1.0 / (tpot_ms / 1000.0)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def summarize_cells(cells: list[dict[str, Any]], *, stat: str) -> dict[str, Any]:
    """Headline tok/s from valid cells. Invalid and unverified cells stay in the list."""
    valid = [cell for cell in cells if cell.get("valid") and cell.get("validity", "valid") == "valid"]

    def _pick(shape: str, concurrency: int, key: str) -> float | None:
        chosen = [
            float(cell[key])
            for cell in valid
            if cell.get("shape") == shape
            and cell.get("concurrency") == concurrency
            and isinstance(cell.get(key), (int, float))
        ]
        return median(chosen)

    invalid = sum(
        1
        for cell in cells
        if cell.get("validity") == "invalid" or (cell.get("validity") is None and not cell.get("valid"))
    )
    unverified = sum(1 for cell in cells if cell.get("validity") == "unverified")
    return {
        "prefill_tok_s": _pick("prefill", 1, "prefill_tok_s"),
        "decode_tok_s": _pick("decode", 1, "decode_tok_s"),
        "output_tok_s_c4": _pick("sweep", 4, "output_tok_s"),
        "output_tok_s_c8": _pick("sweep", 8, "output_tok_s"),
        "invalid_cells": invalid,
        "unverified_cells": unverified,
        "derive_stat": stat,
        "cells": cells,
    }


def bandwidth_ceiling_tok_s(
    active_params: int | None,
    quant_label: str,
    bandwidth_gbps: float,
) -> tuple[float | None, str]:
    """Spec ceiling, not a measurement. NVFP4 scale bytes are not included."""
    note = (
        f"Derived from the configured spec bandwidth ({bandwidth_gbps} GB/s) divided by "
        "approximate active weight bytes. Not a measurement. NVFP4/MXFP4 use 0.5 bytes "
        "per parameter and ignore scale overhead."
    )
    if active_params is None or active_params <= 0 or bandwidth_gbps <= 0:
        return None, note
    bytes_per = {
        "bf16": 2.0,
        "nvfp4": 0.5,
        "nvfp4_w4a16": 0.5,
        "mxfp4": 0.5,
    }.get(quant_label)
    if bytes_per is None:
        return None, note
    return (bandwidth_gbps * 1_000_000_000.0) / (active_params * bytes_per), note
