"""Derived speed numbers. Null when the bench JSON does not contain the stat."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_mapping(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[-1], dict):
        return payload[-1]
    return None


def load_bench(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _as_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return None


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
    """Headline tok/s from valid cells. Invalid cells stay in the list and do not contribute."""
    valid = [cell for cell in cells if cell.get("valid")]

    def _pick(shape: str, concurrency: int, key: str) -> float | None:
        chosen = [
            float(cell[key])
            for cell in valid
            if cell.get("shape") == shape
            and cell.get("concurrency") == concurrency
            and isinstance(cell.get(key), (int, float))
        ]
        return median(chosen)

    invalid = sum(1 for cell in cells if not cell.get("valid"))
    return {
        "prefill_tok_s": _pick("prefill", 1, "prefill_tok_s"),
        "decode_tok_s": _pick("decode", 1, "decode_tok_s"),
        "output_tok_s_c4": _pick("sweep", 4, "output_tok_s"),
        "output_tok_s_c8": _pick("sweep", 8, "output_tok_s"),
        "invalid_cells": invalid,
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
