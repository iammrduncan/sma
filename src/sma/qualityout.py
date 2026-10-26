"""Normalize lm-eval output into scores.json and samples.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sma.atomic import atomic_write_bytes, atomic_write_json, atomic_write_text
from sma.errors import SmaError

_SCORE_HINTS = ("acc", "exact_match", "inst_level", "prompt_level", "f1", "em")


def primary_score(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    candidates: list[tuple[str, float]] = []
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if "stderr" in key:
            continue
        if any(hint in key for hint in _SCORE_HINTS):
            candidates.append((key, float(value)))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (0 if "strict" in item[0] else 1, item[0]))
    return candidates[0]


def normalize_suite(output_dir: Path, suite_name: str, *, limit: int | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.json"
    if not scores_path.exists():
        found = sorted(output_dir.rglob("results.json"))
        if not found:
            raise SmaError(
                f"lm-eval did not write results.json for {suite_name} under {output_dir}"
            )
        raw = json.loads(found[0].read_text(encoding="utf-8"))
        results = raw.get("results") if isinstance(raw, dict) else None
        suite_blob: dict[str, Any] = {}
        if isinstance(results, dict) and results:
            if isinstance(results.get(suite_name), dict):
                suite_blob = results[suite_name]
            else:
                first = next(iter(results.values()))
                suite_blob = first if isinstance(first, dict) else {}
        metric, score = primary_score(suite_blob)
        atomic_write_json(
            scores_path,
            {
                "scorer": "lm-eval-harness",
                "suite": suite_name,
                "primary_metric": metric,
                "primary_score": score,
                "limit": limit,
                "results": results if isinstance(results, dict) else None,
            },
        )
    samples_path = output_dir / "samples.jsonl"
    if not samples_path.exists():
        found_samples = sorted(output_dir.rglob("*.jsonl"))
        if found_samples:
            atomic_write_bytes(samples_path, found_samples[0].read_bytes())
        else:
            atomic_write_text(samples_path, "")
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmaError(f"{scores_path} is not a JSON object")
    return payload
