"""Determinism probe and prompt_logprobs parsing.

Logprobs use the running server's completions API with raw token ids.
A second offline LLM() process is not started: the quality server already
holds the GPU. No chat template is applied.
"""

from __future__ import annotations

import math
from typing import Any

from sma.errors import SmaError
from sma.httpapi import request_json
from sma.kl import probability


def run_determinism(
    *,
    base_url: str,
    served_name: str,
    prompts: list[str],
    batch_sizes: list[int],
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    if not batch_sizes:
        raise SmaError("determinism.batch_sizes is empty")
    baseline: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    passed = True
    for prompt in prompts:
        for batch in batch_sizes:
            body = request_json(
                "POST",
                base_url.rstrip("/") + "/v1/completions",
                timeout_s=timeout_s,
                payload={
                    "model": served_name,
                    "prompt": [prompt] * batch,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                    "stream": False,
                },
            )
            choices = body.get("choices") if isinstance(body, dict) else None
            texts = []
            if isinstance(choices, list):
                texts = [str(choice.get("text", "")) for choice in choices if isinstance(choice, dict)]
            self_match = len(texts) == batch and len(set(texts)) == 1
            if batch == batch_sizes[0]:
                baseline[prompt] = texts[0] if texts else ""
            base_match = self_match and texts[0] == baseline.get(prompt)
            if not base_match:
                passed = False
            rows.append(
                {
                    "prompt": prompt,
                    "batch_size": batch,
                    "outputs": texts,
                    "self_match": self_match,
                    "matches_batch_1": base_match,
                }
            )
    return {
        "schema_version": 1,
        "passed": passed,
        "method": "single_request_prompt_list",
        "note": (
            "Each batch size is one completions request whose prompt is that many "
            "copies of the same string, temperature 0. The client cannot force the "
            "scheduler to fuse them, so max-num-seqs is set at least as high as the "
            "largest batch. A failure is recorded and does not stop the run."
        ),
        "batch_sizes": list(batch_sizes),
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "results": rows,
    }


def _logprob_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("logprob"), (int, float)):
        return float(value["logprob"])
    return None


def parse_prompt_logprobs(
    body: dict[str, Any],
    token_ids: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise SmaError("completions response did not include choices")
    raw = choices[0].get("prompt_logprobs")
    if not isinstance(raw, list):
        raise SmaError(
            "completions response has no prompt_logprobs. "
            "The server needs --max-logprobs at least as large as logprobs.top_k."
        )
    positions: list[dict[str, Any]] = []
    for index in range(1, len(token_ids)):
        entry = raw[index] if index < len(raw) else None
        top: list[dict[str, Any]] = []
        true_logprob = None
        if isinstance(entry, dict):
            for key, value in entry.items():
                try:
                    token_id = int(key)
                except (TypeError, ValueError):
                    continue
                logprob = _logprob_value(value)
                if logprob is None or not math.isfinite(logprob):
                    continue
                top.append({"id": token_id, "logprob": logprob})
                if token_id == token_ids[index]:
                    true_logprob = logprob
        top.sort(key=lambda item: (-float(item["logprob"]), int(item["id"])))
        top = top[:top_k]
        mass = sum(probability(float(item["logprob"])) for item in top)
        tail = max(0.0, min(1.0, 1.0 - mass))
        positions.append(
            {
                "index": index,
                "true_token_id": token_ids[index],
                "true_logprob": true_logprob,
                "top": top,
                "tail_mass": tail,
                "tail_logprob": math.log(tail) if tail > 0 else None,
            }
        )
    return positions


def request_window_logprobs(
    *,
    base_url: str,
    served_name: str,
    token_ids: list[int],
    top_k: int,
    seed: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    body = request_json(
        "POST",
        base_url.rstrip("/") + "/v1/completions",
        timeout_s=timeout_s,
        payload={
            "model": served_name,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "seed": seed,
            "prompt_logprobs": top_k,
            "stream": False,
        },
    )
    if not isinstance(body, dict):
        raise SmaError("logprob response was not a JSON object")
    return parse_prompt_logprobs(body, token_ids, top_k)
