"""Compare two vLLM runs of the same tokenizer. Refuse when they are not comparable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sma.atomic import atomic_write_json
from sma.errors import CompareRefused
from sma.kl import compare_dumps
from sma.records import load_manifest


def _dump(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "logprobs" / "dump.json"
    if not path.is_file():
        raise CompareRefused(f"{run_dir.name} has no logprobs/dump.json. The logprob step has not finished.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompareRefused(f"{path} is not a JSON object")
    return payload


def compare_run_dirs(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference = load_manifest(reference_dir)
    candidate = load_manifest(candidate_dir)
    ref_fp = reference.get("tokenizer_fingerprint")
    cand_fp = candidate.get("tokenizer_fingerprint")
    if not ref_fp or not cand_fp or ref_fp != cand_fp:
        raise CompareRefused(
            "Tokenizers differ or a fingerprint is missing. "
            "Refusing KL, top-1 agreement, and delta perplexity. "
            "Both runs must score the same token ids."
        )
    ref_dump = _dump(reference_dir)
    cand_dump = _dump(candidate_dir)
    if ref_dump.get("corpus_version") != cand_dump.get("corpus_version"):
        raise CompareRefused(
            "Corpus versions differ "
            f"({ref_dump.get('corpus_version')} vs {cand_dump.get('corpus_version')}). Refusing KL."
        )
    if ref_dump.get("window_tokens") != cand_dump.get("window_tokens"):
        raise CompareRefused("Logprob window lengths differ. Refusing KL.")
    if ref_dump.get("top_k") != cand_dump.get("top_k"):
        raise CompareRefused(
            f"top-k differs ({ref_dump.get('top_k')} vs {cand_dump.get('top_k')}). "
            "Refusing KL so the recorded k stays meaningful."
        )
    cand_windows = {int(item["index"]): item for item in cand_dump.get("windows") or []}
    for window in ref_dump.get("windows") or []:
        other = cand_windows.get(int(window["index"]))
        if other is None or list(other.get("token_ids") or []) != list(window.get("token_ids") or []):
            raise CompareRefused(
                "The fixed token-id windows differ. Refusing KL. "
                "Tokenization must be identical or the divergence is meaningless."
            )
    result = compare_dumps(ref_dump, cand_dump)
    if result.get("token_sequence_mismatches"):
        raise CompareRefused(
            "True next-token ids differ between the dumps. Refusing KL."
        )
    notes: list[str] = []
    if reference.get("architecture") != candidate.get("architecture"):
        notes.append(
            "Architectures differ. The tokenizer matched, so KL was computed, "
            "but these runs may not be the same base model."
        )
    if reference.get("engine", {}).get("name") != "vllm" or candidate.get("engine", {}).get("name") != "vllm":
        notes.append("A run is not labeled vllm. Do not compare it with vLLM rows.")
    result["reference_run_id"] = reference["run_id"]
    result["candidate_run_id"] = candidate["run_id"]
    result["tokenizer_fingerprint"] = ref_fp
    result["reference_quant"] = reference.get("quant_label") or reference.get("quant_flag")
    result["candidate_quant"] = candidate.get("quant_label") or candidate.get("quant_flag")
    result["engine"] = "vllm"
    result["notes"] = notes
    return result


def write_compare(runs_dir: Path, result: dict[str, Any]) -> Path:
    ref = result["reference_run_id"]
    cand = result["candidate_run_id"]
    path = runs_dir / "compares" / f"{ref}__{cand}.json"
    atomic_write_json(path, result)
    return path
