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


def _engine_name(manifest: dict[str, Any]) -> str:
    engine = manifest.get("engine")
    if isinstance(engine, dict):
        return str(engine.get("name") or "vllm")
    if isinstance(engine, str) and engine:
        return engine
    return "vllm"


def compare_run_dirs(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    allow_cross_engine: bool = False,
) -> dict[str, Any]:
    reference = load_manifest(reference_dir)
    candidate = load_manifest(candidate_dir)
    if "llamacpp" in (_engine_name(reference), _engine_name(candidate)):
        raise CompareRefused(
            "A llama.cpp run has no per-position logprob dump: llama-server returns no prompt logprobs. "
            "Its KL comes from llama-perplexity during the run. Start the quant with --reference <full-precision "
            "GGUF run id> and read logprobs/kl_vs_reference.json."
        )
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
    ref_engine = _engine_name(reference)
    cand_engine = _engine_name(candidate)
    if ref_engine != cand_engine and not allow_cross_engine:
        raise CompareRefused(
            f"Refusing to compare {ref_engine} with {cand_engine}. "
            "Engine is part of the subject. Pass --allow-cross-engine to write a labeled "
            "comparison. Those numbers are not a quant comparison."
        )
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
    cross = ref_engine != cand_engine
    if cross:
        notes.append(
            f"CROSS-ENGINE comparison of {ref_engine} vs {cand_engine}. "
            "This is not a quant comparison and must stay labeled."
        )
    ref_digest = _image_digest(reference)
    cand_digest = _image_digest(candidate)
    if ref_digest and cand_digest and ref_digest != cand_digest:
        notes.append(
            "IMAGE DIGEST differs "
            f"({ref_digest} vs {cand_digest}). These runs did not use the same image."
        )
    elif not ref_digest or not cand_digest:
        notes.append(
            "One or both runs have no recorded image digest, so image identity was not checked."
        )
    result["reference_run_id"] = reference["run_id"]
    result["candidate_run_id"] = candidate["run_id"]
    result["tokenizer_fingerprint"] = ref_fp
    result["reference_quant"] = reference.get("quant_label") or reference.get("quant_flag")
    result["candidate_quant"] = candidate.get("quant_label") or candidate.get("quant_flag")
    result["reference_engine"] = ref_engine
    result["candidate_engine"] = cand_engine
    result["engine"] = ref_engine if not cross else "cross-engine"
    result["cross_engine"] = cross
    result["reference_image_digest"] = ref_digest
    result["candidate_image_digest"] = cand_digest
    result["image_digest_match"] = (
        ref_digest == cand_digest if ref_digest and cand_digest else None
    )
    result["notes"] = notes
    return result


def _image_digest(manifest: dict[str, Any]) -> str | None:
    engine = manifest.get("engine")
    if isinstance(engine, dict):
        digest = engine.get("image_digest")
        if isinstance(digest, str) and digest:
            return digest
    return None


def write_compare(runs_dir: Path, result: dict[str, Any]) -> Path:
    ref = result["reference_run_id"]
    cand = result["candidate_run_id"]
    path = runs_dir / "compares" / f"{ref}__{cand}.json"
    atomic_write_json(path, result)
    return path
