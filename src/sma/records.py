"""Manifest, summary, and the on-disk run directory."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sma import __version__
from sma.atomic import atomic_write_json
from sma.catalog import Catalog
from sma.commands import moe_backend_chain
from sma.config import Config
from sma.errors import SmaError
from sma.hashing import sha256_file
from sma.schemautil import validate_instance
from sma.scrub import scrub_obj, scrub_text
from sma.speedmath import bandwidth_ceiling_tok_s
from sma.telemetry import integrate_joules, peak_memory_bytes, read_memory, read_samples

STEP_NAMES = [
    "fetch",
    "serve_quality",
    "determinism",
    "logprobs",
    "quality",
    "serve_speed",
    "speed",
    "telemetry",
    "teardown",
    "cleanup",
]

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")

# Files too big for git. .gitignore lists the same paths. The manifest keeps each one's sha256 and
# size, so a copy kept elsewhere can be checked against the committed record.
LARGE_FILES = ("logprobs/dump.json", "logprobs/kld-base.bin")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def model_slug(repo: str) -> str:
    slug = _SLUG.sub("_", repo).strip("._-")
    if not slug:
        raise SmaError(f"Repository {repo!r} does not make a usable directory name")
    return slug


def git_identity() -> tuple[str | None, bool]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None, False
    dirty = False
    try:
        probe = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet"],
            timeout=5,
            check=False,
        )
        dirty = probe.returncode != 0
    except (OSError, subprocess.SubprocessError):
        dirty = False
    return sha or None, dirty


def _blank_step() -> dict[str, Any]:
    return {
        "status": "pending",
        "started": None,
        "finished": None,
        "error": None,
        "attempts": 0,
    }


def new_manifest(
    *,
    cfg: Config,
    repo: str,
    quant_flag: str,
    revision: str | None,
    run_id: str,
    reference_run_id: str | None,
    pair_job_id: str | None,
    engine: str = "vllm",
    tool_call_parser: str | None = None,
    gguf: str | None = None,
    selected_by: str | None = None,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    from sma.backends import get_backend
    from sma.engines import configured_digest, image_for

    commit, dirty = git_identity()
    now = utc_now()
    if engine == "llamacpp":
        # llama-server parses tool calls with the model's own chat template (--jinja). No parser flag.
        tool_call_parser = "llama.cpp --jinja"
    return {
        "schema_version": 2,
        "run_id": run_id,
        "model_repo": repo,
        "model_slug": model_slug(repo),
        "revision_requested": revision,
        "revision_sha": None,
        "quant_flag": quant_flag,
        "quant_label": gguf,
        "quant_detected": None,
        "quant_detail": None,
        "quant_producer": None,
        "quant_composition": None,
        "architecture_check": None,
        "architecture": None,
        "architectures": [],
        "is_moe": None,
        "params": {
            "total": None,
            "active": None,
            "total_b": None,
            "active_b": None,
            "size_class": None,
            "method": None,
            "source": None,
        },
        "file_hashes": {},
        "tokenizer_fingerprint": None,
        "subject": {
            "model_repo": repo,
            "revision_sha": None,
            "quant": quant_flag,
            "engine": engine,
            "hardware_alias": cfg.host_alias,
        },
        "engine": {
            "name": engine,
            "selected_by": selected_by or "default",
            "selection_reason": selection_reason,
            "image": image_for(cfg, engine),
            "image_digest": configured_digest(cfg, engine),
            "build_info": None,
            "bench_image": None,
            "kv_cache_type": get_backend(engine).kv_cache_type,
            "kv_cache_scales": None,
            "kv_cache_reported": {},
            "context_per_stream": None,
            "tensor_parallel_size": 1,
            "moe_backend_config": cfg.engine.moe_backend,
            "moe_backend_chain": moe_backend_chain(cfg),
            "moe_backend_used": None,
            "moe_backend_fallback": False,
            "moe_backend_quality": None,
            "moe_backend_speed": None,
            "moe_backend_quality_fallback": False,
            "moe_backend_speed_fallback": False,
            "linear_backend": None,
            "batch_invariant_quality": False,
            "batch_invariant_supported": None,
            "batch_invariant_warning": None,
            "flashinfer_autotune_quality": not cfg.engine.no_flashinfer_autotune_quality,
            "no_async_scheduling_quality": bool(
                cfg.engine.batch_invariant and cfg.engine.no_async_scheduling_with_batch_invariant
            ),
            "seed": cfg.seed,
            "max_logprobs": cfg.logprobs.top_k,
            "logprobs_mode": {
                "sglang": "sglang_input_top_logprobs",
                "llamacpp": "llama-perplexity full-vocabulary KL",
            }.get(engine, "raw_logprobs"),
            "flags_quality": [],
            "flags_speed": [],
            "env_quality": {},
            "env_speed": {},
            "clock_cap_mhz": cfg.engine.clock_cap_mhz,
            "resolved_backend_note": None,
            "deterministic_quality": _deterministic_quality(cfg, engine),
            "tool_call_parser": tool_call_parser,
        },
        "sma_version": __version__,
        "sma_git_commit": commit,
        "sma_git_dirty": dirty,
        "host_alias": cfg.host_alias,
        "timestamps": {"created": now, "started": None, "finished": None},
        "steps": {name: _blank_step() for name in STEP_NAMES},
        "warnings": [],
        "reference_run_id": reference_run_id,
        "pair_job_id": pair_job_id,
        "dispatch_concurrency": cfg.quality.concurrency,
        "config_snapshot": scrub_obj(cfg.to_snapshot()),
        # Filled at fetch: weights, KV pool, and context per stream, computed before launch.
        "memory_plan": None,
        # GGUF runs only: the quant file, the one llama.cpp loads, and what the file's metadata says.
        "gguf": None if gguf is None else {"selector": gguf},
        # A full-precision GGUF run writes the logits a later KL reads.
        "logprobs_reference_file": None,
        # Filled at logprobs: the sha256 and size of each file in LARGE_FILES that the run wrote.
        "large_files": {},
    }


def _deterministic_quality(cfg: Config, engine: str) -> dict[str, Any]:
    from sma.backends import get_backend

    return get_backend(engine).deterministic(cfg)


def run_dir_for(
    cfg: Config, repo: str, quant_flag: str, run_id: str, engine: str = "vllm", *, gguf: str | None = None
) -> Path:
    # Engine is part of the subject. Old runs live at runs/<slug>/<quant>/<id>.
    # Each GGUF file of a repo is its own quant: runs/<slug>/llamacpp/gguf-<selector>/<id>.
    quant_dir = f"gguf-{model_slug(gguf)}" if gguf else quant_flag
    return cfg.runs_path() / model_slug(repo) / engine / quant_dir / run_id


def large_file_hashes(run_dir: Path) -> dict[str, dict[str, Any]]:
    """The sha256 and size of each file in LARGE_FILES that exists in the run folder."""
    found = {}
    for rel in LARGE_FILES:
        path = run_dir / rel
        if path.is_file():
            found[rel] = {"sha256": f"sha256:{sha256_file(path)}", "bytes": path.stat().st_size}
    return found


def large_file_problem(run_dir: Path, manifest: dict[str, Any], rel: str) -> str | None:
    """Why a run's large file does not match its manifest, or None. A run with no recorded hash passes."""
    recorded = (manifest.get("large_files") or {}).get(rel)
    if not recorded:
        return None
    path = run_dir / rel
    if path.stat().st_size != recorded["bytes"] or f"sha256:{sha256_file(path)}" != recorded["sha256"]:
        return f"{rel} does not match the sha256 in its manifest. It is not the file this run wrote."
    return None


def marker_path(run_dir: Path, step: str) -> Path:
    return run_dir / "steps" / f"{step}.done"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise SmaError(f"No manifest.json in {run_dir}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise SmaError(f"{path} is not a JSON object")
    return payload


def save_manifest(run_dir: Path, manifest: dict[str, Any], *, strict: bool = True) -> None:
    cleaned = scrub_obj(manifest)
    if not isinstance(cleaned, dict):
        raise SmaError("manifest scrub failed")
    if strict:
        validate_instance(cleaned, "manifest.schema.json")
    atomic_write_json(run_dir / "manifest.json", cleaned)
    manifest.clear()
    manifest.update(cleaned)


def save_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    cleaned = scrub_obj(summary)
    validate_instance(cleaned, "summary.schema.json")
    atomic_write_json(run_dir / "summary.json", cleaned)


def publish(cfg: Config, run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    save_manifest(run_dir, manifest)
    summary = build_summary(run_dir, manifest, cfg)
    save_summary(run_dir, summary)
    Catalog(cfg.runs_path()).upsert(manifest, run_dir)
    return summary


def overall_status(manifest: dict[str, Any]) -> str:
    steps = manifest.get("steps") or {}
    if any(isinstance(step, dict) and step.get("status") == "failed" for step in steps.values()):
        return "failed"
    timestamps = manifest.get("timestamps") or {}
    if timestamps.get("finished"):
        return "done"
    if timestamps.get("started"):
        return "running"
    return "pending"


def build_summary(run_dir: Path, manifest: dict[str, Any], cfg: Config) -> dict[str, Any]:
    determinism = _optional_json(run_dir / "determinism.json")
    perplexity = _optional_json(run_dir / "logprobs" / "perplexity.json")
    kl = _optional_json(run_dir / "logprobs" / "kl_vs_reference.json")
    speed = _optional_json(run_dir / "speed" / "summary.json")
    windows = _optional_json(run_dir / "speed" / "windows.json")
    params = manifest.get("params") or {}
    engine = manifest.get("engine") or {}
    samples = read_samples(run_dir / "telemetry.csv")
    throttle_samples = sum(1 for sample in samples if sample.throttle)
    temps = [sample.temp_c for sample in samples if sample.temp_c is not None]
    ceiling, ceiling_note = bandwidth_ceiling_tok_s(
        params.get("active"),
        manifest.get("quant_label") or manifest.get("quant_flag") or "",
        cfg.telemetry.spec_bandwidth_gbps,
    )
    energy_joules, energy_method = _energy(samples, windows, max_gap_s=cfg.telemetry.max_sample_gap_s)
    bench_requests = _bench_requests(windows, cfg)
    per_request = None
    if energy_joules is not None and bench_requests:
        per_request = energy_joules / bench_requests
    quality: dict[str, Any] = {}
    quality_root = run_dir / "quality"
    if quality_root.is_dir():
        for suite_dir in sorted(path for path in quality_root.iterdir() if path.is_dir()):
            scores = _optional_json(suite_dir / "scores.json")
            if not isinstance(scores, dict):
                continue
            quality[suite_dir.name] = {
                "score": scores.get("primary_score"),
                "metric": scores.get("primary_metric"),
                # Runs from before NeMo Evaluator have no scorer field and were scored by lm-eval.
                "scorer": scores.get("scorer") or "lm-eval-harness",
                "limit": scores.get("limit"),
                "n": scores.get("n"),
                "ci_lower": scores.get("ci_lower"),
                "ci_upper": scores.get("ci_upper"),
                "context_per_stream": scores.get("context_per_stream"),
                "error": scores.get("error"),
            }
    speed_block = {
        "prefill_tok_s": _dig(speed, "prefill_tok_s"),
        "prefill_tok_s_stat": cfg.speed.derive_stat,
        "decode_tok_s": _dig(speed, "decode_tok_s"),
        "decode_tok_s_stat": cfg.speed.derive_stat,
        "output_tok_s_c4": _dig(speed, "output_tok_s_c4"),
        "output_tok_s_c8": _dig(speed, "output_tok_s_c8"),
        "invalid_cells": int(_dig(speed, "invalid_cells") or 0),
        "unverified_cells": int(
            _dig(speed, "unverified_cells")
            if _dig(speed, "unverified_cells") is not None
            else sum(
                1
                for cell in (_dig(speed, "cells") or [])
                if isinstance(cell, dict) and cell.get("validity") == "unverified"
            )
        ),
        "bandwidth_ceiling_tok_s": ceiling,
        "bandwidth_ceiling_note": ceiling_note,
        "cells": list(_dig(speed, "cells") or []),
    }
    warnings = list(manifest.get("warnings") or [])
    deployment = _deployment(manifest, speed, run_dir)
    return {
        "schema_version": 2,
        "run_id": manifest["run_id"],
        "model_repo": manifest["model_repo"],
        "model_slug": manifest["model_slug"],
        "quant_flag": manifest["quant_flag"],
        "quant_label": manifest.get("quant_label"),
        "quant_detected": manifest.get("quant_detected"),
        "revision_sha": manifest.get("revision_sha"),
        "host_alias": manifest["host_alias"],
        "engine": (manifest.get("engine") or {}).get("name") or "vllm",
        "engine_image": engine.get("image") or cfg.engine.image,
        "engine_image_digest": engine.get("image_digest"),
        "status": overall_status(manifest),
        "determinism_passed": determinism.get("passed") if isinstance(determinism, dict) else None,
        "batch_invariant": engine.get("batch_invariant_quality"),
        "perplexity": perplexity.get("perplexity") if isinstance(perplexity, dict) else None,
        "perplexity_tokens": perplexity.get("token_count") if isinstance(perplexity, dict) else None,
        "kl_mean": kl.get("kl_mean") if isinstance(kl, dict) else None,
        "kl_method": kl.get("method") if isinstance(kl, dict) else None,
        "kl_k": kl.get("k") if isinstance(kl, dict) else None,
        "top1_agreement": kl.get("top1_agreement") if isinstance(kl, dict) else None,
        "delta_ppl": kl.get("delta_ppl") if isinstance(kl, dict) else None,
        "reference_run_id": manifest.get("reference_run_id"),
        "quality": quality,
        "speed": speed_block,
        "energy": {
            "joules": energy_joules,
            "joules_per_bench_request": per_request,
            "method": energy_method,
            "sample_hz": cfg.telemetry.interval_s,
            "scope": "gpu",
            "note": (
                "Each valid speed-cell window is measured on its own and the joules are summed. "
                "nvml-energy-counter is the GPU's cumulative energy counter read before and after the "
                "bench command. nvidia-smi-1hz-trapezoid integrates sampled power and is used only when "
                "the counter was not read; gaps longer than telemetry.max_sample_gap_s are not bridged. "
                "Invalid or unverified windows are excluded. GPU only: on a DGX Spark the CPU shares the "
                "140 W SoC budget and is not counted. Not joules per correct answer, and not a wall meter. "
                "Null means it was not measured."
            ),
        },
        "thermal": {
            "temp_start_c": temps[0] if temps else None,
            "temp_end_c": temps[-1] if temps else None,
            "throttle_samples": throttle_samples,
            "clock_cap_mhz": engine.get("clock_cap_mhz"),
            "sw_power_cap_invalidates": bool(cfg.telemetry.sw_power_cap_invalidates),
            "log": "telemetry.csv",
        },
        "params_total": params.get("total"),
        "params_active": params.get("active"),
        "params_total_b": params.get("total_b"),
        "params_active_b": params.get("active_b"),
        "size_class": params.get("size_class"),
        "deployment": deployment,
        "warnings": [scrub_text(str(item)) for item in warnings],
    }


_KV_LABEL = {"fp8": "an FP8", "q8_0": "a Q8_0"}


def _deployment(manifest: dict[str, Any], speed: Any, run_dir: Path) -> dict[str, Any]:
    """What someone gets running this model this way on a Spark. Computed values come from the memory
    plan before launch; measured values from the engine's log, the speed cells, and memory samples."""
    plan = manifest.get("memory_plan") if isinstance(manifest.get("memory_plan"), dict) else {}
    engine = manifest.get("engine") or {}
    context = plan.get("context")
    reported = engine.get("kv_cache_reported") or {}
    pool_tokens = next(
        (item.get("pool_tokens") for item in (reported.get("speed"), reported.get("quality")) if isinstance(item, dict) and isinstance(item.get("pool_tokens"), int)),
        None,
    )
    measured_streams = measured_32k = None
    if isinstance(pool_tokens, int) and isinstance(context, int) and context > 0:
        measured_streams = pool_tokens // context
        measured_32k = pool_tokens // min(32_768, context)
    rates = {concurrency: None for concurrency in (1, 4, 8)}
    for cell in (_dig(speed, "cells") or []):
        if isinstance(cell, dict) and cell.get("shape") == "sweep" and cell.get("valid") and cell.get("concurrency") in rates:
            rates[cell["concurrency"]] = cell.get("output_tok_s")
    peak = peak_memory_bytes(read_memory(run_dir / "memory.csv"))
    computed = {
        "weights_gb": plan.get("weights_gb"),
        "kv_bytes_per_token": plan.get("kv_bytes_per_token"),
        "kv_pool_gb": plan.get("pool_gb"),
        "kv_pool_tokens": plan.get("pool_tokens"),
        "streams_full_context": plan.get("streams_full_context"),
        "streams_32k": plan.get("streams_32k"),
        "planned_footprint_gb": plan.get("footprint_gb"),
        "memory_budget_gb": plan.get("budget_gb"),
    }
    measured = {
        "kv_pool_tokens": pool_tokens,
        "streams_full_context": measured_streams,
        "streams_32k": measured_32k,
        "peak_memory_gb": None if peak is None else round(peak / 1e9, 2),
        "output_tok_s_c1": rates[1],
        "output_tok_s_c4": rates[4],
        "output_tok_s_c8": rates[8],
    }
    return {
        "sentence": _deployment_sentence(manifest, plan, measured) if plan else None,
        "engine": engine.get("name") or "vllm",
        "weight_format": manifest.get("quant_label") or manifest.get("quant_flag"),
        "kv_cache_type": engine.get("kv_cache_type"),
        "context_per_stream": context,
        "context_reason": plan.get("context_reason"),
        "computed": computed,
        "measured": measured,
    }


def _deployment_sentence(manifest: dict[str, Any], plan: dict[str, Any], measured: dict[str, Any]) -> str:
    engine = manifest.get("engine") or {}
    weights = manifest.get("quant_label") or str(manifest.get("quant_flag") or "").upper()
    kv = _KV_LABEL.get(str(engine.get("kv_cache_type")), "an 8-bit")
    streams = measured.get("streams_full_context")
    streams_text = (
        f"{streams} full-length streams by the engine's own KV report"
        if streams is not None
        else f"{plan.get('streams_full_context')} full-length streams (computed)"
    )
    sentence = (
        f"{weights} weights with {kv} KV cache on {engine.get('name')}: {plan.get('weights_gb')} GB of weights, "
        f"{plan.get('context'):,} tokens per stream, {streams_text} within {plan.get('budget_gb')} GB."
    )
    if measured.get("peak_memory_gb") is not None:
        sentence += f" Peak memory {measured['peak_memory_gb']} GB."
    return sentence


def _optional_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _dig(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return None


_COUNTER = "nvml-energy-counter-per-window"
_TRAPEZOID = "nvidia-smi-1hz-trapezoid-per-window"


def _energy(samples: list[Any], windows: Any, *, max_gap_s: float) -> tuple[float | None, str]:
    """Sum valid windows: the energy counter delta when it was read, else the power integral."""
    if not isinstance(windows, dict):
        return None, _COUNTER
    total = 0.0
    methods: set[str] = set()
    for item in windows.get("windows") or []:
        if not isinstance(item, dict) or not item.get("valid"):
            continue
        counted = item.get("energy_j")
        if isinstance(counted, (int, float)) and not isinstance(counted, bool):
            total += float(counted)
            methods.add(_COUNTER)
            continue
        start = item.get("start_unix")
        end = item.get("end_unix")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        selected = [
            sample
            for sample in samples
            if float(start) <= sample.ts_unix <= float(end)
        ]
        joules = integrate_joules(selected, max_gap_s=max_gap_s)
        if joules is None:
            continue
        total += joules
        methods.add(_TRAPEZOID)
    if not methods:
        return None, _COUNTER
    method = methods.pop() if len(methods) == 1 else f"mixed: {_COUNTER} where read, else {_TRAPEZOID}"
    return total, method


def _bench_requests(windows: Any, cfg: Config) -> int:
    if not isinstance(windows, dict):
        return 0
    count = 0
    for item in windows.get("windows") or []:
        if isinstance(item, dict) and item.get("valid"):
            count += cfg.speed.num_prompts * int(item.get("repeats_in_window") or 1)
    return count
