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
from sma.schemautil import validate_instance
from sma.scrub import scrub_obj, scrub_text
from sma.speedmath import bandwidth_ceiling_tok_s
from sma.telemetry import integrate_joules, read_samples

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
) -> dict[str, Any]:
    commit, dirty = git_identity()
    now = utc_now()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model_repo": repo,
        "model_slug": model_slug(repo),
        "revision_requested": revision,
        "revision_sha": None,
        "quant_flag": quant_flag,
        "quant_label": None,
        "quant_detected": None,
        "quant_detail": None,
        "quant_producer": None,
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
            "engine": "vllm",
            "hardware_alias": cfg.host_alias,
        },
        "engine": {
            "name": "vllm",
            "image": cfg.engine.image,
            "image_digest": cfg.engine.image_digest,
            "tensor_parallel_size": 1,
            "moe_backend_config": cfg.engine.moe_backend,
            "moe_backend_chain": moe_backend_chain(cfg),
            "moe_backend_used": None,
            "moe_backend_fallback": False,
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
            "logprobs_mode": "raw_logprobs",
            "flags_quality": [],
            "flags_speed": [],
            "env_quality": {},
            "env_speed": {},
            "clock_cap_mhz": cfg.engine.clock_cap_mhz,
            "resolved_backend_note": None,
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
    }


def run_dir_for(cfg: Config, repo: str, quant_flag: str, run_id: str) -> Path:
    return cfg.runs_path() / model_slug(repo) / quant_flag / run_id


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
    energy_joules = _energy(samples, windows)
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
                "scorer": scores.get("scorer") or "lm-eval-harness",
                "limit": scores.get("limit"),
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
        "bandwidth_ceiling_tok_s": ceiling,
        "bandwidth_ceiling_note": ceiling_note,
        "cells": list(_dig(speed, "cells") or []),
    }
    warnings = list(manifest.get("warnings") or [])
    return {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "model_repo": manifest["model_repo"],
        "model_slug": manifest["model_slug"],
        "quant_flag": manifest["quant_flag"],
        "quant_label": manifest.get("quant_label"),
        "quant_detected": manifest.get("quant_detected"),
        "revision_sha": manifest.get("revision_sha"),
        "host_alias": manifest["host_alias"],
        "engine": "vllm",
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
            "method": "nvidia-smi-1hz-trapezoid",
            "sample_hz": cfg.telemetry.interval_s,
            "scope": "gpu",
            "note": (
                "Integral of GPU power over valid speed-cell windows. "
                "Not joules per correct answer, and not a wall-meter reading. "
                "Null means it was not measured."
            ),
        },
        "thermal": {
            "temp_start_c": temps[0] if temps else None,
            "temp_end_c": temps[-1] if temps else None,
            "throttle_samples": throttle_samples,
            "clock_cap_mhz": engine.get("clock_cap_mhz"),
            "log": "telemetry.csv",
        },
        "params_total": params.get("total"),
        "params_active": params.get("active"),
        "params_total_b": params.get("total_b"),
        "params_active_b": params.get("active_b"),
        "size_class": params.get("size_class"),
        "warnings": [scrub_text(str(item)) for item in warnings],
    }


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


def _energy(samples: list[Any], windows: Any) -> float | None:
    if not isinstance(windows, dict):
        return None
    spans = []
    for item in windows.get("windows") or []:
        if not isinstance(item, dict) or not item.get("valid"):
            continue
        start = item.get("start_unix")
        end = item.get("end_unix")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            spans.append((float(start), float(end)))
    if not spans:
        return None
    selected = [
        sample
        for sample in samples
        if any(start <= sample.ts_unix <= end for start, end in spans)
    ]
    return integrate_joules(selected)


def _bench_requests(windows: Any, cfg: Config) -> int:
    if not isinstance(windows, dict):
        return 0
    count = 0
    for item in windows.get("windows") or []:
        if isinstance(item, dict) and item.get("valid"):
            count += cfg.speed.num_prompts * int(item.get("repeats_in_window") or 1)
    return count
