"""One checkpoint, one Spark, resumable steps, teardown on every exit."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sma.atomic import atomic_write_json, atomic_write_text
from sma.cleanup import drop_page_cache, idle_until_cool
from sma.commands import (
    EVAL_FACTORY_CONFIG,
    NEL_CONFIG,
    bench_argv,
    bench_cells,
    container_name,
    docker_argv,
    eval_factory_argv,
    eval_factory_config,
    moe_backend_chain,
    nel_argv,
    nel_config,
    packaged_bench,
    serve_argv,
    serve_env,
)
from sma.compare import compare_run_dirs
from sma.config import Config, QualitySuite
from sma.corpus import build_windows
from sma.errors import CompareRefused, SmaError
from sma.fetch import FetchResult, download_snapshot, inspect_snapshot
from sma.identity import assert_alias_safe
from sma.kl import perplexity
from sma.locks import GpuLock
from sma.nvml import energy_counter_mj
from sma.logsetup import add_run_log, log_event
from sma.probe import run_determinism
from sma.proc import require_ok, run_command
from sma.prom import diff_samples, parse_prometheus
from sma.qualityout import normalize_eval_factory, normalize_suite, scrub_output
from sma.records import (
    STEP_NAMES,
    load_manifest,
    marker_path,
    new_manifest,
    new_run_id,
    publish,
    run_dir_for,
    utc_now,
)
from sma.runtime import DockerRuntime, StartSpec
from sma.scrub import assert_text_clean, scrub_obj, scrub_text
from sma.serverwatch import (
    STEP_SERVER,
    kv_capacity_warning,
    needed_kv_tokens,
    report_wait,
    server_log_path,
    start_following,
    stopped_server_note,
    tail_lines,
)
from sma.speedmath import (
    bench_request_problem,
    decode_tok_s,
    load_bench,
    output_tok_s,
    percentile_ms,
    prefill_tok_s,
    summarize_cells,
)
from sma.telemetry import TelemetryRecorder, throttled_between
from sma.httpapi import scrape_metrics, wait_until_healthy
from sma.engines import compatibility_error, configured_digest, engine_name, image_for, image_ref
from sma.preflight import check_architecture

PERCENTILES = (50, 90, 99)


@dataclass
class Hooks:
    kill_step: str | None = None
    before_speed_cell: Callable[["Ctx"], None] | None = None


@dataclass
class Services:
    fetcher: Any
    runtime: Any
    evaluator: Any
    bencher: Any
    telemetry_factory: Callable[[Path, Config], TelemetryRecorder]
    encode: Callable[[str], list[int]] | None = None
    hooks: Hooks = field(default_factory=Hooks)
    registry: Any = None
    # The GPU's cumulative energy counter in millijoules, or None. Unset means power integration.
    energy_counter: Callable[[], int | None] | None = None


@dataclass
class RunSpec:
    repo: str
    quant_flag: str
    revision: str | None = None
    reference_run_id: str | None = None
    pair_job_id: str | None = None
    resume_run_id: str | None = None
    lock_held: bool = False
    engine: str | None = None
    # The engine's tool-call parser for this model. None falls back to engine.tool_call_parser.
    tool_parser: str | None = None


@dataclass
class Ctx:
    cfg: Config
    services: Services
    run_dir: Path
    manifest: dict[str, Any]
    telemetry: TelemetryRecorder | None = None
    snapshot: Path | None = None
    batch_invariant: bool = True
    closed: bool = False
    # Live server-log copiers by mode. Stopped and joined when that server stops.
    followers: dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.cfg.engine.bind_host}:{self.cfg.engine.port}"

    @property
    def served_name(self) -> str:
        return str(self.manifest["model_repo"])

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])


class HubFetcher:
    def __init__(self, cfg: Config, registry: Any = None, engine: str = "vllm") -> None:
        self.cfg = cfg
        self.registry = registry
        self.engine = engine

    def fetch(self, repo: str, revision: str | None, quant_flag: str) -> FetchResult:
        from huggingface_hub import HfApi, hf_hub_download

        from sma.fetch import _read_json

        token = None
        from sma.config import hf_token

        token = hf_token(self.cfg)
        api = HfApi(token=token)
        info = api.model_info(repo_id=repo, revision=revision)
        sha = info.sha
        files = [item.rfilename for item in (info.siblings or [])]
        config_path = hf_hub_download(
            repo_id=repo, filename="config.json", revision=sha, cache_dir=str(self.cfg.cache_path()), token=token
        )
        config = _read_json(Path(config_path)) or {}
        hf_quant = None
        if "hf_quant_config.json" in files:
            quant_path = hf_hub_download(
                repo_id=repo,
                filename="hf_quant_config.json",
                revision=sha,
                cache_dir=str(self.cfg.cache_path()),
                token=token,
            )
            hf_quant = _read_json(Path(quant_path))
        from sma.quant import assert_flag_matches, detect_quant

        detected = detect_quant(config, hf_quant)
        assert_flag_matches(quant_flag, detected)
        blocked = compatibility_error(self.engine, detected)
        if blocked:
            raise SmaError(blocked)
        verdict = check_architecture(
            engine=self.engine,
            detected=detected,
            config=config,
            repo_files=files,
            trust_remote_code=self.cfg.engine.trust_remote_code,
            registry=self.registry,
            image=image_ref(image_for(self.cfg, self.engine), configured_digest(self.cfg, self.engine)),
        )
        if not verdict.ok:
            raise SmaError(verdict.message)
        path, sha = download_snapshot(repo, sha, self.cfg)
        return inspect_snapshot(
            path,
            repo=repo,
            revision_sha=sha,
            quant_flag=quant_flag,
            revision_requested=revision,
        )


class CommandEvaluator:
    def run(self, argv: list[str], output_dir: Path, timeout_s: float, log_path: Path) -> None:
        import shutil

        output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which(argv[0]) is None:
            raise SmaError(
                f"{argv[0]} is not on PATH. Install the quality extra "
                "(`uv sync --extra quality`) or set quality.executable."
            )
        # nel reads nel.yaml and the packaged benchmark file from the suite folder. Python would cache
        # that file's bytecode next to it, and bytecode stores the absolute path, hostname included.
        result = run_command(
            argv,
            timeout_s=timeout_s,
            what="nemo-evaluator",
            cwd=output_dir,
            extra_env={"PYTHONDONTWRITEBYTECODE": "1"},
        )
        _write_log(log_path, result.stdout + "\n" + result.stderr)
        require_ok(result, what=f"nemo-evaluator {' '.join(argv)}")


class CommandBencher:
    def run(self, argv: list[str], result_path: Path, timeout_s: float, log_path: Path) -> None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result = run_command(argv, timeout_s=timeout_s, what="bench")
        _write_log(log_path, result.stdout + "\n" + result.stderr)
        require_ok(result, what="bench")
        if not result_path.is_file():
            raise SmaError(f"bench finished without writing {result_path.name}")


def production_services(cfg: Config) -> Services:
    def _telemetry(run_dir: Path, config: Config) -> TelemetryRecorder:
        recorder = TelemetryRecorder(run_dir / "telemetry.csv", config)
        recorder.start(sampler=True)
        return recorder

    from sma.preflight import default_registry

    registry = default_registry(cfg)
    return Services(
        fetcher=HubFetcher(cfg, registry=registry, engine=cfg.engine.name),
        runtime=DockerRuntime(cfg),
        evaluator=CommandEvaluator(),
        bencher=CommandBencher(),
        telemetry_factory=_telemetry,
        registry=registry,
        energy_counter=lambda: energy_counter_mj(cfg.doctor.gpu_index),
    )


def execute(
    spec: RunSpec,
    cfg: Config,
    services: Services,
    *,
    on_created: Callable[[str], None] | None = None,
) -> str:
    assert_alias_safe(cfg.host_alias)
    chosen = engine_name(spec.engine, cfg)
    spec.engine = chosen
    ctx: Ctx | None = None
    lock = None if spec.lock_held else GpuLock(cfg.lock_file())
    if lock is not None:
        lock.acquire()
    try:
        if spec.resume_run_id:
            ctx = _load(spec, cfg, services)
        else:
            ctx = _create(spec, cfg, services)
        if on_created is not None:
            on_created(ctx.run_id)
        handler = _SignalGuard(lambda: _teardown_containers(ctx))
        with handler:
            _guard_containers(ctx)
            _maybe_clock_cap(ctx)
            if ctx.manifest["timestamps"]["started"] is None:
                ctx.manifest["timestamps"]["started"] = utc_now()
            publish(cfg, ctx.run_dir, ctx.manifest)
            ctx.telemetry = services.telemetry_factory(ctx.run_dir, cfg)
            ctx.telemetry.set_phase("startup")
            try:
                _run_step(ctx, "fetch", _step_fetch)
                _run_step(ctx, "serve_quality", _step_serve_quality)
                _run_step(ctx, "determinism", _step_determinism)
                _run_step(ctx, "logprobs", _step_logprobs)
                _run_step(ctx, "quality", _step_quality)
                _run_step(ctx, "serve_speed", _step_serve_speed)
                _run_step(ctx, "speed", _step_speed)
                ctx.manifest["timestamps"]["finished"] = utc_now()
                publish(cfg, ctx.run_dir, ctx.manifest)
            except BaseException:
                ctx.manifest["timestamps"]["finished"] = utc_now()
                try:
                    publish(cfg, ctx.run_dir, ctx.manifest)
                except Exception:
                    log_event("publish_failed", run_id=ctx.run_id)
                raise
            return ctx.run_id
    finally:
        if ctx is not None:
            _finish(ctx)
        if lock is not None:
            lock.release()


def _create(spec: RunSpec, cfg: Config, services: Services) -> Ctx:
    run_id = new_run_id()
    run_dir = run_dir_for(cfg, spec.repo, spec.quant_flag, run_id, spec.engine or "vllm")
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("fetch", "logprobs", "quality", "speed", "logs", "steps"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    manifest = new_manifest(
        cfg=cfg,
        repo=spec.repo,
        quant_flag=spec.quant_flag,
        revision=spec.revision,
        run_id=run_id,
        reference_run_id=spec.reference_run_id,
        pair_job_id=spec.pair_job_id,
        engine=spec.engine or "vllm",
        tool_call_parser=spec.tool_parser or cfg.engine.tool_call_parser,
    )
    add_run_log(run_dir / "logs" / "sma.log")
    publish(cfg, run_dir, manifest)
    log_event("run_created", run_id=run_id, repo=spec.repo, quant=spec.quant_flag)
    return Ctx(cfg=cfg, services=services, run_dir=run_dir, manifest=manifest)


def _load(spec: RunSpec, cfg: Config, services: Services) -> Ctx:
    from sma.catalog import Catalog

    found = Catalog(cfg.runs_path()).find(spec.resume_run_id or "")
    if found is None:
        raise SmaError(f"No run with id {spec.resume_run_id}")
    manifest = load_manifest(found)
    if manifest.get("host_alias") and manifest.get("host_alias") != cfg.host_alias:
        raise SmaError(
            f"Run {manifest.get('run_id')} is pinned to host alias {manifest.get('host_alias')}. "
            f"This process is {cfg.host_alias}. A different Spark must not resume it."
        )
    recorded_engine = (manifest.get("engine") or {}).get("name")
    if spec.engine and recorded_engine and recorded_engine != spec.engine:
        raise SmaError(
            f"Run {spec.resume_run_id} was started on {recorded_engine}, not {spec.engine}."
        )
    recorded_parser = (manifest.get("engine") or {}).get("tool_call_parser")
    if spec.tool_parser and spec.tool_parser != recorded_parser:
        raise SmaError(
            f"Run {spec.resume_run_id} was started with tool parser {recorded_parser or 'none'}, "
            f"not {spec.tool_parser}. A resumed run keeps its server flags."
        )
    if manifest.get("model_repo") != spec.repo or manifest.get("quant_flag") != spec.quant_flag:
        raise SmaError(
            f"Run {spec.resume_run_id} is {manifest.get('model_repo')} "
            f"--{manifest.get('quant_flag')}, not {spec.repo} --{spec.quant_flag}."
        )
    add_run_log(found / "logs" / "sma.log")
    log_event("run_resume", run_id=manifest["run_id"])
    ctx = Ctx(cfg=cfg, services=services, run_dir=found, manifest=manifest)
    fetch = _optional(found / "fetch" / "result.json")
    if isinstance(fetch, dict) and fetch.get("snapshot_path"):
        ctx.snapshot = Path(str(fetch["snapshot_path"]))
    ctx.batch_invariant = bool(manifest["engine"].get("batch_invariant_quality")) and bool(
        manifest["engine"].get("batch_invariant_supported", True)
    )
    return ctx


def _finish(ctx: Ctx) -> None:
    if ctx.closed:
        return
    ctx.closed = True
    if ctx.telemetry is not None:
        ctx.telemetry.set_phase("teardown")
        try:
            ctx.telemetry.stop()
            _mark_side(ctx, "telemetry", "done", None)
        except Exception as exc:
            _mark_side(ctx, "telemetry", "failed", str(exc))
    else:
        _mark_side(ctx, "telemetry", "skipped", "telemetry never started")
    try:
        _teardown_containers(ctx)
        _mark_side(ctx, "teardown", "done", None)
    except Exception as exc:
        _mark_side(ctx, "teardown", "failed", str(exc))
    try:
        _cleanup(ctx)
        _mark_side(ctx, "cleanup", "done", None)
    except Exception as exc:
        _mark_side(ctx, "cleanup", "failed", str(exc))
    try:
        publish(ctx.cfg, ctx.run_dir, ctx.manifest)
    except Exception as exc:
        log_event("final_publish_failed", error=str(exc))


def _mark_side(ctx: Ctx, name: str, status: str, error: str | None) -> None:
    if marker_path(ctx.run_dir, name).is_file() and status == "done":
        ctx.manifest["steps"][name]["status"] = "done"
        return
    step = ctx.manifest["steps"][name]
    if step["status"] == "done" and status != "failed":
        return
    now = utc_now()
    step["status"] = status
    step["started"] = step.get("started") or now
    step["finished"] = now
    step["attempts"] = int(step.get("attempts") or 0) + 1
    step["error"] = None if error is None else scrub_text(error)[:2000]
    if status == "done":
        atomic_write_json(marker_path(ctx.run_dir, name), {"step": name, "finished": now})


def _run_step(ctx: Ctx, name: str, fn: Callable[[Ctx], None]) -> None:
    if name not in STEP_NAMES:
        raise SmaError(f"unknown step {name}")
    if marker_path(ctx.run_dir, name).is_file():
        ctx.manifest["steps"][name]["status"] = "done"
        ctx.manifest["steps"][name]["error"] = None
        publish(ctx.cfg, ctx.run_dir, ctx.manifest)
        log_event("step_skip", step=name, run_id=ctx.run_id)
        return
    if ctx.telemetry is not None:
        ctx.telemetry.set_phase(name)
    step = ctx.manifest["steps"][name]
    step["attempts"] = int(step.get("attempts") or 0) + 1
    step["status"] = "running"
    step["started"] = step.get("started") or utc_now()
    step["error"] = None
    publish(ctx.cfg, ctx.run_dir, ctx.manifest)
    log_event("step_start", step=name, run_id=ctx.run_id, attempt=step["attempts"])
    try:
        fn(ctx)
        if ctx.services.hooks.kill_step == name:
            ctx.services.hooks.kill_step = None
            raise RuntimeError("killed mid-step")
        atomic_write_json(
            marker_path(ctx.run_dir, name),
            {"step": name, "finished": utc_now()},
        )
        done = ctx.manifest["steps"][name]
        done["status"] = "done"
        done["finished"] = utc_now()
        done["error"] = None
        publish(ctx.cfg, ctx.run_dir, ctx.manifest)
        log_event("step_done", step=name, run_id=ctx.run_id)
    except Exception as exc:
        note = _stopped_server_note(ctx, name)
        message = str(exc) if note is None else f"{exc}\n{note}"
        failed = ctx.manifest["steps"][name]
        failed["status"] = "failed"
        failed["finished"] = utc_now()
        failed["error"] = scrub_text(message)[:2000]
        try:
            publish(ctx.cfg, ctx.run_dir, ctx.manifest)
        except Exception:
            log_event("step_publish_failed", step=name)
        log_event("step_failed", step=name, run_id=ctx.run_id, error=message)
        raise


def _stopped_server_note(ctx: Ctx, step: str) -> str | None:
    mode = STEP_SERVER.get(step)
    if mode is None:
        return None
    try:
        return stopped_server_note(
            ctx.services.runtime,
            name=container_name(ctx.cfg, ctx.run_id, mode),
            mode=mode,
            log_path=server_log_path(ctx.run_dir, mode),
            poll_s=ctx.cfg.timeouts.health_poll_s,
        )
    except Exception as exc:
        log_event("server_check_failed", step=step, error=str(exc))
        return None


def _warn(ctx: Ctx, message: str) -> None:
    text = scrub_text(message)
    if text not in ctx.manifest["warnings"]:
        ctx.manifest["warnings"].append(text)
    log_event("warning", message=text)


def _step_fetch(ctx: Ctx) -> None:
    fetcher = ctx.services.fetcher
    if hasattr(fetcher, "engine"):
        fetcher.engine = ctx.manifest["engine"]["name"]
    if hasattr(fetcher, "registry") and ctx.services.registry is not None:
        fetcher.registry = ctx.services.registry
    result = ctx.services.fetcher.fetch(
        ctx.manifest["model_repo"],
        ctx.manifest["revision_requested"],
        ctx.manifest["quant_flag"],
    )
    if not isinstance(result, FetchResult):
        raise SmaError("fetcher returned an unexpected result")
    snapshot = assert_text_clean(result.snapshot_path, "snapshot path")
    ctx.snapshot = Path(snapshot)
    detected = result.detected
    ctx.manifest["revision_sha"] = result.revision_sha
    ctx.manifest["quant_label"] = result.quant_label
    ctx.manifest["quant_detected"] = detected.format
    ctx.manifest["quant_detail"] = detected.detail
    ctx.manifest["quant_producer"] = detected.producer
    ctx.manifest["architecture"] = detected.model_type
    ctx.manifest["architectures"] = list(detected.architectures)
    ctx.manifest["is_moe"] = detected.is_moe
    ctx.manifest["file_hashes"] = dict(result.file_hashes)
    ctx.manifest["tokenizer_fingerprint"] = result.tokenizer_fingerprint
    ctx.manifest["quant_composition"] = result.detected.composition
    blocked = compatibility_error(ctx.manifest["engine"]["name"], result.detected)
    if blocked:
        raise SmaError(blocked)
    repo_files = [path.relative_to(ctx.snapshot).as_posix() for path in ctx.snapshot.rglob("*") if path.is_file()]
    verdict = check_architecture(
        engine=ctx.manifest["engine"]["name"],
        detected=result.detected,
        config=result.config,
        repo_files=repo_files,
        trust_remote_code=ctx.cfg.engine.trust_remote_code,
        registry=getattr(ctx.services, "registry", None),
        image=_preflight_image(ctx),
    )
    ctx.manifest["architecture_check"] = {
        "ok": verdict.ok,
        "source": verdict.source,
        "message": verdict.message,
    }
    _remember_registry_digest(ctx)
    if verdict.ok and verdict.source == "bundled":
        _warn(ctx, verdict.message)
    if not verdict.ok:
        raise SmaError(verdict.message)
    ctx.manifest["params"] = {
        "total": result.params_total,
        "active": result.params_active,
        "total_b": result.params_total_b,
        "active_b": result.params_active_b,
        "size_class": result.size_class,
        "method": result.params_method,
        "source": "safetensors shapes" if result.params_total is not None else None,
    }
    ctx.manifest["subject"]["revision_sha"] = result.revision_sha
    ctx.manifest["subject"]["quant"] = result.quant_label or ctx.manifest["quant_flag"]
    ctx.manifest["engine"]["batch_invariant_supported"] = detected.batch_invariant_supported
    ctx.manifest["engine"]["batch_invariant_warning"] = detected.batch_invariant_reason
    ctx.batch_invariant = bool(ctx.cfg.engine.batch_invariant and detected.batch_invariant_supported)
    ctx.manifest["engine"]["batch_invariant_quality"] = ctx.batch_invariant
    for warning in result.warnings:
        _warn(ctx, warning)
    if detected.is_moe and ctx.manifest["quant_flag"] == "bf16":
        _warn(
            ctx,
            "BF16 MoE does not get --moe-backend flashinfer_b12x; that backend is an FP4 kernel. "
            "The flag is set for NVFP4 and MXFP4 MoE only.",
        )
    atomic_write_json(ctx.run_dir / "fetch" / "result.json", result.to_json())


def _step_serve_quality(ctx: Ctx) -> None:
    _remember_digest(ctx)
    _start_server(ctx, "quality")


def _step_determinism(ctx: Ctx) -> None:
    _ensure_server(ctx, "quality")
    result = run_determinism(
        base_url=ctx.base_url,
        served_name=ctx.served_name,
        prompts=list(ctx.cfg.determinism.prompts),
        batch_sizes=list(ctx.cfg.determinism.batch_sizes),
        max_tokens=ctx.cfg.determinism.max_tokens,
        temperature=ctx.cfg.determinism.temperature,
        seed=ctx.cfg.seed,
        timeout_s=ctx.cfg.timeouts.logprob_request_s,
    )
    atomic_write_json(ctx.run_dir / "determinism.json", result)
    if not result["passed"]:
        _warn(
            ctx,
            "Determinism probe failed. The run continues. Use the recorded concurrency "
            "for both quants and treat quality deltas against a wider noise band.",
        )


def _step_logprobs(ctx: Ctx) -> None:
    _ensure_server(ctx, "quality")
    encode = ctx.services.encode or _load_encoder(ctx)
    windows, _text, corpus_meta = build_windows(
        encode,
        version=ctx.cfg.logprobs.corpus_version,
        window_tokens=ctx.cfg.logprobs.window_tokens,
        num_windows=ctx.cfg.logprobs.num_windows,
    )
    stored_windows = []
    true_logprobs: list[float] = []
    missing = 0
    for index, token_ids in enumerate(windows):
        positions = _window_positions(ctx, token_ids)
        for position in positions:
            if position["true_logprob"] is None:
                missing += 1
            else:
                true_logprobs.append(float(position["true_logprob"]))
        stored_windows.append({"index": index, "token_ids": token_ids, "positions": positions})
    ppl = perplexity(true_logprobs)
    dump = {
        "schema_version": 1,
        "corpus_version": ctx.cfg.logprobs.corpus_version,
        "corpus_sha256": corpus_meta.get("sha256"),
        "corpus_dataset": corpus_meta.get("dataset"),
        "corpus_revision": corpus_meta.get("revision"),
        "corpus_license": corpus_meta.get("license"),
        "window_tokens": ctx.cfg.logprobs.window_tokens,
        "top_k": ctx.cfg.logprobs.top_k,
        "tokenizer_fingerprint": ctx.manifest.get("tokenizer_fingerprint"),
        "encode": "tokenizer.json with post-processor disabled; chat template not applied",
        "windows": stored_windows,
        "perplexity": ppl,
        "token_count": len(true_logprobs),
        "missing_true_logprob": missing,
    }
    atomic_write_json(ctx.run_dir / "logprobs" / "dump.json", dump)
    atomic_write_json(
        ctx.run_dir / "logprobs" / "corpus.json",
        {
            "corpus_version": ctx.cfg.logprobs.corpus_version,
            "corpus_sha256": corpus_meta.get("sha256"),
            "corpus_dataset": corpus_meta.get("dataset"),
            "corpus_revision": corpus_meta.get("revision"),
            "corpus_license": corpus_meta.get("license"),
            "window_tokens": ctx.cfg.logprobs.window_tokens,
            "num_windows": len(stored_windows),
            "tokenizer_fingerprint": ctx.manifest.get("tokenizer_fingerprint"),
            "windows": [{"index": item["index"], "token_ids": item["token_ids"]} for item in stored_windows],
        },
    )
    atomic_write_json(
        ctx.run_dir / "logprobs" / "perplexity.json",
        {"perplexity": ppl, "token_count": len(true_logprobs), "missing_true_logprob": missing},
    )
    reference = ctx.manifest.get("reference_run_id")
    if reference:
        _write_reference_kl(ctx, reference)


def _window_positions(ctx: Ctx, token_ids: list[int]) -> list[dict[str, Any]]:
    from sma.backends import get_backend

    backend = get_backend(ctx.manifest["engine"]["name"])
    return backend.logprob_positions(
        base_url=ctx.base_url,
        served_name=ctx.served_name,
        token_ids=token_ids,
        top_k=ctx.cfg.logprobs.top_k,
        seed=ctx.cfg.seed,
        timeout_s=ctx.cfg.timeouts.logprob_request_s,
    )


def _write_reference_kl(ctx: Ctx, reference_run_id: str) -> None:
    from sma.catalog import Catalog

    ref_dir = Catalog(ctx.cfg.runs_path()).find(reference_run_id)
    if ref_dir is None:
        _warn(ctx, f"Reference run {reference_run_id} was not found. KL was not computed.")
        return
    try:
        result = compare_run_dirs(ref_dir, ctx.run_dir)
    except CompareRefused as exc:
        _warn(ctx, str(exc))
        return
    atomic_write_json(ctx.run_dir / "logprobs" / "kl_vs_reference.json", result)


def _step_quality(ctx: Ctx) -> None:
    if not ctx.cfg.quality.enabled:
        _warn(ctx, "Quality suites are disabled in config. No scores were produced.")
        return
    _ensure_server(ctx, "quality")
    for suite in ctx.cfg.quality.suites:
        suite_dir = ctx.run_dir / "quality" / suite.name
        done = suite_dir / ".done"
        if done.is_file():
            log_event("suite_skip", suite=suite.name, run_id=ctx.run_id)
            continue
        suite_dir.mkdir(parents=True, exist_ok=True)
        # A retry writes to a fresh folder, so a failed attempt's partial output is never mixed in.
        attempt = 1 + sum(1 for path in suite_dir.glob("attempt-*") if path.is_dir())
        attempt_dir = suite_dir / f"attempt-{attempt}"
        if suite.tools and not ctx.manifest["engine"].get("tool_call_parser"):
            message = (
                f"Quality suite {suite.name} calls tools, and this run's server has no tool-call parser. "
                "Pass --tool-parser (for example hermes for Qwen3 on vLLM). The score is null."
            )
            _warn(ctx, message)
            atomic_write_json(
                suite_dir / "scores.json",
                {"suite": suite.name, "scorer": "not run", "primary_metric": None, "primary_score": None, "error": message},
            )
        elif suite.image is not None:
            _run_eval_factory_suite(ctx, suite, suite_dir, attempt_dir)
        else:
            _run_nel_suite(ctx, suite, suite_dir, attempt_dir)
        atomic_write_json(done, {"suite": suite.name, "finished": utc_now()})


def _run_nel_suite(ctx: Ctx, suite: QualitySuite, suite_dir: Path, attempt_dir: Path) -> None:
    packaged = packaged_bench(suite.bench)
    if packaged is not None:
        # The grader that scored this run is kept with it.
        atomic_write_text(suite_dir / packaged.name, packaged.read_text(encoding="utf-8"))
    config = nel_config(ctx.cfg, served_name=ctx.served_name, base_url=ctx.base_url, suite=suite)
    atomic_write_text(suite_dir / NEL_CONFIG, yaml.safe_dump(config, sort_keys=False))
    argv = nel_argv(ctx.cfg, output_name=attempt_dir.name)
    atomic_write_json(
        suite_dir / "command.json",
        {"argv": argv, "cwd": f"quality/{suite.name}", "concurrency": ctx.cfg.quality.concurrency},
    )
    try:
        ctx.services.evaluator.run(
            argv, suite_dir, float(ctx.cfg.timeouts.eval_s), ctx.run_dir / "logs" / f"quality-{suite.name}.log"
        )
    finally:
        scrub_output(attempt_dir)
    normalize_suite(suite_dir, attempt_dir, suite.name, bench=suite.bench or suite.name, limit=suite.limit)


def _run_eval_factory_suite(ctx: Ctx, suite: QualitySuite, suite_dir: Path, attempt_dir: Path) -> None:
    assert suite.image is not None and suite.task is not None
    digest = ctx.services.runtime.ensure_image(suite.image)
    image = image_ref(suite.image, digest)
    config_path = suite_dir / EVAL_FACTORY_CONFIG
    config = eval_factory_config(ctx.cfg, served_name=ctx.served_name, base_url=ctx.base_url, suite=suite)
    atomic_write_text(config_path, yaml.safe_dump(config, sort_keys=False))
    attempt_dir.mkdir(parents=True, exist_ok=True)
    argv = eval_factory_argv(
        name=container_name(ctx.cfg, ctx.run_id, f"quality-{suite.name}"),
        image=image,
        attempt_dir=attempt_dir.resolve(),
        config_path=config_path.resolve(),
        uid=os.getuid(),
        gid=os.getgid(),
    )
    # The mounts are absolute host paths, which can hold the hostname.
    atomic_write_json(
        suite_dir / "command.json",
        {"argv": scrub_obj(argv), "concurrency": ctx.cfg.quality.concurrency, "image": image},
    )
    try:
        ctx.services.evaluator.run(
            argv, suite_dir, float(ctx.cfg.timeouts.eval_s), ctx.run_dir / "logs" / f"quality-{suite.name}.log"
        )
    finally:
        scrub_output(attempt_dir)
    normalize_eval_factory(suite_dir, attempt_dir, suite.name, image=image, task=suite.task)


def _step_serve_speed(ctx: Ctx) -> None:
    _stop_mode(ctx, "quality")
    _remember_digest(ctx)
    _start_server(ctx, "speed")


def _step_speed(ctx: Ctx) -> None:
    _ensure_server(ctx, "speed")
    _install_sglang_dataset(ctx)
    _warmup(ctx)
    cells_out: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for cell in bench_cells(ctx.cfg):
        record, window = _run_cell(ctx, cell)
        cells_out.append(record)
        if window is not None:
            windows.append(window)
    summary = summarize_cells(cells_out, stat=ctx.cfg.speed.derive_stat)
    summary["prefill_formula"] = "input_len / TTFT_seconds at concurrency 1, using the configured percentile"
    summary["decode_formula"] = "1 / TPOT_seconds at concurrency 1, using the configured percentile"
    atomic_write_json(ctx.run_dir / "speed" / "summary.json", summary)
    atomic_write_json(ctx.run_dir / "speed" / "windows.json", {"windows": windows})
    cells = summary.get("cells") or []
    throttled_n = sum(1 for cell in cells if cell.get("invalidation_reason") == "throttled")
    bench_n = sum(1 for cell in cells if cell.get("invalidation_reason") == "bench_failed")
    gap_n = sum(1 for cell in cells if cell.get("invalidation_reason") == "telemetry_gap")
    if throttled_n:
        _warn(
            ctx,
            f"{throttled_n} speed cell(s) were invalid because of a thermal or power throttle. "
            "Those retries waited through a cool-down. Throttled cells stay on disk and are excluded from the headline tok/s.",
        )
    if bench_n:
        _warn(
            ctx,
            f"{bench_n} speed cell(s) were invalid because the bench failed "
            "(completed count, request errors, or a missing bench result). "
            "Those retries used a backoff and did not cool down. "
            "Bench-failed cells stay on disk and are excluded from the headline tok/s.",
        )
    if gap_n:
        _warn(
            ctx,
            f"{gap_n} speed cell(s) are unverified because of a telemetry gap, missing samples, or N/A flags. "
            "They were not retried and are excluded from the headline tok/s.",
        )


def _read_energy(ctx: Ctx) -> int | None:
    if ctx.services.energy_counter is None:
        return None
    try:
        return ctx.services.energy_counter()
    except Exception as exc:
        log_event("energy_counter_failed", error=str(exc))
        return None


def _run_cell(ctx: Ctx, cell: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    attempts = 1 + max(0, ctx.cfg.speed.throttle_retries)
    last: dict[str, Any] | None = None
    last_window: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        cell_dir = ctx.run_dir / "speed" / "cells" / cell.name / f"attempt-{attempt}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        if ctx.telemetry is not None:
            ctx.telemetry.set_phase(f"speed:{cell.name}:a{attempt}")
        started = time.time()
        if ctx.services.hooks.before_speed_cell is not None:
            ctx.services.hooks.before_speed_cell(ctx)
        before, before_err = _scrape(ctx)
        container = container_name(ctx.cfg, ctx.run_id, "speed")
        result_dir = f"/sma-run/speed/cells/{cell.name}/attempt-{attempt}"
        argv = bench_argv(
            ctx.cfg,
            container=container,
            served_name=ctx.served_name,
            cell=cell,
            engine=ctx.manifest["engine"]["name"],
            result_dir=result_dir,
        )
        atomic_write_json(cell_dir / "command.json", {"argv": argv})
        energy_before = _read_energy(ctx)
        ctx.services.bencher.run(
            argv,
            cell_dir / "bench.json",
            float(ctx.cfg.timeouts.bench_s),
            ctx.run_dir / "logs" / f"speed-{cell.name}-a{attempt}.log",
        )
        energy_after = _read_energy(ctx)
        after, after_err = _scrape(ctx)
        ended = time.time()
        from sma.backends import get_backend

        backend = get_backend(ctx.manifest["engine"]["name"])
        scrape_error = "; ".join(item for item in (before_err, after_err) if item)
        if scrape_error:
            diff = {
                "watched": {name: None for name in backend.watched_metrics()},
                "counters": {},
                "missing_reason": scrape_error,
            }
            metrics_note = (
                "The /metrics scrape failed. Watched counters are null. "
                f"{scrape_error}"
            )
        else:
            diff = diff_samples(
                parse_prometheus(before),
                parse_prometheus(after),
                watched=backend.watched_metrics(),
            )
            metrics_note = backend.metrics_note()
            if diff.get("missing_reason"):
                metrics_note = f"{metrics_note} {diff['missing_reason']}"
        atomic_write_json(
            cell_dir / "metrics.json",
            {
                "before": before or None,
                "after": after or None,
                "diff": diff,
                "metrics_note": metrics_note,
                "scrape_error": scrape_error or None,
            },
        )
        phase = f"speed:{cell.name}:a{attempt}"
        window_samples = []
        if ctx.telemetry is not None:
            from sma.telemetry import read_samples

            window_samples = [
                sample
                for sample in read_samples(ctx.telemetry.path)
                if sample.phase == phase and (started - 0.05) <= sample.ts_unix <= (ended + 0.05)
            ]
        record = _cell_record(
            ctx, cell, attempt, cell_dir / "bench.json", window_samples, started, ended
        )
        last = record
        last_window = {
            "name": cell.name,
            "start_unix": started,
            "end_unix": ended,
            "valid": record["valid"],
            "repeats_in_window": 1,
            # The counter delta over the bench command alone. Null when the counter was not read.
            "energy_j": (
                (energy_after - energy_before) / 1000.0
                if energy_before is not None and energy_after is not None and energy_after >= energy_before
                else None
            ),
        }
        if record["validity"] != "invalid" or attempt == attempts:
            break
        kind = record.get("invalidation_reason")
        if kind == "throttled":
            _warn(
                ctx,
                f"Speed cell {cell.name} throttled on attempt {attempt}. Cooling down and retrying.",
            )
            _cool(ctx)
        elif kind == "bench_failed":
            _warn(
                ctx,
                f"Speed cell {cell.name} bench failed on attempt {attempt}: {record.get('invalid_reason')}. "
                "Retrying after backoff. This is not a thermal or power throttle, so there is no cool-down.",
            )
            if ctx.cfg.retries.backoff_s > 0:
                time.sleep(ctx.cfg.retries.backoff_s)
        else:
            _warn(
                ctx,
                f"Speed cell {cell.name} was invalid on attempt {attempt} ({kind}): {record.get('invalid_reason')}. "
                "Retrying without a cool-down.",
            )
    if last is None or last_window is None:
        raise SmaError(f"speed cell {cell.name} produced no result")
    atomic_write_json(ctx.run_dir / "speed" / "cells" / cell.name / ".done", {"attempt": last["attempt"]})
    return last, last_window


def _derive_percentile(cfg_stat: str) -> int:
    if cfg_stat in {"p50", "p90", "p99"}:
        return int(cfg_stat[1:])
    raise SmaError(f"speed.derive_stat must be p50, p90, or p99, not {cfg_stat}")


def _cell_record(
    ctx: Ctx,
    cell: Any,
    attempt: int,
    bench_path: Path,
    samples: list[Any],
    started: float,
    ended: float,
) -> dict[str, Any]:
    from sma.telemetry import coverage_problem, invalidating_throttle

    payload = load_bench(bench_path) or {}
    stat = _derive_percentile(ctx.cfg.speed.derive_stat)
    ttft = {p: percentile_ms(payload, "ttft", p) for p in PERCENTILES}
    tpot = {p: percentile_ms(payload, "tpot", p) for p in PERCENTILES}
    itl = {p: percentile_ms(payload, "itl", p) for p in PERCENTILES}
    e2el = {p: percentile_ms(payload, "e2el", p) for p in PERCENTILES}
    unverified = [sample for sample in samples if getattr(sample, "flags_unverified", False)]
    throttled = [
        sample
        for sample in samples
        if invalidating_throttle(sample, sw_power_cap_invalidates=ctx.cfg.telemetry.sw_power_cap_invalidates)
    ]
    measurement_note = None
    bench_problem = bench_request_problem(payload, ctx.cfg.speed.num_prompts)
    if bench_problem:
        validity = "invalid"
        invalidation_reason = "bench_failed"
        reason = bench_problem
    elif ctx.telemetry is None:
        validity = "unverified"
        invalidation_reason = "telemetry_gap"
        reason = "telemetry was not recorded for this speed cell"
    elif not samples:
        validity = "unverified"
        invalidation_reason = "telemetry_gap"
        reason = "no telemetry samples in this speed cell"
    else:
        gap = coverage_problem(
            samples,
            start=started,
            end=ended,
            interval_s=ctx.cfg.telemetry.interval_s,
            min_ratio=ctx.cfg.telemetry.min_sample_ratio,
            max_gap_s=ctx.cfg.telemetry.max_sample_gap_s,
        )
        if gap:
            validity = "unverified"
            invalidation_reason = "telemetry_gap"
            reason = gap
        elif unverified:
            validity = "unverified"
            invalidation_reason = "telemetry_gap"
            reason = "nvidia-smi throttle flags were N/A or unreadable"
        elif throttled:
            validity = "invalid"
            invalidation_reason = "throttled"
            reason = f"thermal or power throttle on {len(throttled)} telemetry sample(s)"
        else:
            validity = "valid"
            invalidation_reason = None
            reason = None
    if ttft.get(stat) is None or tpot.get(stat) is None:
        measurement_note = (
            f"Bench JSON has no {ctx.cfg.speed.derive_stat} TTFT/TPOT. "
            "Derived tok/s stays null. Nothing was invented."
        )
    return {
        "name": cell.name,
        "shape": cell.shape,
        "concurrency": cell.concurrency,
        "input_len": cell.input_len,
        "output_len": cell.output_len,
        "repeat": cell.repeat,
        "attempt": attempt,
        "valid": validity == "valid",
        "validity": validity,
        "invalid_reason": reason,
        "invalidation_reason": invalidation_reason,
        "measurement_note": measurement_note,
        "ttft_ms_p50": ttft[50],
        "ttft_ms_p90": ttft[90],
        "ttft_ms_p99": ttft[99],
        "tpot_ms_p50": tpot[50],
        "tpot_ms_p90": tpot[90],
        "tpot_ms_p99": tpot[99],
        "itl_ms_p50": itl[50],
        "itl_ms_p90": itl[90],
        "itl_ms_p99": itl[99],
        "e2el_ms_p50": e2el[50],
        "e2el_ms_p90": e2el[90],
        "e2el_ms_p99": e2el[99],
        "output_tok_s": output_tok_s(payload),
        "prefill_tok_s": prefill_tok_s(cell.input_len, ttft[stat]) if cell.concurrency == 1 else None,
        "decode_tok_s": decode_tok_s(tpot[stat]) if cell.concurrency == 1 else None,
    }


def _warmup(ctx: Ctx) -> None:
    done = ctx.run_dir / "speed" / "warmup.done"
    if done.is_file():
        return
    started = utc_now()
    if ctx.telemetry is not None:
        ctx.telemetry.set_phase("speed:warmup")
    for _ in range(ctx.cfg.speed.warmup_requests):
        from sma.httpapi import request_json

        request_json(
            "POST",
            ctx.base_url.rstrip("/") + "/v1/completions",
            timeout_s=ctx.cfg.timeouts.logprob_request_s,
            payload={
                "model": ctx.served_name,
                "prompt": "warmup",
                "max_tokens": ctx.cfg.speed.warmup_max_tokens,
                "temperature": 0,
                "seed": ctx.cfg.seed,
            },
        )
    if ctx.cfg.speed.warmup_soak_s > 0:
        time.sleep(ctx.cfg.speed.warmup_soak_s)
    atomic_write_json(
        ctx.run_dir / "speed" / "warmup.json",
        {
            "requests": ctx.cfg.speed.warmup_requests,
            "max_tokens": ctx.cfg.speed.warmup_max_tokens,
            "soak_s": ctx.cfg.speed.warmup_soak_s,
            "started": started,
            "finished": utc_now(),
        },
    )
    atomic_write_json(done, {"finished": utc_now()})


def _preflight_image(ctx: Ctx) -> str:
    from sma.engines import image_ref

    return image_ref(ctx.manifest["engine"]["image"], ctx.manifest["engine"].get("image_digest"))


def _remember_registry_digest(ctx: Ctx) -> None:
    registry = getattr(ctx.services, "registry", None)
    digest = getattr(registry, "last_digest", None)
    if digest and not ctx.manifest["engine"].get("image_digest"):
        ctx.manifest["engine"]["image_digest"] = digest


def _install_sglang_dataset(ctx: Ctx) -> None:
    if ctx.manifest["engine"]["name"] != "sglang":
        return
    import hashlib
    import json

    from sma.backends import SGLANG_DATASET_NAME

    source = Path(__file__).resolve().parent / "data" / SGLANG_DATASET_NAME
    meta_path = source.with_name(source.name.replace(".json", ".meta.json"))
    if not source.is_file():
        raise SmaError(f"Pinned SGLang dataset is missing: {source.name}")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sha256") and meta.get("sha256") != digest:
            raise SmaError("Pinned SGLang dataset hash does not match its metadata.")
    dest = ctx.run_dir / "speed" / SGLANG_DATASET_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    atomic_write_json(
        ctx.run_dir / "speed" / "sglang-random-dataset.meta.json",
        {"sha256": digest, "rows": 256, "container_path": "/sma-run/speed/" + SGLANG_DATASET_NAME},
    )


def _health_kwargs(ctx: Ctx) -> dict[str, Any]:
    from sma.backends import get_backend

    backend = get_backend(ctx.manifest["engine"]["name"])
    return {"engine_label": backend.name, "health_paths": backend.health_paths()}


def _scrape(ctx: Ctx) -> tuple[str, str | None]:
    from sma.backends import get_backend

    backend = get_backend(ctx.manifest["engine"]["name"])
    try:
        return (
            scrape_metrics(
                ctx.base_url,
                ctx.cfg.timeouts.metrics_s,
                path=backend.metrics_path(),
            ),
            None,
        )
    except SmaError as exc:
        return "", scrub_text(str(exc))


def _remember_digest(ctx: Ctx) -> None:
    if ctx.manifest["engine"].get("image_digest"):
        return
    digest = ctx.services.runtime.ensure_image(_resolved_image(ctx))
    if digest:
        ctx.manifest["engine"]["image_digest"] = digest


def _start_server(ctx: Ctx, mode: str) -> None:
    if ctx.snapshot is None or not ctx.snapshot.exists():
        raise SmaError("Snapshot path is missing. Re-run starts at fetch if that step is not done.")
    quant = ctx.manifest["quant_flag"]
    is_moe = bool(ctx.manifest.get("is_moe"))
    use_chain = (
        ctx.manifest["engine"]["name"] == "vllm"
        and is_moe
        and quant in {"nvfp4", "mxfp4", "mixed"}
    )
    chain: list[str | None]
    if use_chain:
        chain = list(moe_backend_chain(ctx.cfg))
        used = ctx.manifest["engine"].get("moe_backend_used")
        if used:
            chain = [used] + [name for name in chain if name != used]
    else:
        chain = [None]
    batch_inv = mode == "quality" and ctx.batch_invariant
    errors: list[str] = []
    attempts = max(1, ctx.cfg.retries.attempts)
    for index, backend in enumerate(chain):
        spec = _start_spec(ctx, mode, backend, batch_inv)
        for attempt in range(1, attempts + 1):
            try:
                ctx.services.runtime.start(spec)
                _follow(ctx, mode, spec.name)
                started = time.monotonic()
                wait_until_healthy(
                    ctx.base_url,
                    deadline_s=float(ctx.cfg.timeouts.server_start_s),
                    request_timeout_s=ctx.cfg.timeouts.health_request_s,
                    poll_s=ctx.cfg.timeouts.health_poll_s,
                    alive=lambda: ctx.services.runtime.is_running(spec.name),
                    on_wait=lambda elapsed: report_wait(ctx.run_dir, mode, elapsed),
                    **_health_kwargs(ctx),
                )
                log_event("server_ready", mode=mode, elapsed_s=round(time.monotonic() - started))
                primary = moe_backend_chain(ctx.cfg)[0] if use_chain else None
                fallback = bool(use_chain and backend and backend != primary)
                _record_start(ctx, mode, spec, backend, fallback=fallback)
                if mode == "speed":
                    warmup_done = ctx.run_dir / "speed" / "warmup.done"
                    if warmup_done.exists():
                        warmup_done.unlink()
                return
            except Exception as exc:
                last = tail_lines(server_log_path(ctx.run_dir, mode), 3)
                detail = f" Last server log lines: {' | '.join(last)}" if last else ""
                errors.append(f"{backend or 'default'} attempt {attempt}: {scrub_text(str(exc))}{detail}")
                try:
                    ctx.services.runtime.stop(spec.name)
                except Exception:
                    pass
                _unfollow(ctx, mode)
                if attempt < attempts and not use_chain:
                    time.sleep(ctx.cfg.retries.backoff_s * (ctx.cfg.retries.backoff_factor ** (attempt - 1)))
                    continue
                if attempt < attempts and _daemon_failure(str(exc)):
                    time.sleep(ctx.cfg.retries.backoff_s)
                    continue
                break
    joined = " | ".join(errors)[-2000:]
    engine_name_now = ctx.manifest["engine"]["name"]
    raise SmaError(
        f"{engine_name_now} failed to become healthy. The checkpoint was not modified. "
        f"Attempts: {joined}"
    )


def _record_start(ctx: Ctx, mode: str, spec: StartSpec, backend: str | None, fallback: bool) -> None:
    engine = ctx.manifest["engine"]
    if backend:
        engine["moe_backend_used"] = backend
        engine["moe_backend_fallback"] = fallback
        if mode == "quality":
            engine["moe_backend_quality"] = backend
            engine["moe_backend_quality_fallback"] = fallback
        else:
            engine["moe_backend_speed"] = backend
            engine["moe_backend_speed_fallback"] = fallback
        if fallback:
            _warn(
                ctx,
                f"MoE backend fell back to {backend}. The result is labeled with that backend "
                "and is not the default flashinfer_b12x path.",
            )
    elif engine.get("moe_backend_used") is None:
        engine["moe_backend_used"] = None
    if mode == "quality":
        engine["flags_quality"] = list(spec.serve_argv)
        engine["env_quality"] = dict(spec.env)
        engine["batch_invariant_quality"] = "VLLM_BATCH_INVARIANT" in spec.env
        engine["no_async_scheduling_quality"] = "--no-async-scheduling" in spec.serve_argv
        engine["flashinfer_autotune_quality"] = "--no-enable-flashinfer-autotune" not in spec.serve_argv
        if not ctx.manifest.get("is_moe") and ctx.manifest["quant_flag"] == "nvfp4":
            engine["linear_backend"] = ctx.cfg.engine.linear_backend
    else:
        engine["flags_speed"] = list(spec.serve_argv)
        engine["env_speed"] = dict(spec.env)
    try:
        text = ctx.services.runtime.logs(spec.name)
    except Exception as exc:
        text = scrub_text(str(exc))
    if mode not in ctx.followers:
        # A followed log is still being written. Replacing the file would cut it off here.
        _write_log(server_log_path(ctx.run_dir, mode), text)
    note = _backend_note(text)
    if note:
        engine["resolved_backend_note"] = note
    short = kv_capacity_warning(text, needed_kv_tokens(ctx.cfg))
    if short:
        _warn(ctx, short)
    publish(ctx.cfg, ctx.run_dir, ctx.manifest)


def _resolved_image(ctx: Ctx) -> str:
    name = ctx.manifest["engine"]["name"]
    return image_ref(image_for(ctx.cfg, name), ctx.manifest["engine"].get("image_digest"))


def _start_spec(ctx: Ctx, mode: str, backend: str | None, batch_invariant: bool) -> StartSpec:
    name_engine = ctx.manifest["engine"]["name"]
    serve = serve_argv(
        ctx.cfg,
        mode=mode,
        served_name=ctx.served_name,
        is_moe=bool(ctx.manifest.get("is_moe")),
        quant_flag=str(ctx.manifest["quant_flag"]),
        moe_backend=backend,
        batch_invariant=batch_invariant,
        engine=name_engine,
        tool_parser=ctx.manifest["engine"].get("tool_call_parser"),
    )
    env = serve_env(ctx.cfg, mode=mode, batch_invariant=batch_invariant, engine=name_engine)
    name = container_name(ctx.cfg, ctx.run_id, mode)
    assert ctx.snapshot is not None
    command = docker_argv(
        ctx.cfg,
        name=name,
        model_dir=ctx.snapshot,
        run_dir=ctx.run_dir,
        serve=serve,
        env=env,
        image=_resolved_image(ctx),
    )
    return StartSpec(
        name=name,
        docker_argv=command,
        env=env,
        serve_argv=serve,
        port=ctx.cfg.engine.port,
        mode=mode,
    )


def _ensure_server(ctx: Ctx, mode: str) -> None:
    name = container_name(ctx.cfg, ctx.run_id, mode)
    if ctx.services.runtime.is_running(name):
        if mode not in ctx.followers:
            _follow(ctx, mode, name)
        try:
            wait_until_healthy(
                ctx.base_url,
                deadline_s=min(30.0, float(ctx.cfg.timeouts.server_start_s)),
                request_timeout_s=ctx.cfg.timeouts.health_request_s,
                poll_s=ctx.cfg.timeouts.health_poll_s,
                **_health_kwargs(ctx),
            )
            return
        except SmaError:
            log_event("server_unhealthy", name=name)
    other = "speed" if mode == "quality" else "quality"
    _stop_mode(ctx, other)
    _start_server(ctx, mode)


def _stop_mode(ctx: Ctx, mode: str) -> None:
    name = container_name(ctx.cfg, ctx.run_id, mode)
    following = ctx.followers.get(mode)
    try:
        if ctx.services.runtime.is_running(name) or name in ctx.services.runtime.list_sma():
            text = ""
            try:
                text = ctx.services.runtime.logs(name)
            except Exception:
                text = ""
            ctx.services.runtime.stop(name)
            _unfollow(ctx, mode)
            # The followed log interleaves stdout and stderr as they arrive. The dump puts all stdout first.
            path = server_log_path(ctx.run_dir, mode)
            followed = following is not None and path.is_file() and path.stat().st_size > 0
            if text and not followed:
                _write_log(path, text)
    except Exception as exc:
        log_event("stop_failed", name=name, error=str(exc))
    finally:
        _unfollow(ctx, mode)


def _follow(ctx: Ctx, mode: str, name: str) -> None:
    _unfollow(ctx, mode)
    following = start_following(ctx.services.runtime, name, ctx.run_dir, mode)
    if following is not None:
        ctx.followers[mode] = following


def _unfollow(ctx: Ctx, mode: str) -> None:
    following = ctx.followers.pop(mode, None)
    if following is not None:
        following.stop()


def _teardown_containers(ctx: Ctx) -> None:
    if ctx.telemetry is not None:
        ctx.telemetry.set_phase("teardown")
    for mode in ("quality", "speed"):
        _stop_mode(ctx, mode)
    leftover = [name for name in ctx.services.runtime.list_sma() if ctx.run_id in name]
    if leftover:
        raise SmaError(f"Containers still present after teardown: {', '.join(leftover)}")


def _guard_containers(ctx: Ctx) -> None:
    for name in ctx.services.runtime.list_sma():
        if ctx.run_id in name:
            ctx.services.runtime.stop(name)
            continue
        raise SmaError(
            f"Container {name} is still on this Spark. SMA runs one model at a time and "
            "does not use tensor parallelism. Stop it with `sma cleanup` if it is a leftover."
        )


def _cleanup(ctx: Ctx) -> None:
    page = {"attempted": False, "permitted": None, "ok": None, "error": None}
    if ctx.cfg.telemetry.drop_page_cache:
        page = drop_page_cache()
        if page.get("ok") is not True:
            _warn(ctx, "Page cache was not dropped. The next model may see leftover file cache.")
    cool = _cool(ctx)
    atomic_write_json(
        ctx.run_dir / "cleanup.json",
        {"page_cache": page, "idle_until_cool": cool},
    )


def _cool(ctx: Ctx) -> dict[str, Any]:
    if ctx.telemetry is None:
        return {"reached": None, "note": "no telemetry recorder"}

    def _read() -> float | None:
        assert ctx.telemetry is not None
        return ctx.telemetry.latest_temp()

    return idle_until_cool(
        _read,
        target_c=ctx.cfg.telemetry.idle_until_cool_c,
        timeout_s=ctx.cfg.telemetry.idle_timeout_s,
        poll_s=ctx.cfg.telemetry.poll_s,
    )


def _maybe_clock_cap(ctx: Ctx) -> None:
    cap = ctx.cfg.engine.clock_cap_mhz
    if cap is None or not ctx.cfg.engine.apply_clock_cap:
        return
    argv = ["nvidia-smi", "-i", str(ctx.cfg.doctor.gpu_index), "-lgc", f"{cap},{cap}"]
    require_ok(run_command(argv, timeout_s=30, what="nvidia-smi clock cap"), what="clock cap")
    log_event("clock_cap", mhz=cap)


def _load_encoder(ctx: Ctx) -> Callable[[str], list[int]]:
    if ctx.snapshot is None:
        raise SmaError("No snapshot is available to load tokenizer.json")
    path = ctx.snapshot / "tokenizer.json"
    if not path.is_file():
        raise SmaError(
            "tokenizer.json is missing from the snapshot. Logprobs need a fast tokenizer "
            "and do not apply a chat template. SentencePiece-only tokenizers are not loaded in v0."
        )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.post_processor = None

    def _encode(text: str) -> list[int]:
        return list(tokenizer.encode(text).ids)

    return _encode


def _backend_note(text: str) -> str | None:
    lines = []
    for line in text.splitlines():
        folded = line.lower()
        if "backend" in folded and any(word in folded for word in ("moe", "linear", "marlin", "flashinfer", "b12x")):
            lines.append(scrub_text(line.strip())[:240])
    if not lines:
        return None
    return " | ".join(lines[:8])[:1000]


def _daemon_failure(text: str) -> bool:
    folded = text.lower()
    return "docker daemon" in folded or "cannot connect to the docker" in folded


def _optional(path: Path) -> Any:
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = scrub_text(text)
    if not body.endswith("\n"):
        body += "\n"
    atomic_write_text(path, body)


class _SignalGuard:
    def __init__(self, teardown: Callable[[], None]) -> None:
        self._teardown = teardown
        self._prev_int: Any = None
        self._prev_term: Any = None

    def __enter__(self) -> "_SignalGuard":
        self._prev_int = signal.getsignal(signal.SIGINT)
        self._prev_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)
        return self

    def _handle(self, signum: int, _frame: Any) -> None:
        try:
            self._teardown()
        finally:
            raise KeyboardInterrupt(f"signal {signum}")

    def __exit__(self, *exc: object) -> None:
        if self._prev_int is not None:
            signal.signal(signal.SIGINT, self._prev_int)
        if self._prev_term is not None:
            signal.signal(signal.SIGTERM, self._prev_term)
