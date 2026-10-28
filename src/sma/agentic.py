"""KernelBench Level 1 and SWE-bench Verified-mini as steps of a run (owner decision 2026-10-27).

Both run against the quality server with the model's own thinking mode and sampling. Each writes
quality/<suite>/scores.json in the same shape as the other suites, so the summary lists them together.
The pipeline passes in its own server and warning helpers; this module does not import the pipeline.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sma import kernelbench as kb
from sma import swebench as sb
from sma.atomic import atomic_write_bytes, atomic_write_json, atomic_write_text
from sma.errors import SmaError
from sma.httpapi import ApiError, request_json
from sma.kbservice import finished, service_from_env
from sma.logsetup import log_event
from sma.qualityout import wilson_interval
from sma.records import large_file_hashes
from sma.scrub import scrub_obj, scrub_text
from sma.thinking import chat_template, thinking_default

KERNELBENCH_SUITE = "kernelbench_l1"
# The service's own defaults (KernelBench's eval_from_generations.py); they name the baseline file.
KERNELBENCH_DEFAULT_OPTIONS = {"precision": "fp32", "timing_method": "cuda_event", "num_perf_trials": 100}
# The service refuses a kernel over 512 KiB.
_KERNEL_LIMIT = 512 * 1024
_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
_TUNNEL_RESTARTS = 10

Warn = Callable[[Any, str], None]
EnsureServer = Callable[[Any, str], None]


def model_defaults(ctx: Any) -> dict[str, Any]:
    """The thinking mode and sampling someone running this checkpoint gets."""
    model_file = (ctx.manifest.get("gguf") or {}).get("model_file")
    try:
        template = chat_template(ctx.snapshot, model_file) if ctx.snapshot else None
    except (OSError, ValueError, SmaError) as exc:
        thinking = {"enable_thinking": None, "how": f"the chat template could not be read ({type(exc).__name__})"}
    else:
        thinking = thinking_default(template)
    if model_file:
        sampling: dict[str, Any] = {"source": "llama-server's own defaults; a GGUF has no generation_config.json"}
    else:
        path = ctx.snapshot / "generation_config.json" if ctx.snapshot else None
        found = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.is_file() else {}
        source = (
            "the checkpoint's generation_config.json; requests set no sampling field, so the engine applies it"
            if found
            else "the engine's defaults; the checkpoint has no generation_config.json"
        )
        sampling = {"source": source, **{key: found[key] for key in _SAMPLING_KEYS if key in found}}
    return {"thinking": thinking, "sampling": sampling}


def _skip(ctx: Any, warn: Warn, suite: str, message: str) -> None:
    warn(ctx, message)
    atomic_write_json(
        ctx.run_dir / "quality" / suite / "scores.json",
        {"suite": suite, "scorer": "not run", "primary_metric": None, "primary_score": None, "error": message},
    )


def _context(ctx: Any, need: str) -> int | None:
    return ((ctx.manifest.get("memory_plan") or {}).get("benchmarks") or {}).get(need)


# KernelBench


def kernelbench_step(ctx: Any, *, ensure_server: EnsureServer, warn: Warn) -> None:
    """Ask the model for a kernel for each Level 1 problem, then submit them to the service in one job."""
    cfg = ctx.cfg.kernelbench
    suite_dir = ctx.run_dir / "quality" / KERNELBENCH_SUITE
    if not cfg.enabled:
        return  # Off in config; the config snapshot says so, and the summary has no KernelBench suite.
    service = ctx.services.kernelbench or service_from_env()
    if service is None:
        _skip(ctx, warn, KERNELBENCH_SUITE, "KERNELBENCH_URL is not set (see .env.example). KernelBench did not run.")
        return
    if (suite_dir / "job.json").is_file():
        return
    ensure_server(ctx, "quality")
    problems = kb.load_problems(cfg.level)
    defaults = model_defaults(ctx)
    server_context = int((ctx.manifest.get("memory_plan") or {}).get("context") or cfg.max_tokens)
    # Room for the prompt: a Level 1 prompt is a few thousand tokens.
    max_tokens = max(1024, min(cfg.max_tokens, server_context - 4096))
    thinking = defaults["thinking"]["enable_thinking"]
    atomic_write_json(
        suite_dir / "settings.json",
        {
            **defaults,
            "level": cfg.level,
            "max_tokens": max_tokens,
            "concurrency": cfg.concurrency,
            "prompt": f"KernelBench {kb.COMMIT[:8]} one-shot CUDA fp32, as a chat with KernelBench's system prompt",
            "system_prompt": kb.SYSTEM_PROMPT,
            "extraction": "KernelBench's extract_first_code after the reasoning block",
        },
    )
    generations = suite_dir / "generations"

    def write_kernel(problem: Any) -> None:
        body: dict[str, Any] = {"model": ctx.served_name, "messages": kb.chat_messages(problem), "max_tokens": max_tokens}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        record: dict[str, Any] = {"problem_id": problem.problem_id, "name": problem.name}
        try:
            reply = request_json(
                "POST",
                ctx.base_url + "/v1/chat/completions",
                timeout_s=cfg.request_timeout_s,
                payload=body,
                attempts=max(1, ctx.cfg.retries.attempts),
                backoff_s=ctx.cfg.retries.backoff_s,
            )
        except ApiError:
            raise  # The server is down or timed out: the step fails, and a resume asks again.
        except SmaError as exc:
            # The server refused this one request (a 4xx). The problem counts as a failure, as in KernelBench.
            record.update(error=str(exc), kernel=None)
        else:
            choice = (reply.get("choices") or [{}])[0] if isinstance(reply, dict) else {}
            message = choice.get("message") or {}
            content = message.get("content")
            kernel = kb.extract_kernel(content)
            if kernel is not None and len(kernel.encode("utf-8")) > _KERNEL_LIMIT:
                record["error"] = "The kernel is over the service's 512 KiB limit."
                kernel = None
            record.update(
                finish_reason=choice.get("finish_reason"),
                usage=reply.get("usage"),
                content=content,
                reasoning=message.get("reasoning_content") or message.get("reasoning"),
                kernel=kernel,
            )
        atomic_write_json(generations / f"{problem.problem_id}.json", scrub_obj(record))

    todo = [problem for problem in problems if not (generations / f"{problem.problem_id}.json").is_file()]
    log_event("kernelbench_generate", problems=len(todo), run_id=ctx.run_id)
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        for _ in pool.map(write_kernel, todo):
            pass
    records = [json.loads((generations / f"{p.problem_id}.json").read_text(encoding="utf-8")) for p in problems]
    kernels = [
        {"problem_id": row["problem_id"], "label": f"{ctx.run_id}/{row['problem_id']}", "kernel_src": row["kernel"]}
        for row in records
        if row.get("kernel")
    ]
    environment = service.environment()
    job_id = service.submit(level=cfg.level, name=ctx.run_id, kernels=kernels, options=cfg.options) if kernels else None
    atomic_write_json(
        suite_dir / "job.json",
        scrub_obj(
            {
                "job_id": job_id,
                "level": cfg.level,
                "kernels": len(kernels),
                "no_kernel": [row["problem_id"] for row in records if not row.get("kernel")],
                "options": cfg.options,
                "environment": environment,
            }
        ),
    )


def kernelbench_results_step(ctx: Any, *, warn: Warn) -> None:
    """Wait for the service to finish the job, then score it with KernelBench's own formulas."""
    cfg = ctx.cfg.kernelbench
    suite_dir = ctx.run_dir / "quality" / KERNELBENCH_SUITE
    job_path = suite_dir / "job.json"
    if not job_path.is_file():
        return  # KernelBench did not run; its scores.json says why.
    job = json.loads(job_path.read_text(encoding="utf-8"))
    service = ctx.services.kernelbench or service_from_env()
    if service is None:
        raise SmaError("KERNELBENCH_URL is not set, and this run's KernelBench job is waiting for its results. Set it and resume.")
    state: dict[str, Any] = {"status": "done", "results": [], "options": {}, "environment": job.get("environment") or {}}
    if job.get("job_id"):
        deadline = time.monotonic() + cfg.wait_s
        while True:
            state = service.job(job["job_id"])
            if finished(state):
                break
            if time.monotonic() > deadline:
                raise SmaError(
                    f"KernelBench job {job['job_id']} did not finish within {cfg.wait_s:.0f} s. Resume the run to keep waiting."
                )
            log_event("kernelbench_wait", job=job["job_id"], progress=state.get("progress"), run_id=ctx.run_id)
            time.sleep(cfg.poll_s)
        atomic_write_json(suite_dir / "results.json", scrub_obj(state))
    base = {
        "suite": KERNELBENCH_SUITE,
        "primary_metric": "fast_1",
        "limit": None,
        "job_id": job.get("job_id"),
        "context_per_stream": _context(ctx, "kernelbench"),
        "concurrency": cfg.concurrency,
    }
    if state.get("status") != "done":
        message = f"The KernelBench job ended {state.get('status')}: {state.get('error')}. The score is null."
        warn(ctx, message)
        atomic_write_json(suite_dir / "scores.json", {**base, "scorer": "not scored", "primary_score": None, "error": message})
        return
    options = {**KERNELBENCH_DEFAULT_OPTIONS, **cfg.options, **(state.get("options") or {})}
    fingerprint = str((state.get("environment") or {}).get("fingerprint") or "")
    baseline_ids = service.baseline_ids(level=cfg.level, options=options, fingerprint=fingerprint)
    scored = kb.score_level(kb.load_problems(cfg.level), list(state.get("results") or []), baseline_ids)
    fast_1 = scored["fast_p"]["1.0"]
    n = scored["with_baseline"]
    lower, upper = wilson_interval(round(fast_1 * n), n) if fast_1 is not None and n else (None, None)
    atomic_write_json(
        suite_dir / "scores.json",
        {
            **base,
            "scorer": f"KernelBench {kb.COMMIT[:8]} score.py, timed on the KernelBench service (environment {fingerprint})",
            "primary_score": fast_1,
            "n": n,
            "ci_lower": lower,
            "ci_upper": upper,
            "fingerprint": fingerprint,
            "options": options,
            "kernelbench": scored,
            "error": None,
        },
    )


# SWE-bench


def swebench_step(ctx: Any, *, ensure_server: EnsureServer, warn: Warn) -> None:
    """Run mini-swe-agent on the runner against this run's server, grade the patches, and score them."""
    cfg = ctx.cfg.swebench
    suite_dir = ctx.run_dir / "quality" / sb.SUITE
    if not cfg.enabled:
        return  # Off in config; the config snapshot says so, and the summary has no SWE-bench suite.
    runner = ctx.services.swe_runner or sb.runner_from_env(ctx.cfg)
    if runner is None:
        _skip(ctx, warn, sb.SUITE, "SWE_RUNNER_HOST, SWE_RUNNER_USER, and SWE_RUNNER_SSH_KEY are not all set (see .env.example). SWE-bench did not run.")
        return
    if not ctx.manifest["engine"].get("tool_call_parser"):
        _skip(ctx, warn, sb.SUITE, "mini-swe-agent's swebench.yaml acts through a bash tool call, and this run's server has no tool-call parser. Pass --tool-parser. The score is null.")
        return
    runner.check(ctx.cfg)
    ensure_server(ctx, "quality")
    defaults = model_defaults(ctx)
    overlay = sb.agent_overlay(
        served_name=ctx.served_name,
        port=int(ctx.cfg.engine.port),
        step_limit=cfg.step_limit,
        enable_thinking=defaults["thinking"]["enable_thinking"],
    )
    rel = sb.run_rel(ctx.run_id)
    atomic_write_json(
        suite_dir / "settings.json",
        scrub_obj(
            {
                **defaults,
                "workers": cfg.workers,
                "step_limit": cfg.step_limit,
                "agent_overlay": overlay,
                "agent_command": sb.agent_command(ctx.cfg, ctx.run_id),
                "grade_command": sb.grade_command(ctx.cfg, ctx.run_id),
            }
        ),
    )
    runner.write_text(f"{rel}/sma-agent.yaml", sb.overlay_yaml(overlay))
    total = sb.dataset_size(ctx.cfg, runner)
    tunnel_log = ctx.run_dir / "logs" / "swebench-tunnel.log"
    tunnel = [runner.tunnel(int(ctx.cfg.engine.port), tunnel_log)]
    try:
        for launch in range(2):
            code, alive = runner.status(rel, "agent")
            if not alive:
                if code is not None and not sb.unanswered(sb.task_statuses(runner, ctx.run_id), total):
                    break  # An earlier attempt finished every task.
                sb.prune_unanswered(runner, ctx.run_id)
                runner.start(rel, "agent", sb.agent_command(ctx.cfg, ctx.run_id))
                log_event("swebench_start", launch=launch + 1, run_id=ctx.run_id)
            _wait_remote(ctx, runner, rel, "agent", cfg.agent_wait_s, tunnel=tunnel, tunnel_log=tunnel_log)
            if not sb.unanswered(sb.task_statuses(runner, ctx.run_id), total):
                break
    finally:
        tunnel[0].stop(timeout_s=1)
    statuses = sb.task_statuses(runner, ctx.run_id)
    report = None
    if not sb.unanswered(statuses, total):
        code, alive = runner.status(rel, "grade")
        if code != 0:
            if not alive:
                runner.start(rel, "grade", sb.grade_command(ctx.cfg, ctx.run_id))
            _wait_remote(ctx, runner, rel, "grade", cfg.grade_wait_s, tunnel=None, tunnel_log=None)
        text = runner.read_text(f"{rel}/report.json")
        report = json.loads(text) if text else None
    _collect(ctx, runner, suite_dir, rel, statuses, report)
    scores = sb.score(report, statuses, total)
    scores.update(context_per_stream=_context(ctx, "swebench"), concurrency=cfg.workers)
    if scores["error"]:
        warn(ctx, scores["error"])
    atomic_write_json(suite_dir / "scores.json", scores)


def _wait_remote(ctx: Any, runner: Any, rel: str, name: str, wait_s: float, *, tunnel: list[Any] | None, tunnel_log: Path | None) -> int:
    deadline = time.monotonic() + wait_s
    restarts = 0
    while True:
        code, alive = runner.status(rel, name)
        if code is not None:
            return code
        if not alive:
            raise SmaError(f"SWE-bench's {name} stopped on the runner without an exit status. Resume the run to start it again.")
        if tunnel is not None and tunnel[0].process.poll() is not None:
            restarts += 1
            if restarts > _TUNNEL_RESTARTS:
                raise SmaError("The SSH tunnel to the SWE-bench runner keeps closing. See logs/swebench-tunnel.log, then resume the run.")
            log_event("swebench_tunnel_restart", restarts=restarts, run_id=ctx.run_id)
            tunnel[0] = runner.tunnel(int(ctx.cfg.engine.port), tunnel_log)
        if time.monotonic() > deadline:
            runner.stop(ctx.run_id)
            raise SmaError(f"SWE-bench's {name} did not finish within {wait_s:.0f} s. SMA stopped it on the runner.")
        log_event("swebench_wait", step=name, run_id=ctx.run_id)
        time.sleep(ctx.cfg.swebench.poll_s)


def _collect(ctx: Any, runner: Any, suite_dir: Path, rel: str, statuses: list[dict[str, Any]], report: dict[str, Any] | None) -> None:
    """Bring the patches, the report, the per-task statuses, the logs, and the trajectories back, scrubbed."""
    atomic_write_text(suite_dir / "tasks.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in statuses))
    preds = runner.read_text(f"{rel}/preds.json")
    if preds:
        atomic_write_json(suite_dir / "preds.json", scrub_obj(json.loads(preds)))
    if report is not None:
        atomic_write_json(suite_dir / "report.json", scrub_obj(report))
    for name in ("agent", "grade"):
        text = runner.read_text(f"{rel}/{name}.log")
        if text:
            atomic_write_text(ctx.run_dir / "logs" / f"swebench-{name}.log", scrub_text(text))
    lines = sb.trajectories(runner, ctx.run_id)
    if lines:
        data = "".join(scrub_text(line) + "\n" for line in lines).encode("utf-8")
        atomic_write_bytes(suite_dir / "trajectories.jsonl.gz", gzip.compress(data, mtime=0))
        # The bundle stays out of git (records.LARGE_FILES); the manifest keeps its hash.
        ctx.manifest["large_files"] = large_file_hashes(ctx.run_dir)
