"""KernelBench Level 1, SWE-bench Verified-mini, and BrowseComp-Plus as steps of a run (owner decisions 2026-10-27/28).

All run against the quality server with the model's own thinking mode and sampling. Each writes
quality/<suite>/scores.json in the same shape as the other suites, so the summary lists them together.
The pipeline passes in its own server and warning helpers; this module does not import the pipeline.
"""

from __future__ import annotations

import gzip
import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from sma import browsecomp as bcp
from sma import kernelbench as kb
from sma import swebench as sb
from sma.atomic import atomic_write_bytes, atomic_write_json, atomic_write_text
from sma.errors import SmaError
from sma.httpapi import ApiError, request_json
from sma.judge import judge_from_env
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


# BrowseComp-Plus

_SEARCH_SERVER = "search_server.py"


def _research_runner(ctx: Any) -> Any:
    return ctx.services.research_runner or sb.runner_from_env(
        ctx.cfg, remote_dir=ctx.cfg.browsecomp.remote_dir, label="BrowseComp-Plus runner"
    )


def _judge(ctx: Any) -> Any:
    cfg = ctx.cfg.browsecomp
    return ctx.services.judge or judge_from_env(
        cfg.judge_model, timeout_s=cfg.judge_timeout_s, attempts=max(1, ctx.cfg.retries.attempts), backoff_s=ctx.cfg.retries.backoff_s
    )


def _bcp_data(runner: Any) -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, list[str]]]:
    """The decrypted questions and answers, and both qrels, from the runner. They stay in memory."""
    rows: dict[str, dict[str, str]] = {}
    for line in (runner.read_text("data/queries.jsonl") or "").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["query_id"])] = {"query": row["query"], "answer": row["answer"]}
    if not rows:
        raise SmaError("The BrowseComp-Plus runner has no data/queries.jsonl. Run services/browsecomp-plus/setup.sh there.")
    evidence = bcp.read_qrels(runner.read_text("data/qrel_evidence.txt") or "")
    golds = bcp.read_qrels(runner.read_text("data/qrel_golds.txt") or "")
    return rows, evidence, golds


def search_command(cfg: Any, run_id: str) -> str:
    bc = cfg.browsecomp
    return (
        f".venv/bin/python {_SEARCH_SERVER} --index indexes/bm25 --tokenizer data/tokenizer "
        f"--port {int(bc.search_port)} --snippet-max-tokens {int(bc.snippet_max_tokens)} --tag {sb.run_rel(run_id)}"
    )


def browsecomp_step(ctx: Any, *, ensure_server: EnsureServer, warn: Warn) -> None:
    """Each picked question: the model searches the runner's BM25 index until it answers. The judge comes later."""
    cfg = ctx.cfg.browsecomp
    suite_dir = ctx.run_dir / "quality" / bcp.SUITE
    if not cfg.enabled:
        return  # Off in config; the config snapshot says so, and the summary has no BrowseComp-Plus suite.
    if _judge(ctx) is None:
        _skip(ctx, warn, bcp.SUITE, "SMA_JUDGE_URL and SMA_JUDGE_API_KEY are not both set (see .env.example), so no answer could be graded. BrowseComp-Plus did not run.")
        return
    runner = _research_runner(ctx)
    if runner is None:
        _skip(ctx, warn, bcp.SUITE, "SWE_RUNNER_HOST, SWE_RUNNER_USER, and SWE_RUNNER_SSH_KEY are not all set (see .env.example). BrowseComp-Plus's search runs there, so it did not run.")
        return
    if not ctx.manifest["engine"].get("tool_call_parser"):
        _skip(ctx, warn, bcp.SUITE, "BrowseComp-Plus's agent searches through a tool call, and this run's server has no tool-call parser. Pass --tool-parser. The score is null.")
        return
    runner.shell(
        f"test -x .venv/bin/python && test -f {_SEARCH_SERVER} && test -d indexes/bm25 && test -d data/tokenizer "
        "&& test -f data/queries.jsonl && test -f data/qrel_evidence.txt && test -f data/qrel_golds.txt",
        what="BrowseComp-Plus runner check (venv, search server, index, tokenizer, queries, qrels)",
        timeout_s=60,
    )
    rows, _evidence, _golds = _bcp_data(runner)
    picked = bcp.pick_queries(list(rows), cfg.limit, ctx.cfg.seed)
    ensure_server(ctx, "quality")
    defaults = model_defaults(ctx)
    thinking = defaults["thinking"]["enable_thinking"]
    atomic_write_json(
        suite_dir / "settings.json",
        scrub_obj(
            {
                **defaults,
                "protocol": f"BrowseComp-Plus {bcp.COMMIT[:8]} search_agent/glm_zai_client.py with its defaults, QUERY_TEMPLATE_NO_GET_DOCUMENT, no system prompt",
                "queries_revision": bcp.QUERIES_REVISION,
                "index": f"Tevatron/browsecomp-plus-indexes bm25 at {bcp.INDEX_REVISION} (Pyserini BM25, k1=0.9, b=0.4)",
                "limit": cfg.limit,
                "seed": ctx.cfg.seed,
                "query_ids": picked,
                "workers": cfg.workers,
                "max_tokens": cfg.max_tokens,
                "max_iterations": cfg.max_iterations,
                "k": cfg.k,
                "snippet_max_tokens": cfg.snippet_max_tokens,
                "search_command": search_command(ctx.cfg, ctx.run_id),
            }
        ),
    )
    answers = suite_dir / "queries"
    todo = [qid for qid in picked if not (answers / f"{qid}.json").is_file()]
    if not todo:
        return
    rel = sb.run_rel(ctx.run_id)
    forward_log = ctx.run_dir / "logs" / "browsecomp-forward.log"
    search_url = f"http://127.0.0.1:{int(cfg.local_port)}"
    runner.start(rel, "search", search_command(ctx.cfg, ctx.run_id))
    forward = [runner.forward(int(cfg.local_port), int(cfg.search_port), forward_log)]
    try:
        _wait_search(ctx, runner, rel, search_url, forward, forward_log)

        def search(query: str) -> str:
            if forward[0].process.poll() is not None:
                log_event("browsecomp_forward_restart", run_id=ctx.run_id)
                forward[0] = runner.forward(int(cfg.local_port), int(cfg.search_port), forward_log)
            reply = request_json("POST", search_url + "/search", timeout_s=120, payload={"query": query, "k": cfg.k},
                                 attempts=max(1, ctx.cfg.retries.attempts), backoff_s=max(1.0, ctx.cfg.retries.backoff_s))
            return str((reply or {}).get("output"))

        def chat(body: dict[str, Any]) -> dict[str, Any]:
            return request_json("POST", ctx.base_url + "/v1/chat/completions", timeout_s=cfg.request_timeout_s, payload=body,
                                attempts=max(1, ctx.cfg.retries.attempts), backoff_s=ctx.cfg.retries.backoff_s)

        def answer(qid: str) -> None:
            record = bcp.run_query(
                bcp.question_text(rows[qid]["query"]), chat=chat, search=search, model=ctx.served_name, k=cfg.k,
                max_tokens=cfg.max_tokens, max_iterations=cfg.max_iterations, enable_thinking=thinking,
            )  # fmt: skip
            atomic_write_json(answers / f"{qid}.json", scrub_obj({"query_id": qid, **record}))

        log_event("browsecomp_start", questions=len(todo), run_id=ctx.run_id)
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            for _ in pool.map(answer, todo):
                pass
    finally:
        forward[0].stop(timeout_s=1)
        runner.stop(ctx.run_id)
        text = runner.read_text(f"{rel}/search.log")
        if text:
            atomic_write_text(ctx.run_dir / "logs" / "browsecomp-search.log", scrub_text(text))


def _wait_search(ctx: Any, runner: Any, rel: str, url: str, forward: list[Any], forward_log: Path) -> None:
    """Until the search server answers through the forward. Its index takes a while to open."""
    deadline = time.monotonic() + ctx.cfg.browsecomp.search_start_s
    while True:
        code, alive = runner.status(rel, "search")
        if code is not None or not alive:
            raise SmaError(f"BrowseComp-Plus's search server stopped on the runner (exit {code}). See logs/browsecomp-search.log.")
        if forward[0].process.poll() is not None:
            forward[0] = runner.forward(int(ctx.cfg.browsecomp.local_port), int(ctx.cfg.browsecomp.search_port), forward_log)
        try:
            health = request_json("GET", url + "/health", timeout_s=10, attempts=1)
        except SmaError:
            health = None
        if isinstance(health, dict) and health.get("ok"):
            log_event("browsecomp_search_ready", docs=health.get("docs"), run_id=ctx.run_id)
            return
        if time.monotonic() > deadline:
            raise SmaError(f"BrowseComp-Plus's search server did not answer within {ctx.cfg.browsecomp.search_start_s:.0f} s. See logs/browsecomp-search.log.")
        time.sleep(2.0)


def browsecomp_results_step(ctx: Any, *, warn: Warn) -> None:
    """Grade each finished answer with the judge, score the run, and bundle the plain text into one git-ignored file."""
    cfg = ctx.cfg.browsecomp
    suite_dir = ctx.run_dir / "quality" / bcp.SUITE
    answers = suite_dir / "queries"
    settings_path = suite_dir / "settings.json"
    if not cfg.enabled or not settings_path.is_file():
        return  # BrowseComp-Plus did not run; its scores.json says why.
    if (suite_dir / "scores.json").is_file() and not answers.is_dir():
        return  # Already scored and bundled.
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    picked = [str(qid) for qid in settings["query_ids"]]
    missing = [qid for qid in picked if not (answers / f"{qid}.json").is_file()]
    if missing:
        raise SmaError(f"{len(missing)} BrowseComp-Plus questions have no answer yet. Resume the run.")
    judge = _judge(ctx)
    if judge is None:
        raise SmaError("SMA_JUDGE_URL and SMA_JUDGE_API_KEY are not both set, and this run's BrowseComp-Plus answers are waiting for the judge. Set them in .env and resume.")
    runner = _research_runner(ctx)
    if runner is None:
        raise SmaError("The SWE_RUNNER_* names are not set, and the judge needs the answers kept on the BrowseComp-Plus runner. Set them in .env and resume.")
    rows, evidence, golds = _bcp_data(runner)
    records = {qid: json.loads((answers / f"{qid}.json").read_text(encoding="utf-8")) for qid in picked}

    def grade(qid: str) -> None:
        path = answers / f"{qid}.judge.json"
        if path.is_file():
            return
        response = bcp.final_response(records[qid])
        if response is None:
            judgment: dict[str, Any] = {"skipped": f"status {records[qid].get('status')}: no final answer to judge, so it counts as wrong"}
        else:
            prompt = bcp.judge_prompt(bcp.question_text(rows[qid]["query"]), response, rows[qid]["answer"])
            reply = judge.ask(prompt, max_tokens=cfg.judge_max_tokens, temperature=cfg.judge_temperature, extra_body=cfg.judge_extra_body)
            judgment = {**reply, "parsed": bcp.parse_judge(reply.get("content"))}
        atomic_write_json(path, scrub_obj(judgment))

    log_event("browsecomp_judge", questions=len(picked), judge=cfg.judge_model, run_id=ctx.run_id)
    with ThreadPoolExecutor(max_workers=cfg.judge_concurrency) as pool:
        for _ in pool.map(grade, picked):
            pass
    judgments = {qid: json.loads((answers / f"{qid}.judge.json").read_text(encoding="utf-8")) for qid in picked}
    table = [bcp.per_query(qid, records[qid], judgments[qid], evidence, golds) for qid in picked]
    atomic_write_text(suite_dir / "per_query.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in table))
    scores = bcp.score(table, judge=cfg.judge_model, limit=cfg.limit)
    scores.update(context_per_stream=_context(ctx, "browsecomp"), concurrency=cfg.workers,
                  judge_settings={"max_tokens": cfg.judge_max_tokens, "temperature": cfg.judge_temperature,
                                  "extra_body": cfg.judge_extra_body, "concurrency": cfg.judge_concurrency})
    if scores["judge_parse_errors"]:
        warn(ctx, f"The judge's reply could not be read for {scores['judge_parse_errors']} BrowseComp-Plus answers; they count as wrong, as in BrowseComp-Plus.")
    lines = [json.dumps({"query_id": qid, "answer_run": records[qid], "judgment": judgments[qid]}, sort_keys=True) for qid in picked]
    atomic_write_bytes(suite_dir / "trajectories.jsonl.gz", gzip.compress("".join(line + "\n" for line in lines).encode("utf-8"), mtime=0))
    ctx.manifest["large_files"] = large_file_hashes(ctx.run_dir)
    atomic_write_json(suite_dir / "scores.json", scores)
    # The questions and answers must not appear as plain text online; only the git-ignored bundle keeps them.
    shutil.rmtree(answers)
