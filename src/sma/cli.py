"""Command line for SMA."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from sma import __version__
from sma.config import Config, load_config, load_env_file
from sma.doctor import default_checks, failed, format_checks
from sma.errors import SmaError
from sma.identity import assert_alias_safe
from sma.plan import render_plan
from sma.queue import JobQueue, run_worker
from sma.watch import engine_tag
from sma.runtime import DockerRuntime


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        load_env_file()
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except SmaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sma",
        description="Serve one pinned Hugging Face checkpoint and record the run. vLLM is the default; SGLang is opt-in.",
    )
    parser.add_argument("--version", action="version", version=f"sma {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run one checkpoint")
    _add_repo(run)
    _add_quant_flags(run, required=True)
    _add_common(run)
    run.add_argument("--revision", default=None)
    run.add_argument("--reference", default=None, help="Run id to compare KL against after logprobs")
    _add_engine(run)
    run.add_argument("--dry-run", action="store_true", help="Print the plan and commands, then exit")
    run.add_argument(
        "--thinking",
        choices=["off", "on"],
        default="off",
        help="The model's thinking mode for every benchmark. off (default): thinking off, greedy quality. "
        "on: thinking on, the model's own sampling. Each mode is its own run, in <quant>-thinking for on.",
    )
    run.add_argument(
        "--assume-moe",
        action="store_true",
        help="Dry-run only: print the MoE server command",
    )
    run.set_defaults(func=_cmd_run)

    pair = sub.add_parser(
        "run-pair",
        help="Run a reference, then a 4-bit quant with thinking off and with thinking on, on this Spark",
    )
    _add_pair_args(pair)
    _add_common(pair)
    pair.add_argument("--dry-run", action="store_true")
    pair.set_defaults(func=_cmd_run_pair)

    resume = sub.add_parser("resume", help="Continue a run that stopped")
    resume.add_argument("run_id")
    resume.add_argument(
        "--redo", action="append", default=[], metavar="SUITE",
        help="Run this finished quality suite again (repeatable). Its earlier attempts stay in its folder.",
    )  # fmt: skip
    _add_common(resume)
    resume.set_defaults(func=_cmd_resume)

    rescore = sub.add_parser(
        "rescore",
        help="Score a finished run's quality suites again from their saved output, with this SMA. No model runs.",
    )
    rescore.add_argument("run_id")
    _add_common(rescore)
    rescore.set_defaults(func=_cmd_rescore)

    status = sub.add_parser("status", help="Show a run, or the latest activity")
    status.add_argument("run_id", nargs="?")
    _add_common(status)
    status.set_defaults(func=_cmd_status)

    watching = sub.add_parser(
        "watch",
        help="Live view: health, each slot or run with its steps, the queue, KernelBench, serving, finished runs",
        description="On a terminal: an interactive screen. Tab moves between panels, the arrows select, Enter opens "
        "a run or queue job, Esc goes back, / filters, r reads again, ? shows the keys, q quits. Piped, or with "
        "--once: the same main screen as plain text. It only reads, and the screen is scrubbed like a run record.",
    )
    watching.add_argument("--interval", type=float, default=5.0, help="Seconds between reads (default 5)")
    watching.add_argument("--once", action="store_true", help="Print the main screen once as plain text and exit")
    watching.add_argument(
        "--plan",
        type=Path,
        default=None,
        metavar="FILE",
        help="Runs a script will start, one 'REPO QUANT' per line, in order; 'since TIME' skips older runs",
    )
    _add_common(watching)
    watching.set_defaults(func=_cmd_watch)

    listing = sub.add_parser("list", help="List runs from the catalog")
    _add_common(listing)
    listing.add_argument("--rebuild", action="store_true", help="Rebuild the catalog by scanning run folders")
    listing.set_defaults(func=_cmd_list)

    compare = sub.add_parser("compare", help="KL, top-1, and delta perplexity between two runs")
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument(
        "--allow-cross-engine",
        action="store_true",
        help="Compare different engines and label the result. Off by default.",
    )
    _add_common(compare)
    compare.set_defaults(func=_cmd_compare)

    doctor = sub.add_parser("doctor", help="Check Docker, the GPU, the image, disk, and the HF token")
    _add_common(doctor)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=_cmd_doctor)

    cleanup = sub.add_parser("cleanup", help="Remove leftover sma-* containers if this Spark is idle")
    _add_common(cleanup)
    cleanup.set_defaults(func=_cmd_cleanup)

    publish = sub.add_parser(
        "publish-check",
        help="Refuse files that would publish a hostname, a lab name, a key, or a name from SMA_PUBLISH_DENY",
    )
    publish.add_argument("paths", nargs="*", type=Path, help="Files or folders. Default: what git tracks or would add")
    publish.add_argument(
        "--fix",
        action="store_true",
        help="First scrub the run files again, as SMA scrubs what it writes (hostnames, tokens, lab names)",
    )
    publish.set_defaults(func=_cmd_publish_check)

    queue = sub.add_parser("queue", help="Queue whole jobs for Spark workers")
    qsub = queue.add_subparsers(dest="qcmd", required=True)
    qadd = qsub.add_parser("add", help="Queue one checkpoint")
    _add_repo(qadd)
    _add_quant_flags(qadd, required=True)
    qadd.add_argument("--revision", default=None)
    qadd.add_argument("--thinking", choices=["off", "on"], default="off", help="The run's thinking mode (see sma run)")
    _add_engine(qadd)
    _add_common(qadd)
    qadd.set_defaults(func=_cmd_queue_add)

    qpair = qsub.add_parser(
        "add-pair",
        help="Queue a reference and a 4-bit quant (thinking off, then on) as one job on one Spark",
    )
    _add_pair_args(qpair)
    _add_common(qpair)
    qpair.set_defaults(func=_cmd_queue_add_pair)

    qlist = qsub.add_parser("list", help="List queued jobs")
    _add_common(qlist)
    qlist.set_defaults(func=_cmd_queue_list)

    qreclaim = qsub.add_parser("reclaim", help="Return stale claims to pending")
    qreclaim.add_argument(
        "--older-than",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Reclaim claims whose heartbeat is older than this (default queue.stale_claim_s). Use it when "
        "you know the worker is gone, for example after stopping it.",
    )
    _add_common(qreclaim)
    qreclaim.set_defaults(func=_cmd_queue_reclaim)

    qremove = qsub.add_parser("remove", help="Take a pending job that has not started off the queue")
    qremove.add_argument("job_id")
    _add_common(qremove)
    qremove.set_defaults(func=_cmd_queue_remove)

    qunpin = qsub.add_parser(
        "unpin",
        help="Drop a host pin and start the job fresh. Does not resume the old run.",
    )
    qunpin.add_argument("job_id")
    _add_common(qunpin)
    qunpin.set_defaults(func=_cmd_queue_unpin)

    qreassign = qsub.add_parser(
        "reassign",
        help="Pin a job to another alias and start it fresh there. Does not resume cross-host.",
    )
    qreassign.add_argument("job_id")
    qreassign.add_argument("--host", required=True, help="Host alias that should run the job from scratch")
    _add_common(qreassign)
    qreassign.set_defaults(func=_cmd_queue_reassign)

    qdrop = qsub.add_parser(
        "drop",
        help="Take a job off the queue even if it has started (its worker must be gone). Its runs stay resumable by hand.",
    )
    qdrop.add_argument("job_id")
    _add_common(qdrop)
    qdrop.set_defaults(func=_cmd_queue_drop)

    worker = sub.add_parser("worker", help="Pull whole jobs and run them one at a time, or one per GPU slot with --slots")
    _add_common(worker)
    worker.add_argument("--once", action="store_true", help="Run at most one job, then exit")
    worker.add_argument(
        "--slots",
        action="store_true",
        help="Run one leg per GPU slot of the serving machine (remote.slots), many at once: each queued leg takes the "
        "first free slot, in queue order. One per serving machine.",
    )
    worker.add_argument("--run-unit", nargs=2, metavar=("JOB_ID", "LEG"), default=None, help=argparse.SUPPRESS)
    worker.add_argument("--slot", default=None, help=argparse.SUPPRESS)
    worker.set_defaults(func=_cmd_worker)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to sma.yaml")
    parser.add_argument("--runs-dir", default=None, help="Override the runs directory")


def _add_engine(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=["vllm", "sglang", "llamacpp"],
        default=None,
        help="Inference engine. vLLM is the default for safetensors, sglang is opt-in, and GGUF (--gguf) "
        "always runs on llamacpp.",
    )
    parser.add_argument(
        "--tool-parser",
        default=None,
        metavar="NAME",
        help="The engine's tool-call parser for this model, for example hermes for Qwen3 on vLLM. "
        "Needed by suites that call tools (BFCL). Default: engine.tool_call_parser.",
    )


def _add_pair_args(parser: argparse.ArgumentParser) -> None:
    """A pair: a reference checkpoint and a 4-bit one. The 4-bit one runs twice, thinking off and on."""
    parser.add_argument("--bf16", default=None, metavar="REPO")
    parser.add_argument("--mxfp4", default=None, metavar="REPO")
    parser.add_argument("--nvfp4", default=None, metavar="REPO")
    parser.add_argument("--mixed", default=None, metavar="REPO")
    parser.add_argument(
        "--gguf",
        nargs=3,
        default=None,
        metavar=("REPO", "REFERENCE", "QUANT"),
        help="A GGUF repo's full-precision file and a quant file, for example "
        "LiquidAI/LFM3-1.7B-GGUF BF16 QAD-Q4_0. Runs on llama.cpp.",
    )
    _add_engine(parser)
    parser.add_argument("--bf16-revision", default=None)
    parser.add_argument("--mxfp4-revision", default=None)
    parser.add_argument("--nvfp4-revision", default=None)
    parser.add_argument("--mixed-revision", default=None)


def _pair(args: argparse.Namespace, cfg: Config) -> tuple[list[dict[str, Any]], str]:
    """The pair's legs (queue.pair_legs) and its engine. GGUF pairs come from one repo."""
    from sma.queue import pair_legs

    if args.gguf:
        if any((args.bf16, args.mxfp4, args.nvfp4, args.mixed)):
            raise SmaError("A GGUF pair takes --gguf REPO REFERENCE QUANT alone, not --bf16, --mxfp4, --nvfp4, or --mixed.")
        repo, reference_file, quant_file = args.gguf
        engine = _chosen_engine(args, cfg, "gguf")
        reference = {"repo": repo, "quant": "gguf", "gguf": reference_file}
        quant = {"repo": repo, "quant": "gguf", "gguf": quant_file}
    else:
        repo, ref_quant, revision = _reference_leg(args)
        other_repo, other_quant, other_revision = _second_leg(args)
        engine = _chosen_engine(args, cfg, ref_quant)
        reference = {"repo": repo, "quant": ref_quant, "revision": revision}
        quant = {"repo": other_repo, "quant": other_quant, "revision": other_revision}
    _guard_sglang_prompts(cfg, engine)
    shared = {"engine": engine, "engine_requested": args.engine}
    return pair_legs({**reference, **shared}, {**quant, **shared}), engine


def _leg_name(leg: dict[str, Any]) -> str:
    return f"{leg['repo']} {leg.get('gguf') or leg['quant']}" + (" thinking on" if leg.get("thinking") == "on" else "")


def _add_repo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repo", help="Hugging Face repo, for example org/model")


def _add_quant_flags(parser: argparse.ArgumentParser, *, required: bool) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--bf16", action="store_true", help="Checkpoint is BF16. SMA will not cast it.")
    group.add_argument("--nvfp4", action="store_true", help="Checkpoint is already NVFP4.")
    group.add_argument("--mxfp4", action="store_true", help="Checkpoint is already native MXFP4.")
    group.add_argument(
        "--mixed",
        action="store_true",
        help="Checkpoint already mixes FP8 and NVFP4 (for example dgpp_mixed).",
    )
    group.add_argument(
        "--gguf",
        default=None,
        metavar="QUANT",
        help="A GGUF repo's quant file, for example UD-Q4_K_XL, Q8_0, or BF16. Runs on llama.cpp.",
    )


def _flag(args: argparse.Namespace) -> str:
    if getattr(args, "bf16", False) is True:
        return "bf16"
    if getattr(args, "nvfp4", False) is True:
        return "nvfp4"
    if getattr(args, "mxfp4", False) is True:
        return "mxfp4"
    if getattr(args, "mixed", False) is True:
        return "mixed"
    if getattr(args, "gguf", None):
        return "gguf"
    raise SmaError("Pass one of --bf16, --nvfp4, --mxfp4, --mixed, or --gguf QUANT")


def _cfg(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.runs_dir:
        cfg.runs_dir = args.runs_dir
    assert_alias_safe(cfg.host_alias)
    return cfg


def _chosen_engine(args: argparse.Namespace, cfg: Config, quant: str | None) -> str:
    # The caller names the quant. In run-pair and queue add-pair, --bf16 and the others hold repo names,
    # so reading them as true-or-false flags failed every pair command (F1 in model-list.txt).
    from sma.engines import engine_name

    return engine_name(getattr(args, "engine", None), cfg, quant)


def _dry_run_plan(
    cfg: Config, repo: str, revision: str | None, quant: str, engine: str, gguf: str | None
) -> tuple[dict | None, str | None, str | None]:
    """(memory plan, note, GGUF file) from the repo's config and file sizes, or its GGUF header. No weights."""
    from sma.pipeline import HubFetcher

    try:
        plan, model_file = HubFetcher(cfg, registry=None, engine=engine).plan_only(repo, revision, quant, gguf)
    except Exception as exc:  # noqa: BLE001 - a dry run says why it has no numbers and goes on
        return None, f"Not looked up here: {str(exc).splitlines()[0][:300]}", None
    return plan, None, model_file


def _guard_sglang_prompts(cfg: Config, engine: str) -> None:
    """Fail before a pull, a download, or a quality server when SGLang cannot bench."""
    if engine != "sglang":
        return
    from sma.backends import assert_sglang_dataset_covers

    assert_sglang_dataset_covers(cfg.speed.num_prompts)


def _second_leg(args: argparse.Namespace) -> tuple[str, str, str | None]:
    if args.nvfp4 and args.mixed:
        raise SmaError("Pass either --nvfp4 or --mixed as the second checkpoint, not both")
    if args.nvfp4:
        return args.nvfp4, "nvfp4", args.nvfp4_revision
    if args.mixed:
        return args.mixed, "mixed", getattr(args, "mixed_revision", None)
    raise SmaError("Pass --nvfp4 REPO or --mixed REPO as the second checkpoint")


def _reference_leg(args: argparse.Namespace) -> tuple[str, str, str | None]:
    if args.bf16 and args.mxfp4:
        raise SmaError("Pass either --bf16 or --mxfp4 as the reference, not both")
    if args.bf16:
        return args.bf16, "bf16", args.bf16_revision
    if args.mxfp4:
        return args.mxfp4, "mxfp4", args.mxfp4_revision
    raise SmaError("Pass --bf16 REPO or --mxfp4 REPO as the reference checkpoint")


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    quant = _flag(args)
    engine = _chosen_engine(args, cfg, quant)
    _guard_sglang_prompts(cfg, engine)
    if args.dry_run:
        if args.assume_moe and quant == "bf16":
            print(
                "note: --assume-moe does not add an FP4 MoE backend to a BF16 command.",
                file=sys.stderr,
            )
        from sma.config import for_thinking
        from sma.engines import selected_by

        cfg = for_thinking(cfg, args.thinking)
        memory_plan, memory_note, model_file = _dry_run_plan(cfg, args.repo, args.revision, quant, engine, args.gguf)
        print(
            render_plan(
                cfg,
                repo=args.repo,
                quant_flag=quant,
                revision=args.revision,
                assume_moe=bool(args.assume_moe and quant != "bf16"),
                engine=engine,
                tool_parser=args.tool_parser,
                gguf=args.gguf,
                selected=selected_by(args.engine, quant),
                memory_plan=memory_plan,
                memory_note=memory_note,
                model_file=model_file,
            ),
            end="",
        )
        return 0
    if args.assume_moe:
        raise SmaError("--assume-moe is only valid with --dry-run")
    _execute_one(
        cfg,
        args.repo,
        quant,
        args.revision,
        args.reference,
        None,
        None,
        engine=args.engine,
        tool_parser=args.tool_parser,
        gguf=args.gguf,
        thinking=args.thinking,
    )
    return 0


def _cmd_run_pair(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    legs, engine = _pair(args, cfg)
    if args.dry_run:
        from sma.config import for_thinking

        for leg in legs:
            print(
                render_plan(
                    for_thinking(cfg, leg["thinking"]),
                    repo=leg["repo"],
                    quant_flag=leg["quant"],
                    revision=leg.get("revision"),
                    assume_moe=False,
                    engine=engine,
                    tool_parser=args.tool_parser,
                    gguf=leg.get("gguf"),
                ),
                end="",
            )
        order = ", then ".join(_leg_name(leg) for leg in legs)
        print(f"pair: on this Spark under one GPU lock, engine {engine}: {order}. Each quant leg compares with the first.")
        return 0
    from sma.errors import RunIncomplete
    from sma.locks import GpuLock
    from sma.pipeline import RunSpec, execute, production_services

    services = production_services(cfg)
    reference_id: str | None = None
    incomplete: list[str] = []
    with GpuLock(cfg.lock_file()):
        for index, leg in enumerate(legs):
            spec = RunSpec(
                repo=leg["repo"],
                quant_flag=leg["quant"],
                revision=leg.get("revision"),
                reference_run_id=reference_id,
                lock_held=True,
                engine=engine,
                tool_parser=args.tool_parser,
                # Every leg shares one engine, so a pair skips the per-leg preflight choice.
                selected_by="flag" if args.engine else "default",
                gguf=leg.get("gguf"),
                thinking=leg["thinking"],
            )
            try:
                run_id = execute(spec, cfg, services)
            except RunIncomplete as exc:
                # The record has the failed steps; the next legs still run.
                run_id = exc.run_id
                incomplete.append(str(exc))
            if index == 0:
                reference_id = run_id
            print(f"run_id={run_id} {_leg_name(leg)} engine={engine}" + (f" reference={reference_id}" if index else ""))
    if incomplete:
        raise SmaError(" ".join(incomplete))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.catalog import Catalog
    from sma.pipeline import RunSpec, execute, production_services
    from sma.records import load_manifest

    found = Catalog(cfg.runs_path()).find(args.run_id)
    if found is None:
        raise SmaError(f"No run with id {args.run_id}")
    manifest = load_manifest(found)
    execute(
        RunSpec(
            repo=str(manifest["model_repo"]),
            quant_flag=str(manifest["quant_flag"]),
            revision=manifest.get("revision_requested"),
            reference_run_id=manifest.get("reference_run_id"),
            pair_job_id=manifest.get("pair_job_id"),
            resume_run_id=args.run_id,
            engine=(manifest.get("engine") or {}).get("name") or "vllm",
            gguf=(manifest.get("gguf") or {}).get("selector"),
            thinking=manifest.get("thinking") or "off",
            redo=tuple(args.redo),
        ),
        cfg,
        production_services(cfg),
    )
    print(f"run_id={args.run_id} status=resumed")
    return 0


def _cmd_rescore(args: argparse.Namespace) -> int:
    from sma.pipeline import rescore

    suites = rescore(_cfg(args), args.run_id)
    print(f"run_id={args.run_id} rescored={','.join(suites) or 'none'}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.catalog import Catalog
    from sma.records import load_manifest

    catalog = Catalog(cfg.runs_path())
    if args.run_id:
        found = catalog.find(args.run_id)
        if found is None:
            raise SmaError(f"No run with id {args.run_id}")
        manifest = load_manifest(found)
        engine_block = manifest.get("engine") or {}
        engine_label = engine_block.get("name") if isinstance(engine_block, dict) else engine_block
        print(f"run_id: {manifest['run_id']}")
        print(f"model: {manifest['model_repo']}")
        print(f"quant: {manifest.get('quant_label') or manifest['quant_flag']}")
        print(f"engine: {engine_label or 'vllm'}")
        print(f"host_alias: {manifest['host_alias']}")
        print(f"path: {found}")
        for name, step in manifest["steps"].items():
            error = f"  {step['error']}" if step.get("error") else ""
            print(f"  {name:16} {step['status']}{error}")
        return 0
    rows = catalog.list_runs()
    if not rows:
        print("no runs")
    else:
        for row in rows[:20]:
            print(f"{row['run_id']}  {row['status']:8}  {row['quant']:6}  {row['model_repo']}")
    jobs = JobQueue(cfg.queue_path()).list_jobs()
    pending = sum(1 for job in jobs if job.get("state") == "pending")
    claimed = sum(1 for job in jobs if job.get("state") == "claimed")
    print(f"queue: {pending} pending, {claimed} claimed")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.httpapi import scrape_metrics
    from sma.machines import gather
    from sma.telemetry import meminfo
    from sma.watch import watch

    if args.interval <= 0:
        raise SmaError("--interval must be more than 0 seconds")
    runtime = DockerRuntime(cfg)

    def containers() -> list[str]:
        try:
            return [name for name in runtime.list_sma() if runtime.is_running(name)]
        except SmaError:
            return []

    def started_at(name: str):
        try:
            return runtime.started_at(name)
        except SmaError:
            return None

    def swe_progress(run_id: str):
        """A run's finished SWE-bench tasks on the runner, by exit status. None when it cannot be read."""
        from collections import Counter

        from sma.swebench import dataset_size, runner_from_env, task_statuses

        runner = runner_from_env(cfg)
        if runner is None:
            return None
        try:
            statuses = task_statuses(runner, run_id)
            return {
                "done": len(statuses),
                "total": dataset_size(cfg, runner),
                "statuses": dict(Counter(str(item.get("exit_status")) for item in statuses)),
            }
        except SmaError:
            return None

    def scrape(base_url: str, path: str) -> str:
        try:
            return scrape_metrics(base_url, cfg.timeouts.metrics_s, path=path)
        except SmaError:
            return ""

    def kernelbench_jobs():
        """The KernelBench service's queue, or None when KERNELBENCH_URL is not set or the service does not answer."""
        import os

        from sma.kbservice import URL_ENV, KernelBenchService

        url = os.environ.get(URL_ENV)
        if not url or not cfg.kernelbench.enabled:
            return None
        try:
            return KernelBenchService(url, timeout_s=5.0, attempts=1).jobs()
        except SmaError:
            return None

    watch(
        cfg,
        plan=args.plan,
        interval_s=args.interval,
        once=args.once,
        containers=containers,
        memory=meminfo,
        scrape=scrape,
        machines=lambda: gather(cfg),
        started_at=started_at,
        swe_progress=swe_progress,
        kernelbench=kernelbench_jobs,
        # Plain `sma watch` shows every run and every serving machine's slots; --config shows only that config's.
        scoped=args.config is not None,
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.catalog import Catalog

    catalog = Catalog(cfg.runs_path())
    if args.rebuild:
        count = catalog.rebuild()
        print(f"rebuilt {count} runs")
    rows = catalog.list_runs()
    if not rows:
        print("no runs")
        return 0
    for row in rows:
        sha = (row.get("revision_sha") or "")[:12]
        engine = row.get("engine") or "vllm"
        print(f"{row['run_id']}  {row['status']:8}  {engine:6}  {row['quant']:6}  {sha:12}  {row['model_repo']}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.catalog import Catalog
    from sma.compare import compare_run_dirs, write_compare

    catalog = Catalog(cfg.runs_path())
    reference = _resolve(catalog, args.reference)
    candidate = _resolve(catalog, args.candidate)
    result = compare_run_dirs(reference, candidate, allow_cross_engine=args.allow_cross_engine)
    path = write_compare(cfg.runs_path(), result)
    k = result.get("k")
    print(f"k: {k}")
    print(f"method: {result.get('method')}")
    print(f"kl_mean_nats: {result.get('kl_mean')}")
    print(f"top1_agreement: {result.get('top1_agreement')}")
    print(f"delta_ppl: {result.get('delta_ppl')}")
    print(f"approximate_positions: {result.get('kl_approximate_positions')}")
    for note in result.get("notes") or []:
        print(f"note: {note}")
    print(f"wrote: {path}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    checks = default_checks(cfg)
    if args.json:
        print(json.dumps([check.__dict__ for check in checks], indent=2))
    else:
        print(format_checks(checks), end="")
    return 2 if failed(checks) else 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.locks import GpuLock

    runtime = DockerRuntime(cfg)
    with GpuLock(cfg.lock_file()):
        names = runtime.list_sma()
        if not names:
            print("no SMA containers")
            return 0
        for name in names:
            runtime.stop(name)
            print(f"removed {name}")
    return 0


def _cmd_publish_check(args: argparse.Namespace) -> int:
    from sma.atomic import atomic_write_text
    from sma.proc import require_ok, run_command
    from sma.scrub import find_leaks, scrub_text

    # In the repository, only what git tracks or would add: the files too big for git (a 2 GB kld-base.bin) are
    # never published, and their sha256 is in the manifest, so they are neither read nor rewritten. A folder
    # outside it (the pre-push hook's copy of what a push sends) is checked whole.
    root = Path.cwd().resolve()
    inside = [path for path in args.paths if path.resolve().is_relative_to(root)]
    outside = [path for path in args.paths if path not in inside]
    files = [f for path in outside for f in ([path] if path.is_file() else sorted(path.rglob("*"))) if f.is_file()]
    if inside or not args.paths:
        argv = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *map(str, inside)]
        listed = require_ok(run_command(argv, timeout_s=120, what="list the files git publishes"), what="git ls-files")
        files += [Path(name) for name in listed.stdout.split("\0") if name and Path(name).is_file()]
    leaking = 0
    for path in files:
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        # Only run files are scrubbed in place, and only text: a document that names the lab needs rewording,
        # not a placeholder, and a binary file would be broken by a rewrite.
        if args.fix and "runs" in path.parts and text.encode("utf-8") == data:
            clean = scrub_text(text)
            if clean != text:
                atomic_write_text(path, clean)
                print(f"scrubbed {path}")
                text = clean
        found = find_leaks(text)
        if found:
            leaking += 1
            print(f"{path}: {', '.join(found)}")
    if leaking:
        print(f"{leaking} of {len(files)} files would publish what is named above. Fix them before you push.")
        return 1
    print(f"Nothing to hide in {len(files)} files.")
    return 0


def _cmd_queue_add(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    engine = _chosen_engine(args, cfg, _flag(args))
    _guard_sglang_prompts(cfg, engine)
    job = JobQueue(cfg.queue_path()).add(
        [
            {
                "repo": args.repo,
                "quant": _flag(args),
                "revision": args.revision,
                "engine": engine,
                "engine_requested": args.engine,
                "gguf": args.gguf,
                "thinking": args.thinking,
            }
        ],
        kind="single",
        tool_parser=args.tool_parser,
    )
    print(job["job_id"])
    return 0


def _cmd_queue_add_pair(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    legs, _engine = _pair(args, cfg)
    job = JobQueue(cfg.queue_path()).add(legs, kind="pair", tool_parser=args.tool_parser)
    print(job["job_id"])
    return 0


def _cmd_queue_list(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    jobs = JobQueue(cfg.queue_path()).list_jobs()
    if not jobs:
        print("queue empty")
        return 0
    for job in jobs:
        legs = ", ".join(_leg_name(leg) for leg in job["legs"])
        print(f"{engine_tag(job.get('engine')):4}  {job['state']:8}  {job['job_id']}  {job['kind']:6}  {legs}")
    return 0


def _cmd_queue_reclaim(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    stale_s = cfg.queue.stale_claim_s if args.older_than is None else args.older_than
    moved = JobQueue(cfg.queue_path()).reclaim_stale(stale_s, grace_s=min(cfg.queue.claim_grace_s, stale_s))
    if not moved:
        print("nothing to reclaim")
        return 0
    for job_id in moved:
        print(job_id)
    return 0


def _cmd_queue_remove(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    job = JobQueue(cfg.queue_path()).remove(args.job_id)
    print(f"{job['job_id']} removed: {', '.join(_leg_name(leg) for leg in job['legs'])}")
    return 0


def _cmd_queue_drop(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    job = JobQueue(cfg.queue_path()).drop(args.job_id, stale_s=cfg.queue.stale_claim_s)
    runs = [leg["run_id"] for leg in job["legs"] if leg.get("run_id")]
    print(f"{job['job_id']} dropped ({job['removed_from']}): {', '.join(_leg_name(leg) for leg in job['legs'])}")
    for run_id in runs:
        print(f"  run {run_id} stays in runs/; resume it with sma resume {run_id}")
    return 0


def _cmd_queue_unpin(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    job = JobQueue(cfg.queue_path()).unpin(args.job_id)
    print(f"{job['job_id']} unpinned; the next host starts it fresh")
    return 0


def _cmd_queue_reassign(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    job = JobQueue(cfg.queue_path()).reassign(args.job_id, args.host)
    print(f"{job['job_id']} reassigned to {args.host}; the run starts fresh and is not resumed")
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.pipeline import production_services

    if args.run_unit:
        # A dispatcher's child: one leg on one slot. Not for use by hand.
        from sma.dispatcher import run_unit

        if not args.slot:
            raise SmaError("--run-unit needs --slot NAME.")
        job_id, leg = args.run_unit
        if not leg.isdigit():
            raise SmaError("--run-unit takes a job id and a leg number.")
        return run_unit(cfg, production_services, job_id, int(leg), args.slot)
    if args.slots:
        from sma.dispatcher import run_dispatcher

        if args.once:
            raise SmaError("--once runs one job on a one-run worker; --slots keeps every slot busy until stopped.")
        print(f"dispatcher alias {cfg.host_alias}, slots {', '.join(slot.name for slot in cfg.remote.slots)}")
        run_dispatcher(cfg, config_path=args.config, runs_dir=args.runs_dir)
        return 0

    worker_id = "worker-" + uuid.uuid4().hex[:8]
    print(f"worker {worker_id} alias {cfg.host_alias}")
    run_worker(cfg, production_services, once=args.once, worker_id=worker_id)
    return 0


def _execute_one(
    cfg: Config,
    repo: str,
    quant: str,
    revision: str | None,
    reference: str | None,
    pair_job_id: str | None,
    resume: str | None,
    engine: str | None = None,
    tool_parser: str | None = None,
    gguf: str | None = None,
    thinking: str = "off",
) -> str:
    from sma.pipeline import RunSpec, execute, production_services

    run_id = execute(
        RunSpec(
            repo=repo,
            quant_flag=quant,
            revision=revision,
            reference_run_id=reference,
            pair_job_id=pair_job_id,
            resume_run_id=resume,
            engine=engine,
            tool_parser=tool_parser,
            gguf=gguf,
            thinking=thinking,
        ),
        cfg,
        production_services(cfg),
    )
    print(f"run_id={run_id} status=done path={cfg.runs_path()}")
    return run_id


def _resolve(catalog: object, run_id: str) -> Path:
    found = catalog.find(run_id)  # type: ignore[attr-defined]
    if found is None:
        raise SmaError(f"No run with id {run_id}")
    return found
