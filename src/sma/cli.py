"""Command line for SMA."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from sma import __version__
from sma.config import Config, load_config, load_env_file
from sma.doctor import default_checks, failed, format_checks
from sma.errors import SmaError
from sma.identity import assert_alias_safe
from sma.plan import render_plan
from sma.queue import JobQueue, run_worker
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
        "--assume-moe",
        action="store_true",
        help="Dry-run only: print the MoE server command",
    )
    run.set_defaults(func=_cmd_run)

    pair = sub.add_parser("run-pair", help="Run a reference quant, then NVFP4, on this Spark")
    pair.add_argument("--bf16", default=None, metavar="REPO")
    pair.add_argument("--mxfp4", default=None, metavar="REPO")
    pair.add_argument("--nvfp4", default=None, metavar="REPO")
    pair.add_argument("--mixed", default=None, metavar="REPO")
    _add_engine(pair)
    pair.add_argument("--bf16-revision", default=None)
    pair.add_argument("--mxfp4-revision", default=None)
    pair.add_argument("--nvfp4-revision", default=None)
    pair.add_argument("--mixed-revision", default=None)
    _add_common(pair)
    pair.add_argument("--dry-run", action="store_true")
    pair.set_defaults(func=_cmd_run_pair)

    resume = sub.add_parser("resume", help="Continue a run that stopped")
    resume.add_argument("run_id")
    _add_common(resume)
    resume.set_defaults(func=_cmd_resume)

    status = sub.add_parser("status", help="Show a run, or the latest activity")
    status.add_argument("run_id", nargs="?")
    _add_common(status)
    status.set_defaults(func=_cmd_status)

    watching = sub.add_parser(
        "watch", help="Live view: the queue, the current run's steps and benchmarks, and finished runs"
    )
    watching.add_argument("--interval", type=float, default=5.0, help="Seconds between redraws (default 5)")
    watching.add_argument("--once", action="store_true", help="Print one screen and exit")
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

    queue = sub.add_parser("queue", help="Queue whole jobs for Spark workers")
    qsub = queue.add_subparsers(dest="qcmd", required=True)
    qadd = qsub.add_parser("add", help="Queue one checkpoint")
    _add_repo(qadd)
    _add_quant_flags(qadd, required=True)
    qadd.add_argument("--revision", default=None)
    _add_engine(qadd)
    _add_common(qadd)
    qadd.set_defaults(func=_cmd_queue_add)

    qpair = qsub.add_parser("add-pair", help="Queue both quants as one job so they stay on one Spark")
    qpair.add_argument("--bf16", default=None, metavar="REPO")
    qpair.add_argument("--mxfp4", default=None, metavar="REPO")
    qpair.add_argument("--nvfp4", default=None, metavar="REPO")
    qpair.add_argument("--mixed", default=None, metavar="REPO")
    _add_engine(qpair)
    qpair.add_argument("--bf16-revision", default=None)
    qpair.add_argument("--mxfp4-revision", default=None)
    qpair.add_argument("--nvfp4-revision", default=None)
    qpair.add_argument("--mixed-revision", default=None)
    _add_common(qpair)
    qpair.set_defaults(func=_cmd_queue_add_pair)

    qlist = qsub.add_parser("list", help="List queued jobs")
    _add_common(qlist)
    qlist.set_defaults(func=_cmd_queue_list)

    qreclaim = qsub.add_parser("reclaim", help="Return stale claims to pending")
    _add_common(qreclaim)
    qreclaim.set_defaults(func=_cmd_queue_reclaim)

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

    worker = sub.add_parser("worker", help="Pull whole jobs and run them one at a time")
    _add_common(worker)
    worker.add_argument("--once", action="store_true", help="Run at most one job, then exit")
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


def _chosen_engine(args: argparse.Namespace, cfg: Config) -> str:
    from sma.engines import engine_name

    quant = _flag(args) if any(getattr(args, name, None) for name in ("bf16", "nvfp4", "mxfp4", "mixed", "gguf")) else None
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
    engine = _chosen_engine(args, cfg)
    _guard_sglang_prompts(cfg, engine)
    if args.dry_run:
        if args.assume_moe and quant == "bf16":
            print(
                "note: --assume-moe does not add an FP4 MoE backend to a BF16 command.",
                file=sys.stderr,
            )
        from sma.engines import selected_by

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
    )
    return 0


def _cmd_run_pair(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    repo, quant, revision = _reference_leg(args)
    other_repo, other_quant, other_revision = _second_leg(args)
    engine = _chosen_engine(args, cfg)
    _guard_sglang_prompts(cfg, engine)
    if args.dry_run:
        print(
            render_plan(
                cfg,
                repo=repo,
                quant_flag=quant,
                revision=revision,
                assume_moe=False,
                engine=engine,
                tool_parser=args.tool_parser,
            ),
            end="",
        )
        print(
            render_plan(
                cfg,
                repo=other_repo,
                quant_flag=other_quant,
                revision=other_revision,
                assume_moe=False,
                engine=engine,
                tool_parser=args.tool_parser,
            ),
            end="",
        )
        print(
            f"pair: both commands run on this Spark under one GPU lock, {quant} first, then {other_quant}, engine {engine}.",
            end="\n",
        )
        return 0
    from sma.locks import GpuLock
    from sma.pipeline import RunSpec, execute, production_services

    services = production_services(cfg)
    with GpuLock(cfg.lock_file()):
        ref_id = execute(
            RunSpec(
                repo=repo,
                quant_flag=quant,
                revision=revision,
                lock_held=True,
                engine=engine,
                tool_parser=args.tool_parser,
                # Both legs share one engine, so a pair skips the per-leg preflight choice.
                selected_by="flag" if args.engine else "default",
            ),
            cfg,
            services,
        )
        print(f"run_id={ref_id} quant={quant} engine={engine}")
        other_id = execute(
            RunSpec(
                repo=other_repo,
                quant_flag=other_quant,
                revision=other_revision,
                reference_run_id=ref_id,
                lock_held=True,
                engine=engine,
                tool_parser=args.tool_parser,
                selected_by="flag" if args.engine else "default",
            ),
            cfg,
            services,
        )
    print(f"run_id={other_id} quant={other_quant} engine={engine} reference={ref_id}")
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
        ),
        cfg,
        production_services(cfg),
    )
    print(f"run_id={args.run_id} status=resumed")
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


def _cmd_queue_add(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    engine = _chosen_engine(args, cfg)
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
            }
        ],
        kind="single",
        tool_parser=args.tool_parser,
    )
    print(job["job_id"])
    return 0


def _cmd_queue_add_pair(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    repo, quant, revision = _reference_leg(args)
    other_repo, other_quant, other_revision = _second_leg(args)
    engine = _chosen_engine(args, cfg)
    _guard_sglang_prompts(cfg, engine)
    job = JobQueue(cfg.queue_path()).add(
        [
            {"repo": repo, "quant": quant, "revision": revision, "engine": engine, "engine_requested": args.engine},
            {"repo": other_repo, "quant": other_quant, "revision": other_revision, "engine": engine, "engine_requested": args.engine},
        ],
        kind="pair",
        tool_parser=args.tool_parser,
    )
    print(job["job_id"])
    return 0


def _cmd_queue_list(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    jobs = JobQueue(cfg.queue_path()).list_jobs()
    if not jobs:
        print("queue empty")
        return 0
    for job in jobs:
        legs = ", ".join(f"{leg['quant']}:{leg['repo']}" for leg in job["legs"])
        print(f"{job['state']:8}  {job['job_id']}  {job['kind']:6}  {legs}")
    return 0


def _cmd_queue_reclaim(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    moved = JobQueue(cfg.queue_path()).reclaim_stale(
        cfg.queue.stale_claim_s, grace_s=cfg.queue.claim_grace_s
    )
    if not moved:
        print("nothing to reclaim")
        return 0
    for job_id in moved:
        print(job_id)
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
