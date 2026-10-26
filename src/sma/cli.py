"""Command line for SMA."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from sma import __version__
from sma.config import Config, load_config
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
        description="Serve one pinned Hugging Face checkpoint with vLLM and record the run.",
    )
    parser.add_argument("--version", action="version", version=f"sma {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run one checkpoint")
    _add_repo(run)
    _add_quant_flags(run, required=True)
    _add_common(run)
    run.add_argument("--revision", default=None)
    run.add_argument("--reference", default=None, help="Run id to compare KL against after logprobs")
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
    pair.add_argument("--nvfp4", required=True, metavar="REPO")
    pair.add_argument("--bf16-revision", default=None)
    pair.add_argument("--mxfp4-revision", default=None)
    pair.add_argument("--nvfp4-revision", default=None)
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

    listing = sub.add_parser("list", help="List runs from the catalog")
    _add_common(listing)
    listing.add_argument("--rebuild", action="store_true", help="Rebuild the catalog by scanning run folders")
    listing.set_defaults(func=_cmd_list)

    compare = sub.add_parser("compare", help="KL, top-1, and delta perplexity between two runs")
    compare.add_argument("reference")
    compare.add_argument("candidate")
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
    _add_common(qadd)
    qadd.set_defaults(func=_cmd_queue_add)

    qpair = qsub.add_parser("add-pair", help="Queue both quants as one job so they stay on one Spark")
    qpair.add_argument("--bf16", default=None, metavar="REPO")
    qpair.add_argument("--mxfp4", default=None, metavar="REPO")
    qpair.add_argument("--nvfp4", required=True, metavar="REPO")
    qpair.add_argument("--bf16-revision", default=None)
    qpair.add_argument("--mxfp4-revision", default=None)
    qpair.add_argument("--nvfp4-revision", default=None)
    _add_common(qpair)
    qpair.set_defaults(func=_cmd_queue_add_pair)

    qlist = qsub.add_parser("list", help="List queued jobs")
    _add_common(qlist)
    qlist.set_defaults(func=_cmd_queue_list)

    qreclaim = qsub.add_parser("reclaim", help="Return stale claims to pending")
    _add_common(qreclaim)
    qreclaim.set_defaults(func=_cmd_queue_reclaim)

    worker = sub.add_parser("worker", help="Pull whole jobs and run them one at a time")
    _add_common(worker)
    worker.add_argument("--once", action="store_true", help="Run at most one job, then exit")
    worker.set_defaults(func=_cmd_worker)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to sma.yaml")
    parser.add_argument("--runs-dir", default=None, help="Override the runs directory")


def _add_repo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("repo", help="Hugging Face repo, for example org/model")


def _add_quant_flags(parser: argparse.ArgumentParser, *, required: bool) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--bf16", action="store_true", help="Checkpoint is BF16. SMA will not cast it.")
    group.add_argument("--nvfp4", action="store_true", help="Checkpoint is already NVFP4.")
    group.add_argument("--mxfp4", action="store_true", help="Checkpoint is already native MXFP4.")


def _flag(args: argparse.Namespace) -> str:
    if getattr(args, "bf16", False) is True:
        return "bf16"
    if getattr(args, "nvfp4", False) is True:
        return "nvfp4"
    if getattr(args, "mxfp4", False) is True:
        return "mxfp4"
    raise SmaError("Pass one of --bf16, --nvfp4, or --mxfp4")


def _cfg(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.runs_dir:
        cfg.runs_dir = args.runs_dir
    assert_alias_safe(cfg.host_alias)
    return cfg


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
    if args.dry_run:
        if args.assume_moe and quant == "bf16":
            print(
                "note: --assume-moe does not add an FP4 MoE backend to a BF16 command.",
                file=sys.stderr,
            )
        print(
            render_plan(
                cfg,
                repo=args.repo,
                quant_flag=quant,
                revision=args.revision,
                assume_moe=bool(args.assume_moe and quant != "bf16"),
            ),
            end="",
        )
        return 0
    if args.assume_moe:
        raise SmaError("--assume-moe is only valid with --dry-run")
    _execute_one(cfg, args.repo, quant, args.revision, args.reference, None, None)
    return 0


def _cmd_run_pair(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    repo, quant, revision = _reference_leg(args)
    if args.dry_run:
        print(render_plan(cfg, repo=repo, quant_flag=quant, revision=revision, assume_moe=False), end="")
        print(
            render_plan(
                cfg,
                repo=args.nvfp4,
                quant_flag="nvfp4",
                revision=args.nvfp4_revision,
                assume_moe=False,
            ),
            end="",
        )
        print("pair: both commands run on this Spark, reference first, then NVFP4.", end="\n")
        return 0
    from sma.pipeline import RunSpec, execute, production_services

    services = production_services(cfg)
    ref_id = execute(
        RunSpec(repo=repo, quant_flag=quant, revision=revision),
        cfg,
        services,
    )
    print(f"run_id={ref_id} quant={quant}")
    nv_id = execute(
        RunSpec(
            repo=args.nvfp4,
            quant_flag="nvfp4",
            revision=args.nvfp4_revision,
            reference_run_id=ref_id,
        ),
        cfg,
        production_services(cfg),
    )
    print(f"run_id={nv_id} quant=nvfp4 reference={ref_id}")
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
        print(f"run_id: {manifest['run_id']}")
        print(f"model: {manifest['model_repo']}")
        print(f"quant: {manifest['quant_flag']}")
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
        print(f"{row['run_id']}  {row['status']:8}  {row['quant']:6}  {sha:12}  {row['model_repo']}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from sma.catalog import Catalog
    from sma.compare import compare_run_dirs, write_compare

    catalog = Catalog(cfg.runs_path())
    reference = _resolve(catalog, args.reference)
    candidate = _resolve(catalog, args.candidate)
    result = compare_run_dirs(reference, candidate)
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
    job = JobQueue(cfg.queue_path()).add(
        [{"repo": args.repo, "quant": _flag(args), "revision": args.revision}],
        kind="single",
    )
    print(job["job_id"])
    return 0


def _cmd_queue_add_pair(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    repo, quant, revision = _reference_leg(args)
    job = JobQueue(cfg.queue_path()).add(
        [
            {"repo": repo, "quant": quant, "revision": revision},
            {"repo": args.nvfp4, "quant": "nvfp4", "revision": args.nvfp4_revision},
        ],
        kind="pair",
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
    moved = JobQueue(cfg.queue_path()).reclaim_stale(cfg.queue.stale_claim_s)
    if not moved:
        print("nothing to reclaim")
        return 0
    for job_id in moved:
        print(job_id)
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
