"""A filesystem queue. Claims are atomic (link + unlink) on one filesystem."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sma.atomic import atomic_write_json
from sma.config import Config
from sma.errors import SmaError
from sma.identity import assert_alias_safe
from sma.records import utc_now


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class JobQueue:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        for name in ("pending", "claimed", "done", "failed"):
            (self.path / name).mkdir(parents=True, exist_ok=True)

    def add(self, legs: list[dict[str, Any]], *, kind: str) -> dict[str, Any]:
        if not legs:
            raise SmaError("A queue job needs at least one checkpoint")
        job_id = "job-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "kind": kind,
            "created": utc_now(),
            "legs": [_leg(item) for item in legs],
            "claim": None,
            "heartbeat": None,
            "error": None,
        }
        atomic_write_json(self.path / "pending" / f"{job_id}.json", job)
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for state in ("pending", "claimed", "done", "failed"):
            for path in sorted((self.path / state).glob("*.json")):
                job = _read(path)
                job["state"] = state
                jobs.append(job)
        return jobs

    def claim(self, worker_id: str, host_alias: str) -> tuple[Path, dict[str, Any]] | None:
        pending = self.path / "pending"
        claimed = self.path / "claimed"
        for path in sorted(pending.glob("*.json")):
            dest = claimed / path.name
            if dest.exists():
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                os.link(path, dest)
            except FileExistsError:
                continue
            except FileNotFoundError:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            job = _read(dest)
            job["claim"] = {
                "worker_id": worker_id,
                "host_alias": host_alias,
                "when": utc_now(),
            }
            job["heartbeat"] = utc_now()
            job["error"] = None
            atomic_write_json(dest, job)
            return dest, job
        return None

    def heartbeat(self, path: Path, job: dict[str, Any]) -> None:
        job["heartbeat"] = utc_now()
        atomic_write_json(path, job)

    def finish(self, path: Path, job: dict[str, Any], *, ok: bool) -> Path:
        state = "done" if ok else "failed"
        dest = self.path / state / path.name
        atomic_write_json(dest, job)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return dest

    def reclaim_stale(self, stale_s: float) -> list[str]:
        now = datetime.now(timezone.utc)
        moved: list[str] = []
        for path in sorted((self.path / "claimed").glob("*.json")):
            job = _read(path)
            beat = _parse_time(job.get("heartbeat")) or _parse_time(job.get("claim", {}).get("when") if isinstance(job.get("claim"), dict) else None)
            if beat is None:
                age = stale_s + 1
            else:
                age = (now - beat).total_seconds()
            if age < stale_s:
                continue
            job["claim"] = None
            job["heartbeat"] = None
            job["error"] = "reclaimed after a stale claim"
            dest = self.path / "pending" / path.name
            atomic_write_json(dest, job)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            moved.append(job["job_id"])
        return moved


def _leg(item: dict[str, Any]) -> dict[str, Any]:
    quant = item["quant"]
    if quant not in {"bf16", "nvfp4", "mxfp4"}:
        raise SmaError(f"Unsupported quant {quant}")
    return {
        "repo": item["repo"],
        "quant": quant,
        "revision": item.get("revision"),
        "run_id": None,
        "status": "pending",
        "reference_run_id": item.get("reference_run_id"),
    }


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmaError(f"Queue file {path.name} is not a JSON object")
    return payload


def incomplete_legs(job: dict[str, Any]) -> list[int]:
    indexes = []
    for index, leg in enumerate(job.get("legs") or []):
        if leg.get("status") != "done":
            indexes.append(index)
    return indexes


def run_worker(
    cfg: Config,
    services_factory: Any,
    *,
    once: bool,
    worker_id: str | None = None,
) -> int:
    """Pull whole jobs. One job (one or both quants) runs on this Spark, never split."""
    import time

    from sma.locks import GpuLock
    from sma.pipeline import RunSpec, execute

    assert_alias_safe(cfg.host_alias)
    queue = JobQueue(cfg.queue_path())
    identity = worker_id or ("worker-" + uuid.uuid4().hex[:8])
    completed = 0
    while True:
        claimed = queue.claim(identity, cfg.host_alias)
        if claimed is None:
            if once:
                return completed
            time.sleep(cfg.queue.poll_s)
            continue
        path, job = claimed
        lock = GpuLock(cfg.lock_file())
        lock.acquire()
        try:
            _run_job(cfg, services_factory, queue, path, job, execute, RunSpec)
            completed += 1
        finally:
            lock.release()
        if once:
            return completed


def _run_job(cfg: Config, services_factory: Any, queue: JobQueue, path: Path, job: dict[str, Any], execute: Any, run_spec: Any) -> None:
    reference_id = None
    job_lock = threading.Lock()
    stop = threading.Event()

    def _pulse() -> None:
        interval = max(1.0, float(cfg.queue.heartbeat_s))
        while not stop.wait(interval):
            with job_lock:
                if path.is_file():
                    queue.heartbeat(path, job)

    beater = threading.Thread(target=_pulse, name="sma-queue-heartbeat", daemon=True)
    beater.start()
    try:
        for index in incomplete_legs(job):
            leg = job["legs"][index]
            if index > 0 and job["legs"][index - 1]["status"] != "done":
                raise SmaError(
                    f"Leg {index - 1} of {job['job_id']} is not done, so the next quant was not started. "
                    "Both quants of a pair stay on this Spark."
                )
            if index > 0:
                reference_id = job["legs"][0].get("run_id")
            resume = leg.get("run_id") if leg.get("status") in {"running", "failed"} else None

            def _created(run_id: str, leg_index: int = index) -> None:
                with job_lock:
                    job["legs"][leg_index]["run_id"] = run_id
                    job["legs"][leg_index]["status"] = "running"
                    queue.heartbeat(path, job)

            services = services_factory(cfg)
            run_id = execute(
                run_spec(
                    repo=leg["repo"],
                    quant_flag=leg["quant"],
                    revision=leg.get("revision"),
                    reference_run_id=reference_id or leg.get("reference_run_id"),
                    pair_job_id=job["job_id"] if len(job["legs"]) > 1 else None,
                    resume_run_id=resume,
                    lock_held=True,
                ),
                cfg,
                services,
                on_created=_created,
            )
            with job_lock:
                job["legs"][index]["run_id"] = run_id
                job["legs"][index]["status"] = "done"
                queue.heartbeat(path, job)
        with job_lock:
            job["error"] = None
            queue.finish(path, job, ok=True)
    except Exception as exc:
        from sma.scrub import scrub_text

        with job_lock:
            job["error"] = scrub_text(str(exc))[:2000]
            for leg in job["legs"]:
                if leg.get("status") == "running":
                    leg["status"] = "failed"
            queue.finish(path, job, ok=False)
        raise
    finally:
        stop.set()
        beater.join(timeout=1)
