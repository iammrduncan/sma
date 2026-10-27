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

    def add(self, legs: list[dict[str, Any]], *, kind: str, tool_parser: str | None = None) -> dict[str, Any]:
        if not legs:
            raise SmaError("A queue job needs at least one checkpoint")
        engines = {item.get("engine") or "vllm" for item in legs}
        if len(engines) != 1:
            raise SmaError("Every leg of a job must use the same engine.")
        job_id = "job-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        job = {
            "schema_version": 1,
            "job_id": job_id,
            "kind": kind,
            "created": utc_now(),
            "legs": [_leg(item) for item in legs],
            "claim": None,
            "heartbeat": None,
            "error": None,
            "attempts": 0,
            "pinned_host_alias": None,
            "engine": legs[0].get("engine") or "vllm",
            # Both legs of a pair are the same model, so they share the parser.
            "tool_parser": tool_parser,
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
        """Claim a pending job. The claimed file is written with a heartbeat already set."""
        claimed = self.path / "claimed"
        now_dt = datetime.now(timezone.utc)
        for path, job in self._eligible_pending(now_dt):
            pin = job.get("pinned_host_alias")
            if pin and pin != host_alias:
                continue
            dest = claimed / path.name
            if dest.exists():
                continue
            now = utc_now()
            job["claim"] = {"worker_id": worker_id, "host_alias": host_alias, "when": now}
            job["heartbeat"] = now
            tmp = claimed / f".{path.name}.{uuid.uuid4().hex}.tmp"
            atomic_write_json(tmp, job)
            try:
                os.link(tmp, dest)
            except FileExistsError:
                tmp.unlink(missing_ok=True)
                continue
            except FileNotFoundError:
                tmp.unlink(missing_ok=True)
                continue
            tmp.unlink(missing_ok=True)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
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

    def release_to_pending(self, path: Path, job: dict[str, Any], *, delay_s: float = 0) -> None:
        """Return a claimed job to pending without clearing its host pin or run ids.

        delay_s pushes not_before into the future so the worker does not retry
        this job before other pending jobs. Retried jobs also sort behind jobs
        that have not failed yet.
        """
        job["claim"] = None
        job["heartbeat"] = None
        if delay_s > 0:
            when = datetime.now(timezone.utc).timestamp() + delay_s
            job["not_before"] = datetime.fromtimestamp(when, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        dest = self.path / "pending" / path.name
        atomic_write_json(dest, job)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def reclaim_stale(self, stale_s: float, *, grace_s: float = 30.0) -> list[str]:
        now = datetime.now(timezone.utc)
        moved: list[str] = []
        for path in sorted((self.path / "claimed").glob("*.json")):
            if path.name.startswith("."):
                continue
            job = _read(path)
            beat = _parse_time(job.get("heartbeat")) or _parse_time(
                job.get("claim", {}).get("when") if isinstance(job.get("claim"), dict) else None
            )
            if beat is None:
                age = now.timestamp() - path.stat().st_mtime
                if age < grace_s:
                    continue
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

    def unpin(self, job_id: str) -> dict[str, Any]:
        """Drop the host pin and run ids. The next claim starts a fresh run."""
        return self._reassign(job_id, host_alias=None)

    def reassign(self, job_id: str, host_alias: str) -> dict[str, Any]:
        """Pin the job to another alias and clear run ids so that host starts fresh.

        The previous run folder is not resumed. Cross-host resume stays refused.
        """
        assert_alias_safe(host_alias)
        return self._reassign(job_id, host_alias=host_alias)

    def _reassign(self, job_id: str, *, host_alias: str | None) -> dict[str, Any]:
        found = self._find(job_id)
        if found is None:
            raise SmaError(f"No queue job {job_id}")
        path, job, state = found
        if state == "claimed":
            raise SmaError(
                f"{job_id} is claimed. Reclaim it before unpinning or reassigning it."
            )
        previous = job.get("pinned_host_alias")
        for leg in job.get("legs") or []:
            leg["run_id"] = None
            leg["status"] = "pending"
            leg["reference_run_id"] = None
        job["pinned_host_alias"] = host_alias
        job["claim"] = None
        job["heartbeat"] = None
        job["not_before"] = None
        job["attempts"] = 0
        job["error"] = (
            "Reassigned. The next host starts this job fresh and will not resume the previous run."
        )
        job["reassigned_from"] = previous
        dest = self.path / "pending" / path.name
        atomic_write_json(dest, job)
        if path != dest:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return job

    def _find(self, job_id: str) -> tuple[Path, dict[str, Any], str] | None:
        for state in ("pending", "claimed", "done", "failed"):
            path = self.path / state / f"{job_id}.json"
            if not path.is_file():
                continue
            return path, _read(path), state
        return None

    def _eligible_pending(self, now: datetime) -> list[tuple[Path, dict[str, Any]]]:
        pending = self.path / "pending"
        rows: list[tuple[Path, dict[str, Any]]] = []
        for path in pending.glob("*.json"):
            if path.name.startswith("."):
                continue
            try:
                job = _read(path)
            except FileNotFoundError:
                # Another Spark claimed the file between glob and read.
                continue
            except OSError as exc:
                from sma.logsetup import log_event

                log_event("queue_pending_unreadable", file=path.name, error=str(exc))
                continue
            except (json.JSONDecodeError, SmaError) as exc:
                self._quarantine(path, exc)
                continue
            not_before = _parse_time(job.get("not_before"))
            if not_before is not None and not_before > now:
                continue
            rows.append((path, job))

        def _key(item: tuple[Path, dict[str, Any]]) -> tuple[int, str, str, str]:
            path, job = item
            retried = 1 if int(job.get("attempts") or 0) else 0
            return (retried, str(job.get("not_before") or ""), str(job.get("created") or ""), path.name)

        rows.sort(key=_key)
        return rows

    def _quarantine(self, path: Path, exc: BaseException) -> None:
        from sma.logsetup import log_event

        dest_dir = self.path / "corrupt"
        dest_dir.mkdir(parents=True, exist_ok=True)
        log_event("queue_corrupt", file=path.name, error=str(exc))
        (dest_dir / f"{path.name}.error").write_text(str(exc)[:2000], encoding="utf-8")
        try:
            os.replace(path, dest_dir / path.name)
        except OSError as move_exc:
            log_event("queue_corrupt_move_failed", file=path.name, error=str(move_exc))


def _leg(item: dict[str, Any]) -> dict[str, Any]:
    quant = item["quant"]
    if quant not in {"bf16", "nvfp4", "mxfp4", "mixed", "gguf"}:
        raise SmaError(f"Unsupported quant {quant}")
    if (quant == "gguf") != bool(item.get("gguf")):
        raise SmaError("A GGUF job needs its quant file (--gguf QUANT), and --gguf goes with a GGUF repo only.")
    return {
        "repo": item["repo"],
        "quant": quant,
        "gguf": item.get("gguf"),
        # The --engine the job was queued with. None lets the format and preflight choose.
        "engine_requested": item.get("engine_requested"),
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
    executor: Any = None,
    spec_cls: Any = None,
    exit_when_idle: bool = False,
) -> int:
    """Pull whole jobs. One bad model is recorded and the worker keeps going."""
    import time

    from sma.locks import GpuLock
    from sma.logsetup import log_event

    if executor is None or spec_cls is None:
        from sma.pipeline import RunSpec, execute

        executor = executor or execute
        spec_cls = spec_cls or RunSpec

    assert_alias_safe(cfg.host_alias)
    queue = JobQueue(cfg.queue_path())
    identity = worker_id or ("worker-" + uuid.uuid4().hex[:8])
    completed = 0
    while True:
        lock = GpuLock(cfg.lock_file())
        try:
            lock.acquire()
        except SmaError as exc:
            if once or exit_when_idle:
                raise
            log_event("worker_lock_busy", error=str(exc))
            time.sleep(cfg.queue.poll_s)
            continue
        claimed = None
        try:
            claimed = queue.claim(identity, cfg.host_alias)
            if claimed is not None:
                path, job = claimed
                try:
                    _run_job(cfg, services_factory, queue, path, job, executor, spec_cls)
                    completed += 1
                except Exception as exc:
                    log_event("worker_job_failed", job_id=job.get("job_id"), error=str(exc))
        finally:
            lock.release()
        if claimed is None:
            if once or exit_when_idle:
                return completed
            time.sleep(cfg.queue.poll_s)
            continue
        if once:
            return completed


def _run_job(cfg: Config, services_factory: Any, queue: JobQueue, path: Path, job: dict[str, Any], execute: Any, run_spec: Any) -> None:
    if (job.get("engine") or cfg.engine.name) == "sglang":
        from sma.backends import assert_sglang_dataset_covers

        assert_sglang_dataset_covers(cfg.speed.num_prompts)
    reference_id = None
    job_lock = threading.Lock()
    stop = threading.Event()

    def _pulse() -> None:
        interval = max(1.0, float(cfg.queue.heartbeat_s))
        while True:
            with job_lock:
                if path.is_file():
                    queue.heartbeat(path, job)
            if stop.wait(interval):
                break

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
                    job["pinned_host_alias"] = cfg.host_alias
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
                    # A single job lets the format and preflight choose unless --engine was given. A pair's
                    # legs share the job's engine. Jobs queued before engine_requested keep their engine.
                    engine=(
                        leg["engine_requested"]
                        if "engine_requested" in leg and len(job["legs"]) == 1
                        else (job.get("engine") or cfg.engine.name)
                    ),
                    selected_by=None if len(job["legs"]) == 1 or "engine_requested" not in leg else ("flag" if leg["engine_requested"] else "default"),
                    tool_parser=job.get("tool_parser"),
                    gguf=leg.get("gguf"),
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
            job["attempts"] = int(job.get("attempts") or 0) + 1
            job["error"] = scrub_text(str(exc))[:2000]
            for leg in job["legs"]:
                if leg.get("status") == "running":
                    leg["status"] = "failed"
            if job["attempts"] >= cfg.queue.max_attempts:
                queue.finish(path, job, ok=False)
            else:
                delay = cfg.queue.retry_backoff_s * (
                    cfg.queue.retry_backoff_factor ** max(0, job["attempts"] - 1)
                )
                queue.release_to_pending(path, job, delay_s=delay)
        raise
    finally:
        stop.set()
        beater.join(timeout=1)
