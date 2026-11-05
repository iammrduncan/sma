"""`sma worker --slots`: one run per GPU slot of the serving machine, many at once (owner decision 2026-11-01).

The dispatcher runs on the harness machine, one per serving machine. Each leg of a queued job is a unit: the
dispatcher starts it on the first free slot as a child process (`sma worker --run-unit JOB LEG --slot NAME`),
and as soon as a child ends, the next unit starts on that slot. Jobs are taken in the order they were queued;
within a job, its legs in order. A candidate leg (a 4-bit quant) is ready once its reference leg has a run id,
so a pair's three legs start together; the candidate's logprobs step waits for the reference's logits
(pipeline._wait_for_reference).

The dispatcher also guards the harness's memory for every slot. When it runs low, it stops only the run using
the most: that leg goes back to the queue, resumable; a second stop fails it for good and dead-letters it.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sma.atomic import atomic_write_json
from sma.config import Config, slot_config
from sma.errors import RunIncomplete, SmaError
from sma.logsetup import log_event
from sma.queue import JobQueue, leg_spec
from sma.records import utc_now
from sma.scrub import scrub_text

# A unit's exit codes (run_unit): done, some steps failed (incomplete), failed.
EXIT_DONE, EXIT_FAILED, EXIT_INCOMPLETE = 0, 1, 3
# A slot whose server fails to start this many times in a row is marked down and skipped.
SLOT_DOWN_AFTER = 2
# How long a stopping dispatcher waits for its children to tear down their servers: under the systemd unit's
# TimeoutStopSec (300 s), after which systemd kills what is left.
STOP_WAIT_S = 240.0


def dispatcher_id(cfg: Config) -> str:
    """The claim's worker id: stable across restarts, so a restarted dispatcher knows its own jobs."""
    return f"dispatcher-{cfg.host_alias}"


def state_path(cfg: Config) -> Path:
    """The dispatcher's state for `sma watch`: each slot, what runs on it, and the memory guard's last stop."""
    return cfg.queue_path() / "dispatcher" / f"{cfg.host_alias}.json"


def _terminal(leg: dict[str, Any], max_attempts: int) -> bool:
    status = leg.get("status")
    if status == "done":
        return True
    return status in ("incomplete", "failed") and (bool(leg.get("dead_lettered")) or int(leg.get("attempts") or 0) >= max_attempts)


def _time(value: Any) -> float | None:
    import calendar

    try:
        return float(calendar.timegm(time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


# Processes.


class ProcessLauncher:
    """Starts a unit as its own process, in its own session (a stop reaches it only through the dispatcher), on
    the harness's untimed CPUs, with its output appended to log_path."""

    def start(self, argv: list[str], log_path: Path, cpus: str | None) -> Any:
        from sma.proc import start_detached

        return start_detached(argv, log_path, cpus=_cpu_set(cpus) if cpus else None, what="a dispatcher unit")


class Adopted:
    """A unit started by an earlier dispatcher that is still running: watched by pid; its outcome is in the job."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        return None if unit_alive(self.pid) else -1

    def terminate(self) -> None:
        _signal(self.pid, signal.SIGTERM)

    def kill(self) -> None:
        _signal(self.pid, signal.SIGKILL)


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def unit_alive(pid: int | None) -> bool:
    """Whether pid is a live `sma worker --run-unit` process."""
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"--run-unit" in cmdline


def _cpu_set(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        if low:
            cpus.update(range(int(low), int(high or low) + 1))
    return cpus


# Memory.


def cgroup_dir(slice_name: str) -> Path:
    """A systemd slice's cgroup folder. A dash is a level: sma-harness.slice lives in sma.slice."""
    parts = slice_name.removesuffix(".slice").split("-")
    return Path("/sys/fs/cgroup").joinpath(*["-".join(parts[: index + 1]) + ".slice" for index in range(len(parts))])


def harness_memory(cfg: Config) -> tuple[int, str] | None:
    """(bytes left, where from). From the harness's slice (memory.guard_cgroup) when it exists and has a limit:
    memory.max less what it uses (memory.current less its inactive file cache, which the kernel takes back
    first). Otherwise the machine's MemAvailable."""
    from sma.telemetry import meminfo

    name = cfg.memory.guard_cgroup
    if name:
        folder = cgroup_dir(name)
        try:
            limit = (folder / "memory.max").read_text(encoding="utf-8").strip()
            current = int((folder / "memory.current").read_text(encoding="utf-8").strip())
            stat = dict(line.split() for line in (folder / "memory.stat").read_text(encoding="utf-8").splitlines() if line.count(" ") == 1)
            if limit.isdigit():
                used = current - int(stat.get("inactive_file", 0))
                return int(limit) - used, f"{name}'s memory.max"
        except (OSError, ValueError):
            pass
    reading = meminfo()
    return (reading[0], "the machine's MemAvailable") if reading else None


def process_tree_rss(pid: int) -> int:
    """Resident memory of a process and everything it started, in bytes (nel and its clients are a run's children)."""
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    total, todo, seen = 0, [pid], set()
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            for line in Path(f"/proc/{current}/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
        todo.extend(children.get(current, []))
    return total


def container_memory(run_id: str) -> int:
    """Memory of a run's containers on this machine (sma-<run_id>-*), from docker stats, in bytes."""
    from sma.proc import run_command

    result = run_command(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"], timeout_s=60, what="docker stats")
    total = 0
    units = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "kB": 1e3, "MB": 1e6, "GB": 1e9}
    for line in (result.stdout or "").splitlines():
        name, _, usage = line.partition("\t")
        if not name.startswith(f"sma-{run_id}-"):
            continue
        amount = usage.split("/", 1)[0].strip()
        for unit in sorted(units, key=len, reverse=True):
            if amount.endswith(unit):
                try:
                    total += int(float(amount[: -len(unit)]) * units[unit])
                except ValueError:
                    pass
                break
    return total


def kill_run_containers(run_id: str) -> list[str]:
    """Kill a run's containers on this machine at once (the memory guard)."""
    from sma.proc import run_command

    listed = run_command(["docker", "ps", "--format", "{{.Names}}"], timeout_s=60, what="docker ps")
    names = [name for name in (listed.stdout or "").split() if name.startswith(f"sma-{run_id}-")]
    for name in names:
        run_command(["docker", "kill", name], timeout_s=60, what=f"docker kill {name}")
    return names


# The dispatcher.


@dataclass
class Child:
    job_id: str
    leg: int
    slot: str
    handle: Any
    started: float
    guard_note: str | None = None


@dataclass
class Deps:
    """What the dispatcher does to the world, so tests can stand in for it."""

    launcher: Any = field(default_factory=ProcessLauncher)
    memory: Callable[[], tuple[int, str] | None] | None = None
    footprint: Callable[[Child, str | None], int] | None = None
    kill_containers: Callable[[str], list[str]] = kill_run_containers
    # The serving machine's runtime (servers_on, stop), for recovery. None: RemoteRuntime from the config.
    runtime: Any = None
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


class Dispatcher:
    def __init__(self, cfg: Config, *, config_path: Path | None = None, runs_dir: str | None = None, deps: Deps | None = None) -> None:
        if not cfg.remote.enabled or not cfg.remote.slots:
            raise SmaError("sma worker --slots needs remote.enabled and remote.slots in the config (the serving machine's GPUs).")
        self.cfg = cfg
        self.config_path = config_path
        self.runs_dir = runs_dir
        self.deps = deps or Deps()
        if self.deps.memory is None:
            self.deps.memory = lambda: harness_memory(cfg)
        if self.deps.footprint is None:
            self.deps.footprint = lambda child, run_id: process_tree_rss(child.handle.pid) + (container_memory(run_id) if run_id else 0)
        self.queue = JobQueue(cfg.queue_path())
        self.identity = dispatcher_id(cfg)
        self.children: dict[str, Child] = {}
        self.slot_failures: dict[str, int] = {slot.name: 0 for slot in cfg.remote.slots}
        self.down: dict[str, str] = {}
        self.guard_tripped = False
        self.last_guard: dict[str, Any] | None = None
        self.stopping = False
        self._beat = 0.0

    # The loop.

    def run(self) -> None:
        """Until SIGTERM or SIGINT: recover, then tick every queue.poll_s and check memory every second."""
        previous = {sig: signal.signal(sig, self._on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            self.recover()
            last = -1e9
            while not self.stopping:
                self.guard()
                if self.deps.clock() - last >= self.cfg.queue.poll_s:
                    last = self.deps.clock()
                    self.tick()
                self.deps.sleep(1.0)
        finally:
            self.shutdown()
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        log_event("dispatcher_stop", signal=signum)
        self.stopping = True

    def tick(self) -> None:
        """Reap ended units, start ready units on free slots, heartbeat the jobs it holds, write its state."""
        self.reap()
        self.sweep()
        self.start_ready()
        self.heartbeat()
        self.write_state()

    def sweep(self) -> None:
        """Close claimed jobs whose legs have all ended: a dispatcher that stopped between a leg's end and the
        job's close leaves one claimed."""
        for job in self._jobs():
            if job.get("state") == "claimed":
                self._finish_if_done(job["job_id"])

    # Units.

    def _jobs(self) -> list[dict[str, Any]]:
        """Jobs this dispatcher may run, oldest first: its own claimed jobs and eligible pending ones."""
        now = time.time()
        rows = []
        for job in self.queue.list_jobs():
            state = job.get("state")
            claim = job.get("claim") if isinstance(job.get("claim"), dict) else {}
            if state == "claimed" and claim.get("worker_id") != self.identity:
                continue
            if state not in ("pending", "claimed"):
                continue
            if not all(leg.get("gguf") for leg in job.get("legs") or []):
                continue  # The serving machine runs llama.cpp only.
            pin = job.get("pinned_host_alias")
            if pin and pin != self.cfg.host_alias:
                continue
            not_before = _time(job.get("not_before"))
            if state == "pending" and not_before is not None and not_before > now:
                continue
            rows.append(job)
        rows.sort(key=lambda job: (str(job.get("created") or ""), str(job["job_id"])))
        return rows

    def _running(self, job_id: str, index: int) -> bool:
        return any(child.job_id == job_id and child.leg == index for child in self.children.values())

    def next_unit(self) -> tuple[dict[str, Any], int] | str | None:
        """The first ready leg: jobs in queue order, legs in order. A candidate is ready once its reference leg
        has a run id (the reference started), so the candidate's record can name it. "wait" when the first leg in
        line is a candidate whose reference was just started and has no run id yet: it gets one within seconds,
        and the free slots wait for it rather than go to a later job."""
        now = time.time()
        for job in self._jobs():
            legs = job.get("legs") or []
            for index, leg in enumerate(legs):
                if self._running(job["job_id"], index) or _terminal(leg, self.cfg.queue.max_attempts):
                    continue
                if leg.get("status") == "running":
                    continue  # Another process runs it (recover adopts live units).
                retry_at = _time(leg.get("not_before"))
                if retry_at is not None and retry_at > now:
                    continue
                if index > 0 and not legs[0].get("run_id"):
                    if _terminal(legs[0], self.cfg.queue.max_attempts):
                        self._fail_leg(job["job_id"], index, "Its reference leg ended without a run, so there is nothing to compare it with.")
                    elif self._running(job["job_id"], 0):
                        return "wait"
                    continue
                return job, index
        return None

    def _busy(self) -> set[str]:
        """The slots running units hold. A unit across GPUs holds every slot its name joins with "+"."""
        return {part for child in self.children.values() for part in child.slot.split("+")}

    def _pick(self, free: list[str], need: int) -> list[str] | None:
        """Slots for a unit that needs `need` GPUs, in config order: neighbours aligned to `need` first (on the V100
        pod GPUs 0-1 and 2-3 share a PCIe switch), then any neighbours, then any. None while too few are free."""
        if need > len(free):
            return None
        order = [slot.name for slot in self.cfg.remote.slots]
        windows = [order[start : start + need] for start in range(len(order) - need + 1)]
        windows.sort(key=lambda window: order.index(window[0]) % need != 0)
        for window in windows:
            if all(name in free for name in window):
                return window
        return [name for name in order if name in free][:need]

    def start_ready(self) -> None:
        """Start ready legs on free slots, in queue order. A leg across GPUs (its "gpus", from its memory plan)
        waits until that many slots are free, and the slots that free up meanwhile wait with it: a later leg
        does not take them, so a wide leg is never starved by narrow ones."""
        if self.stopping:
            return
        for _ in range(4 * len(self.cfg.remote.slots)):
            busy = self._busy()
            free = [slot.name for slot in self.cfg.remote.slots if slot.name not in busy and slot.name not in self.down]
            if not free:
                return
            unit = self.next_unit()
            if unit is None or unit == "wait":
                return
            job, index = unit
            need = max(1, int((job["legs"][index] or {}).get("gpus") or 1))
            if need > len(self.cfg.remote.slots):
                self._fail_leg(job["job_id"], index, f"It needs {need} GPUs, more than the serving machine's {len(self.cfg.remote.slots)} slots.")
                continue
            chosen = self._pick(free, need)
            if chosen is None:
                return
            if job.get("state") == "pending" and not self.queue.claim_job(job["job_id"], self.identity, self.cfg.host_alias):
                continue  # Another worker took it between the read and the claim.
            self.start("+".join(chosen), job["job_id"], index)

    def unit_argv(self, slot: str, job_id: str, index: int) -> list[str]:
        argv = [sys.executable, "-m", "sma", "worker", "--run-unit", job_id, str(index), "--slot", slot]
        if self.config_path is not None:
            argv += ["--config", str(self.config_path)]
        if self.runs_dir:
            argv += ["--runs-dir", self.runs_dir]
        return argv

    def start(self, slot: str, job_id: str, index: int) -> None:
        log = self.cfg.queue_path() / "logs" / f"{self.cfg.host_alias}-{slot}.log"
        handle = self.deps.launcher.start(self.unit_argv(slot, job_id, index), log, self.cfg.harness.cpus_untimed)
        self.children[slot] = Child(job_id, index, slot, handle, self.deps.clock())

        def _mark(job: dict[str, Any]) -> None:
            leg = job["legs"][index]
            leg.update(status="running", slot=slot, pid=handle.pid, started=utc_now(), not_before=None)
            job["pinned_host_alias"] = self.cfg.host_alias

        self.queue.update(job_id, _mark)
        log_event("unit_start", job_id=job_id, leg=index, slot=slot, pid=handle.pid)

    def reap(self) -> None:
        for slot, child in list(self.children.items()):
            code = child.handle.poll()
            if code is None:
                continue
            del self.children[slot]
            self.settle(child, code)

    def settle(self, child: Child, code: int) -> None:
        """Record how a unit ended: done, back to the queue (a crash, a stop, the guard), or failed for good."""
        cfg = self.cfg
        dead_letter: tuple[dict[str, Any], str] | None = None
        server_failed = False

        def _settle(job: dict[str, Any]) -> None:
            nonlocal dead_letter, server_failed
            leg = job["legs"][child.leg]
            leg["pid"] = None
            status = leg.get("status")
            if child.guard_note is not None:
                if int(leg.get("guard_trips") or 0) >= 2:
                    leg.update(status="failed", dead_lettered=True, error="The memory guard stopped this leg twice.")
                    dead_letter = (dict(leg), "The memory guard stopped this leg twice (owner decision 2026-11-01).")
                else:
                    # Resumable, its attempts kept; it waits queue.retry_backoff_s so the memory can settle first.
                    retry = time.time() + cfg.queue.retry_backoff_s
                    leg.update(status="pending", not_before=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(retry)))
                return
            if status == "done":
                return
            if status == "running" or status is None:
                # The unit ended without recording an outcome: killed, or the dispatcher stopped it.
                if self.stopping:
                    leg["status"] = "pending"
                    return
                leg.update(status="failed", error=f"The unit's process ended (exit {code}) without recording how its run ended.")
            leg["attempts"] = int(leg.get("attempts") or 0) + 1
            server_failed = "failed to become healthy" in str(leg.get("error") or "")
            if not _terminal(leg, cfg.queue.max_attempts):
                delay = cfg.queue.retry_backoff_s * (cfg.queue.retry_backoff_factor ** max(0, leg["attempts"] - 1))
                leg["not_before"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + delay))

        job = self.queue.update(child.job_id, _settle)
        leg = job["legs"][child.leg]
        log_event("unit_end", job_id=child.job_id, leg=child.leg, slot=child.slot, code=code, status=leg.get("status"))
        if dead_letter is not None:
            self.queue.dead_letter(child.job_id, child.leg, *dead_letter)
        self._slot_health(child.slot, server_failed, leg.get("status") == "done")
        self._finish_if_done(child.job_id)

    def _slot_health(self, slot: str, server_failed: bool, ok: bool) -> None:
        if "+" in slot:
            return  # A server across GPUs that fails to start says nothing about any one of its slots.
        if ok or not server_failed:
            self.slot_failures[slot] = 0 if ok else self.slot_failures.get(slot, 0)
            return
        self.slot_failures[slot] = self.slot_failures.get(slot, 0) + 1
        if self.slot_failures[slot] >= SLOT_DOWN_AFTER:
            self.down[slot] = f"its server failed to start {self.slot_failures[slot]} times in a row"
            log_event("slot_down", slot=slot, reason=self.down[slot])

    def _fail_leg(self, job_id: str, index: int, error: str) -> None:
        def _fail(job: dict[str, Any]) -> None:
            job["legs"][index].update(status="failed", error=error, attempts=self.cfg.queue.max_attempts)

        self.queue.update(job_id, _fail)
        self._finish_if_done(job_id)

    def _finish_if_done(self, job_id: str) -> None:
        job = self.queue.read(job_id)
        if job is None or job.get("state") != "claimed":
            return
        legs = job.get("legs") or []
        if any(self._running(job_id, index) for index in range(len(legs))):
            return
        if all(_terminal(leg, self.cfg.queue.max_attempts) for leg in legs):
            ok = all(leg.get("status") == "done" for leg in legs)
            self.queue.finish_job(job_id, ok=ok)
            log_event("job_done", job_id=job_id, ok=ok)

    def heartbeat(self) -> None:
        if self.deps.clock() - self._beat < self.cfg.queue.heartbeat_s:
            return
        self._beat = self.deps.clock()
        for job_id in {child.job_id for child in self.children.values()}:
            self.queue.update(job_id, lambda job: job.update(heartbeat=utc_now()))

    # Recovery and stopping.

    def recover(self) -> None:
        """On start: adopt units still running from an earlier dispatcher, put back the legs whose unit is gone
        (resumable, first in line by queue order), and stop servers on the serving machine that no live unit owns."""
        live_runs: set[str] = set()
        for job in self.queue.list_jobs():
            claim = job.get("claim") if isinstance(job.get("claim"), dict) else {}
            if job.get("state") != "claimed" or claim.get("worker_id") != self.identity:
                continue
            for index, leg in enumerate(job.get("legs") or []):
                if leg.get("status") != "running":
                    continue
                slot = str(leg.get("slot") or "")
                parts = set(slot.split("+"))
                known = parts <= {item.name for item in self.cfg.remote.slots}
                if unit_alive(leg.get("pid")) and slot and known and not parts & self._busy():
                    self.children[slot] = Child(job["job_id"], index, slot, Adopted(int(leg["pid"])), self.deps.clock())
                    if leg.get("run_id"):
                        live_runs.add(str(leg["run_id"]))
                    log_event("unit_adopted", job_id=job["job_id"], leg=index, slot=slot, pid=leg["pid"])
                    continue
                self.queue.update(job["job_id"], lambda j, i=index: j["legs"][i].update(status="pending", pid=None))
                log_event("unit_requeued", job_id=job["job_id"], leg=index, reason="its process is gone")
        runtime = self.deps.runtime
        if runtime is None:
            from sma.remote import RemoteRuntime

            runtime = RemoteRuntime(self.cfg)
        for name in runtime.servers_on():
            if not any(run_id in name for run_id in live_runs):
                log_event("orphan_server_stop", name=name)
                runtime.stop(name)

    def shutdown(self) -> None:
        """Stop every unit: each tears down its own server; their legs stay resumable."""
        self.stopping = True
        for child in self.children.values():
            child.handle.terminate()
        deadline = self.deps.clock() + STOP_WAIT_S
        while self.children and self.deps.clock() < deadline:
            self.reap()
            if self.children:
                self.deps.sleep(1.0)
        for child in list(self.children.values()):
            child.handle.kill()
        self.reap()
        self.write_state()

    # The memory guard.

    def guard(self) -> None:
        """Below memory.guard_free_gb, stop the run using the most memory of the harness's (once per drop)."""
        floor = self.cfg.memory.guard_free_gb * 1e9
        if floor <= 0:
            return
        reading = self.deps.memory() if self.deps.memory else None
        if reading is None:
            return
        left, source = reading
        if left >= floor:
            self.guard_tripped = False
            return
        if self.guard_tripped or not self.children:
            return
        self.guard_tripped = True
        sizes: dict[str, tuple[int, str | None]] = {}
        for slot, child in self.children.items():
            job = self.queue.read(child.job_id) or {}
            run_id = ((job.get("legs") or [{}])[child.leg] or {}).get("run_id")
            sizes[slot] = (self.deps.footprint(child, run_id) if self.deps.footprint else 0, run_id)
        slot = max(sizes, key=lambda name: sizes[name][0])
        victim, (size, run_id) = self.children[slot], sizes[slot]
        note = (
            f"The memory guard stopped this run on slot {slot}: the harness had {left / 1e9:.1f} GB left ({source}), under "
            f"memory.guard_free_gb ({self.cfg.memory.guard_free_gb:g} GB), and this run used the most of the {len(self.children)} "
            f"running ({size / 1e9:.1f} GB). The run is resumable; a second stop fails it."
        )
        victim.guard_note = note

        def _trip(job: dict[str, Any]) -> None:
            leg = job["legs"][victim.leg]
            leg["guard_trips"] = int(leg.get("guard_trips") or 0) + 1
            leg.setdefault("guard_notes", []).append({"at": utc_now(), "note": note})

        self.queue.update(victim.job_id, _trip)
        killed = self.deps.kill_containers(run_id) if run_id else []
        victim.handle.terminate()
        self.last_guard = {"at": utc_now(), "slot": slot, "run_id": run_id, "left_bytes": left, "source": source, "killed": killed}
        log_event("memory_guard", slot=slot, run_id=run_id, left=left, killed=killed)

    # State for sma watch.

    def write_state(self) -> None:
        slots = []
        owners = {part: child for child in self.children.values() for part in child.slot.split("+")}
        for slot in self.cfg.remote.slots:
            child = owners.get(slot.name)
            row: dict[str, Any] = {"name": slot.name, "gpus": slot.gpus, "state": "down" if slot.name in self.down else "busy" if child else "idle"}
            if child is not None:
                job = self.queue.read(child.job_id) or {}
                leg = (job.get("legs") or [{}])[child.leg] or {}
                row.update(job_id=child.job_id, leg=child.leg, run_id=leg.get("run_id"), pid=child.handle.pid,
                           repo=leg.get("repo"), gguf=leg.get("gguf"), thinking=leg.get("thinking"))  # fmt: skip
                if "+" in child.slot:
                    row["unit_slot"] = child.slot  # A run across GPUs: every slot it holds.
            if slot.name in self.down:
                row["reason"] = self.down[slot.name]
            slots.append(row)
        atomic_write_json(state_path(self.cfg), {
            "pid": os.getpid(), "updated": utc_now(), "host_alias": self.cfg.host_alias, "slots": slots,
            "memory_guard": self.last_guard, "stopping": self.stopping,
        })  # fmt: skip


def dispatcher_alive(cfg: Config) -> bool:
    """Whether this config's dispatcher is running, by the pid in its state file."""
    try:
        pid = int(json.loads(state_path(cfg).read_text(encoding="utf-8"))["pid"])
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return b"--slots" in cmdline


def _stop_unit(pid: int) -> None:
    if unit_alive(pid):
        _signal(pid, signal.SIGTERM)


def drop_held_job(
    cfg: Config, job_id: str, *, stop: Callable[[int], None] = _stop_unit, sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:  # fmt: skip
    """Take a job the running dispatcher holds off the queue, and leave its other slots running. Each leg not done
    is marked failed for good, so nothing starts it again, and its unit gets SIGTERM: it tears down its server as on
    a dispatcher stop, and its run stays resumable by hand. The dispatcher then closes the job, and it moves to
    removed/. Until then its file stays where the dispatcher reads it: a job that vanished would stop the dispatcher."""
    queue = JobQueue(cfg.queue_path())
    note = f"Dropped from the queue at {utc_now()} (sma queue drop)."
    units: list[int] = []

    def _mark(job: dict[str, Any]) -> None:
        for leg in job.get("legs") or []:
            if leg.get("status") == "done":
                continue
            if leg.get("status") == "running" and leg.get("pid"):
                units.append(int(leg["pid"]))
            attempts = max(int(leg.get("attempts") or 0), cfg.queue.max_attempts)
            leg.update(status="failed", error=note, not_before=None, attempts=attempts)

    queue.update(job_id, _mark)
    for pid in units:
        stop(pid)
    deadline = clock() + STOP_WAIT_S + 3 * cfg.queue.poll_s
    while (queue.read(job_id) or {}).get("state") == "claimed":
        if clock() > deadline:
            raise SmaError(
                f"{job_id}'s legs are marked failed and its units were stopped, but the dispatcher has not closed the job. "
                f"Run sma queue drop {job_id} again once it has."
            )
        sleep(1.0)
    return queue.drop(job_id, stale_s=cfg.queue.stale_claim_s)


def run_dispatcher(cfg: Config, *, config_path: Path | None, runs_dir: str | None) -> None:
    from sma.identity import assert_alias_safe

    assert_alias_safe(cfg.host_alias)
    Dispatcher(cfg, config_path=config_path, runs_dir=runs_dir).run()


# A unit: one leg on one slot (the child process).


def run_unit(
    cfg: Config, services_factory: Any, job_id: str, index: int, slot: str, *, executor: Any = None, spec_cls: Any = None
) -> int:
    """Run or resume one leg of a job on one slot, and record how it ended in the job. Returns the exit code."""
    if executor is None or spec_cls is None:
        from sma.pipeline import RunSpec, execute

        executor = executor or execute
        spec_cls = spec_cls or RunSpec
    slot_cfg = slot_config(cfg, slot)
    queue = JobQueue(cfg.queue_path())
    job = queue.read(job_id)
    if job is None:
        raise SmaError(f"No queue job {job_id}")
    legs = job.get("legs") or []
    if not 0 <= index < len(legs):
        raise SmaError(f"{job_id} has no leg {index}")
    leg = legs[index]
    reference = legs[0].get("run_id") if index > 0 else None
    if index > 0 and not reference:
        raise SmaError(f"Leg {index} of {job_id} compares with leg 0, which has no run yet.")
    notes = tuple(item["note"] for item in leg.get("guard_notes") or [] if isinstance(item, dict) and not item.get("recorded"))

    def _created(run_id: str) -> None:
        def _mark(job: dict[str, Any]) -> None:
            current = job["legs"][index]
            current.update(run_id=run_id, status="running", slot=slot, pid=os.getpid())
            for item in current.get("guard_notes") or []:
                if isinstance(item, dict):
                    item["recorded"] = True
            job["pinned_host_alias"] = cfg.host_alias

        queue.update(job_id, _mark)

    spec = leg_spec(slot_cfg, job, index, spec_cls, reference_id=reference, resume=leg.get("run_id"), lock_held=False, notes=notes)
    outcome: dict[str, Any]
    try:
        run_id = executor(spec, slot_cfg, services_factory(slot_cfg), on_created=_created)
        outcome, code = {"status": "done", "run_id": run_id, "error": None}, EXIT_DONE
    except RunIncomplete as exc:
        outcome, code = {"status": "incomplete", "run_id": exc.run_id, "error": scrub_text(str(exc))[:2000]}, EXIT_INCOMPLETE
    except Exception as exc:
        outcome, code = {"status": "failed", "error": scrub_text(str(exc))[:2000]}, EXIT_FAILED

    def _record(job: dict[str, Any]) -> None:
        job["legs"][index].update(**outcome, finished=utc_now())

    queue.update(job_id, _record)
    log_event("unit_outcome", job_id=job_id, leg=index, slot=slot, status=outcome["status"])
    return code
