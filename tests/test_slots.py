"""The V100 slot scheduler (owner decisions 2026-11-01): slots in the config, the per-job lock, the KL wait, the
dispatcher's units, the memory guard's requeue and dead letter, the FIFO turns, and queue drop."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sma.config import config_from_mapping, slot_config, validate_config
from sma.dispatcher import Deps, Dispatcher, cgroup_dir, run_unit
from sma.errors import RunIncomplete, SmaError
from sma.queue import JobQueue, pair_legs

SLOTS = [
    {"name": f"v100-{index}", "gpus": str(index), "remote_port_env": f"V100_MODEL{index + 1}_PORT", "local_port": 18001 + index, "search_port": 8791 + index}
    for index in range(4)
]


def _cfg(tmp_path: Path, **overrides: Any):
    raw = {
        "host_alias": "v100-pod",
        "runs_dir": str(tmp_path / "runs"),
        "lock_path": str(tmp_path / "locks" / "gpu.lock"),
        "queue": {"dir": str(tmp_path / "queue"), "max_attempts": 2, "retry_backoff_s": 0},
        "harness": {"lock_dir": str(tmp_path / "locks")},
        "memory": {"guard_free_gb": 4},
        "remote": {"enabled": True, "slots": SLOTS, "threads_per_slot": 3},
        **overrides,
    }
    cfg = config_from_mapping(raw)
    validate_config(cfg)
    return cfg


@pytest.fixture(autouse=True)
def _ports(monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(4):
        monkeypatch.setenv(f"V100_MODEL{index + 1}_PORT", str(8001 + index))


# Step 1: slots in the config.


def test_a_slot_config_fills_in_the_slots_gpu_ports_and_lock(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    one = slot_config(cfg, "v100-2")
    assert (one.remote.gpus, one.remote.remote_port, one.remote.slot) == ("2", 8003, "v100-2")
    assert one.engine.port == 18003 and (one.browsecomp.search_port, one.browsecomp.local_port) == (8793, 8793)
    assert one.lock_file().name == "v100-v100-2.lock" and one.lock_file() != slot_config(cfg, "v100-1").lock_file()
    assert cfg.remote.slot is None and cfg.engine.port != 18003  # The base config is unchanged.
    with pytest.raises(SmaError, match="No slot v100-9"):
        slot_config(cfg, "v100-9")


@pytest.mark.parametrize(
    ("slots", "remote", "message"),
    [
        ([{**SLOTS[0]}, {**SLOTS[1], "gpus": "0"}], {}, "must not share a GPU"),
        ([{**SLOTS[0]}, {**SLOTS[1], "local_port": 18001}], {}, "local_port values must be unique"),
        ([{**SLOTS[0], "gpus": "0,1"}], {}, "one GPU per slot"),
        ([{"name": "a", "gpus": "0"}], {}, "needs remote_port or remote_port_env"),
        ([{**SLOTS[0], "bogus": 1}], {}, "unknown keys"),
        ([SLOTS[0]], {"enabled": False}, "need remote.enabled"),
        ([SLOTS[0]], {"threads_per_slot": 0}, "threads_per_slot"),
    ],
)
def test_bad_slots_are_refused(tmp_path: Path, slots: list[dict[str, Any]], remote: dict[str, Any], message: str) -> None:
    with pytest.raises(SmaError, match=message):
        _cfg(tmp_path, remote={"enabled": True, "slots": slots, **remote})


def test_a_slot_whose_port_is_not_in_the_env_is_refused_when_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("V100_MODEL4_PORT")
    with pytest.raises(SmaError, match="V100_MODEL4_PORT"):
        slot_config(_cfg(tmp_path), "v100-3")


# Step 2: threads on the serving machine.


def test_llama_cpp_on_the_serving_machine_gets_the_slots_threads(tmp_path: Path) -> None:
    from sma.remote import RemoteRuntime, thread_args

    cfg = slot_config(_cfg(tmp_path), "v100-1")
    assert thread_args(cfg) == ["--threads", "3", "--threads-batch", "3"]
    runner = SimpleNamespace(command=lambda script: ["ssh", "pod", script])
    argv = RemoteRuntime(cfg, runner=runner, local=object()).perplexity_command(["-m", "/work/m.gguf"])
    assert "CUDA_VISIBLE_DEVICES=1" in argv[2] and argv[2].endswith("-m /work/m.gguf --threads 3 --threads-batch 3")
    cfg.remote.threads_per_slot = None
    assert thread_args(cfg) == []


# Step 3: the per-job lock.


def _bump(queue_dir: str, job_id: str, key: str, rounds: int) -> None:
    queue = JobQueue(Path(queue_dir))
    for _ in range(rounds):
        def _add(job: dict[str, Any]) -> None:
            value = int(job.get(key) or 0)
            time.sleep(0.001)  # Widen the window a missing lock would lose an update in.
            job[key] = value + 1

        queue.update(job_id, _add)


def test_two_writers_on_one_job_keep_both_updates(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue")
    job = queue.add([{"repo": "r", "quant": "gguf", "gguf": "F16"}], kind="single")
    ctx = multiprocessing.get_context("fork")
    workers = [ctx.Process(target=_bump, args=(str(tmp_path / "queue"), job["job_id"], key, 40)) for key in ("a", "b", "a")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(30)
    final = queue.read(job["job_id"])
    assert final is not None and (final["a"], final["b"]) == (80, 40)
    assert final["legs"][0]["guard_trips"] == 0 and final["legs"][0]["attempts"] == 0  # Legs carry their own state.


# Step 4: the KL wait.


def _reference(tmp_path: Path) -> Path:
    ref = tmp_path / "runs" / "ref"
    (ref / "steps").mkdir(parents=True)
    (ref / "manifest.json").write_text(json.dumps({"steps": {"logprobs": {"status": "running"}}, "timestamps": {}}))
    return ref


def _candidate(tmp_path: Path, wait_s: float):
    from sma.pipeline import Ctx

    cfg = _cfg(tmp_path)
    cfg.logprobs.reference_wait_s = wait_s
    cfg.queue.poll_s = 0.01
    run_dir = tmp_path / "runs" / "cand"
    (run_dir / "logprobs").mkdir(parents=True)
    return Ctx(cfg=cfg, services=None, run_dir=run_dir, manifest={"run_id": "cand", "warnings": []})


def test_a_candidate_waits_for_its_references_logits(tmp_path: Path) -> None:
    from sma import pipeline

    ref, ctx = _reference(tmp_path), _candidate(tmp_path, 30)
    seen: list[bool] = []

    def finish_later() -> None:
        time.sleep(0.3)
        seen.append((ctx.run_dir / "logprobs" / "waiting.json").is_file())
        (ref / "steps" / "logprobs.done").write_text("{}")
        time.sleep(0.2)  # The marker first, then the manifest: the wait needs both.
        (ref / "manifest.json").write_text(json.dumps({"steps": {"logprobs": {"status": "done"}}, "timestamps": {}}))

    thread = threading.Thread(target=finish_later)
    thread.start()
    assert pipeline._wait_for_reference(ctx, "ref", ref) is None
    thread.join()
    assert seen == [True] and not (ctx.run_dir / "logprobs" / "waiting.json").exists()


def test_a_failed_or_late_reference_leaves_kl_null_with_the_reason(tmp_path: Path) -> None:
    from sma import pipeline

    ref, ctx = _reference(tmp_path), _candidate(tmp_path, 0.05)
    assert "within logprobs.reference_wait_s" in pipeline._wait_for_reference(ctx, "ref", ref)
    (ref / "manifest.json").write_text(json.dumps({"steps": {"logprobs": {"status": "failed", "error": "llama-perplexity crashed\nmore"}}, "timestamps": {}}))
    assert pipeline._wait_for_reference(ctx, "ref", ref) == "failed its logprobs step (llama-perplexity crashed)"
    (ref / "manifest.json").write_text(json.dumps({"steps": {"logprobs": {"status": "pending"}}, "timestamps": {"finished": "x"}}))
    assert pipeline._wait_for_reference(ctx, "ref", ref) == "ended without its logprobs step"


# Step 5: the unit (the dispatcher's child).


def _pair_job(queue: JobQueue, repo: str = "LiquidAI/LFM3-1.7B-GGUF", quant: str = "QAD-Q4_0") -> dict[str, Any]:
    reference = {"repo": repo, "quant": "gguf", "gguf": "F16", "engine": "llamacpp", "engine_requested": None}
    candidate = {"repo": repo, "quant": "gguf", "gguf": quant, "engine": "llamacpp", "engine_requested": None}
    return queue.add(pair_legs(reference, candidate), kind="pair")


class FakeExecutor:
    def __init__(self, outcome: str = "done") -> None:
        self.outcome = outcome
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, spec: Any, cfg: Any, services: Any, *, on_created: Any) -> str:
        self.calls.append((spec, cfg))
        run_id = spec.resume_run_id or f"run-{len(self.calls)}"
        on_created(run_id)
        if self.outcome == "incomplete":
            raise RunIncomplete(run_id, ["quality"])
        if self.outcome == "failed":
            raise SmaError("fetch failed")
        return run_id


@pytest.mark.parametrize(("outcome", "code"), [("done", 0), ("incomplete", 3), ("failed", 1)])
def test_a_unit_runs_its_leg_on_its_slot_and_records_how_it_ended(tmp_path: Path, outcome: str, code: int) -> None:
    from sma.pipeline import RunSpec

    cfg = _cfg(tmp_path)
    queue = JobQueue(cfg.queue_path())
    job = _pair_job(queue)
    queue.update(job["job_id"], lambda j: j["legs"][0].update(run_id="ref-run", status="running"))
    executor = FakeExecutor(outcome)
    assert run_unit(cfg, lambda c: c, job["job_id"], 2, "v100-3", executor=executor, spec_cls=RunSpec) == code
    spec, used = executor.calls[0]
    assert (spec.gguf, spec.thinking, spec.reference_run_id, spec.resume_run_id, spec.lock_held) == ("QAD-Q4_0", "on", "ref-run", None, False)
    assert used.remote.slot == "v100-3" and used.engine.port == 18004
    leg = queue.read(job["job_id"])["legs"][2]
    assert leg["status"] == outcome and leg["run_id"] == ("run-1" if outcome != "failed" else "run-1") and leg["slot"] == "v100-3"


def test_a_unit_resumes_its_legs_run_and_records_the_guards_note(tmp_path: Path) -> None:
    from sma.pipeline import RunSpec

    cfg = _cfg(tmp_path)
    queue = JobQueue(cfg.queue_path())
    job = _pair_job(queue)
    queue.update(job["job_id"], lambda j: j["legs"][0].update(run_id="old-run", status="pending", guard_trips=1, guard_notes=[{"at": "t", "note": "stopped"}]))
    executor = FakeExecutor()
    run_unit(cfg, lambda c: c, job["job_id"], 0, "v100-0", executor=executor, spec_cls=RunSpec)
    spec, _ = executor.calls[0]
    assert spec.resume_run_id == "old-run" and spec.notes == ("stopped",)
    assert queue.read(job["job_id"])["legs"][0]["guard_notes"][0]["recorded"] is True


# Step 6: the dispatcher.


class FakeHandle:
    def __init__(self, pid: int) -> None:
        self.pid, self.code, self.signals = pid, None, []

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.signals.append("TERM")
        self.code = 130 if self.code is None else self.code

    def kill(self) -> None:
        self.signals.append("KILL")


class FakeLauncher:
    """Starts no process: a unit 'runs' until the test ends it. With created=True a unit records its run id at
    once, as a real one does within seconds."""

    def __init__(self, queue: JobQueue, *, created: bool = True) -> None:
        self.queue, self.created = queue, created
        self.started: list[tuple[str, int, str]] = []
        self.handles: dict[tuple[str, int], FakeHandle] = {}

    def start(self, argv: list[str], log_path: Path, cpus: str | None) -> FakeHandle:
        job_id, leg, slot = argv[argv.index("--run-unit") + 1], int(argv[argv.index("--run-unit") + 2]), argv[argv.index("--slot") + 1]
        self.started.append((job_id, leg, slot))
        handle = self.handles[(job_id, leg)] = FakeHandle(1000 + len(self.started))
        if self.created:
            self.record_run(job_id, leg)
        return handle

    def record_run(self, job_id: str, leg: int) -> None:
        self.queue.update(job_id, lambda j: j["legs"][leg].update(run_id=j["legs"][leg].get("run_id") or f"{job_id[-4:]}-{leg}", status="running"))

    def end(self, job_id: str, leg: int, status: str = "done", code: int = 0) -> None:
        if status != "crash":
            self.queue.update(job_id, lambda j: j["legs"][leg].update(status=status, error=None if status == "done" else "boom"))
        self.handles[(job_id, leg)].code = code


class FakeServing:
    def __init__(self, servers: dict[str, str] | None = None) -> None:
        self.servers = dict(servers or {})
        self.stopped: list[str] = []

    def servers_on(self) -> dict[str, str]:
        return dict(self.servers)

    def stop(self, name: str) -> None:
        self.stopped.append(name)
        self.servers.pop(name, None)


def _dispatcher(tmp_path: Path, *, created: bool = True, memory: Any = None, footprint: Any = None, servers: dict[str, str] | None = None):
    cfg = _cfg(tmp_path)
    queue = JobQueue(cfg.queue_path())
    launcher = FakeLauncher(queue, created=created)
    killed: list[str] = []
    deps = Deps(
        launcher=launcher, memory=memory or (lambda: (100 * 10**9, "test")), footprint=footprint or (lambda child, run_id: 1),
        kill_containers=lambda run_id: killed.append(run_id) or [f"sma-{run_id}-quality-bfcl_v3"], runtime=FakeServing(servers),
    )  # fmt: skip
    dispatcher = Dispatcher(cfg, config_path=Path("sma-v100.yaml"), deps=deps)
    return dispatcher, queue, launcher, killed


def test_the_worked_example_four_units_start_at_once_and_the_next_takes_the_first_free_slot(tmp_path: Path) -> None:
    dispatcher, queue, launcher, _ = _dispatcher(tmp_path)
    small = _pair_job(queue)
    vision = _pair_job(queue, "LiquidAI/LFM3-VL-2B-GGUF")
    dispatcher.recover()
    dispatcher.tick()
    assert launcher.started == [(small["job_id"], 0, "v100-0"), (small["job_id"], 1, "v100-1"), (small["job_id"], 2, "v100-2"), (vision["job_id"], 0, "v100-3")]
    assert queue.read(small["job_id"])["state"] == "claimed" and queue.read(vision["job_id"])["state"] == "claimed"
    argv = dispatcher.unit_argv("v100-1", small["job_id"], 1)
    assert argv[-4:] == ["--slot", "v100-1", "--config", "sma-v100.yaml"] and "--run-unit" in argv
    # The 4-bit thinking-off leg finishes first: its slot takes the vision job's next leg.
    launcher.end(small["job_id"], 1)
    dispatcher.tick()
    assert launcher.started[-1] == (vision["job_id"], 1, "v100-1")
    launcher.end(small["job_id"], 0)
    dispatcher.tick()
    assert launcher.started[-1] == (vision["job_id"], 2, "v100-0")
    launcher.end(small["job_id"], 2)
    dispatcher.tick()
    assert queue.read(small["job_id"])["state"] == "done" and len(launcher.started) == 6  # Nothing left to start.
    state = json.loads((tmp_path / "queue" / "dispatcher" / "v100-pod.json").read_text())
    assert [row["state"] for row in state["slots"]] == ["busy", "busy", "idle", "busy"]


def test_candidates_wait_for_their_references_run_id_and_slots_are_held_for_them(tmp_path: Path) -> None:
    dispatcher, queue, launcher, _ = _dispatcher(tmp_path, created=False)
    small = _pair_job(queue)
    _pair_job(queue, "LiquidAI/LFM3-VL-2B-GGUF")
    dispatcher.recover()
    dispatcher.tick()
    assert launcher.started == [(small["job_id"], 0, "v100-0")]  # The other slots wait for the reference's run id.
    launcher.record_run(small["job_id"], 0)
    dispatcher.tick()
    assert [unit[1:] for unit in launcher.started[1:]] == [(1, "v100-1"), (2, "v100-2"), (0, "v100-3")]


def test_a_crashed_unit_is_resumed_on_whichever_slot_is_free(tmp_path: Path) -> None:
    dispatcher, queue, launcher, _ = _dispatcher(tmp_path)
    jobs = [queue.add([{"repo": f"r{index}", "quant": "gguf", "gguf": "F16", "engine": "llamacpp"}], kind="single") for index in range(5)]
    dispatcher.recover()
    dispatcher.tick()
    launcher.end(jobs[2]["job_id"], 0, status="crash", code=-9)  # Its process died without recording an outcome.
    launcher.end(jobs[0]["job_id"], 0)
    dispatcher.tick()
    # Job 2 is ahead of job 4 in the queue: its leg takes the first free slot, job 4 the next.
    assert launcher.started[4] == (jobs[2]["job_id"], 0, "v100-0") and launcher.started[5] == (jobs[4]["job_id"], 0, "v100-2")
    leg = queue.read(jobs[2]["job_id"])["legs"][0]
    assert leg["attempts"] == 1 and leg["run_id"] == f"{jobs[2]['job_id'][-4:]}-0"  # Resumed by its run id.


def test_two_jobs_interleave_in_queue_order(tmp_path: Path) -> None:
    dispatcher, queue, launcher, _ = _dispatcher(tmp_path)
    first, second = _pair_job(queue), _pair_job(queue, "LiquidAI/LFM3-VL-2B-GGUF")
    third = _pair_job(queue, "LiquidAI/LFM3-Audio-2B-GGUF", "Q4_0")
    dispatcher.recover()
    dispatcher.tick()
    for leg in range(3):
        launcher.end(first["job_id"], leg)
    dispatcher.tick()
    assert [(unit[0], unit[1]) for unit in launcher.started[4:]] == [(second["job_id"], 1), (second["job_id"], 2), (third["job_id"], 0)]


def test_a_restarted_dispatcher_adopts_live_units_and_stops_servers_no_unit_owns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sma import dispatcher as module

    servers = {"sma-livejob-0-quality": "0", "sma-gone-quality": "1"}
    dispatcher, queue, launcher, _ = _dispatcher(tmp_path, servers=servers)
    job = _pair_job(queue)
    dispatcher.recover()
    dispatcher.tick()
    queue.update(job["job_id"], lambda j: j["legs"][0].update(run_id="livejob-0"))
    # A new dispatcher process: leg 0's unit is still alive, legs 1 and 2's are gone.
    monkeypatch.setattr(module, "unit_alive", lambda pid: pid == launcher.handles[(job["job_id"], 0)].pid)
    fresh, _, fresh_launcher, _ = _dispatcher(tmp_path, servers=servers)
    fresh.recover()
    assert set(fresh.children) == {"v100-0"} and fresh.deps.runtime.stopped == ["sma-gone-quality"]
    assert [leg["status"] for leg in queue.read(job["job_id"])["legs"]] == ["running", "pending", "pending"]
    fresh.tick()
    assert [unit[1:] for unit in fresh_launcher.started] == [(1, "v100-1"), (2, "v100-2")]


def test_a_slot_whose_server_fails_to_start_twice_is_marked_down(tmp_path: Path) -> None:
    dispatcher, queue, launcher, _ = _dispatcher(tmp_path)
    cfg = dispatcher.cfg
    cfg.queue.max_attempts = 5
    job = queue.add([{"repo": "r", "quant": "gguf", "gguf": "F16", "engine": "llamacpp"}], kind="single")
    dispatcher.recover()
    for _ in range(2):
        dispatcher.tick()
        slot = queue.read(job["job_id"])["legs"][0]["slot"]
        queue.update(job["job_id"], lambda j: j["legs"][0].update(status="failed", error="llamacpp failed to become healthy."))
        launcher.handles[(job["job_id"], 0)].code = 1
        dispatcher.reap()
        if slot == "v100-0":
            continue
    assert "v100-0" in dispatcher.down
    dispatcher.tick()
    assert launcher.started[-1][2] == "v100-1"


# Step 6b: the memory guard.


def test_the_guard_stops_only_the_largest_run_which_resumes_then_a_second_stop_dead_letters_it(tmp_path: Path) -> None:
    left = {"bytes": 100 * 10**9}
    sizes = {"v100-0": 2, "v100-1": 9, "v100-2": 3, "v100-3": 1}
    dispatcher, queue, launcher, killed = _dispatcher(
        tmp_path, memory=lambda: (left["bytes"], "sma-harness.slice's memory.max"), footprint=lambda child, run_id: sizes[child.slot] * 10**9
    )
    jobs = [queue.add([{"repo": f"r{index}", "quant": "gguf", "gguf": "F16", "engine": "llamacpp"}], kind="single") for index in range(5)]
    dispatcher.recover()
    dispatcher.tick()
    victim = jobs[1]["job_id"]
    left["bytes"] = 3 * 10**9
    dispatcher.guard()
    dispatcher.guard()  # Once per drop under the floor.
    assert killed == [f"{victim[-4:]}-0"] and launcher.handles[(victim, 0)].signals == ["TERM"]
    assert [launcher.handles[(job["job_id"], 0)].signals for job in jobs[:4] if job["job_id"] != victim] == [[], [], []]
    dispatcher.cfg.queue.retry_backoff_s = 3600
    dispatcher.tick()
    leg = queue.read(victim)["legs"][0]
    assert (leg["status"], leg["guard_trips"], leg["attempts"]) == ("pending", 1, 0) and "used the most" in leg["guard_notes"][0]["note"]
    assert launcher.started[-1] == (jobs[4]["job_id"], 0, "v100-1")  # The stopped leg waits out the backoff; the next job runs.
    queue.update(victim, lambda j: j["legs"][0].update(not_before=None))  # The backoff has passed.
    launcher.end(jobs[0]["job_id"], 0)
    left["bytes"] = 100 * 10**9
    dispatcher.guard()
    dispatcher.tick()
    assert launcher.started[-1] == (victim, 0, "v100-0")  # Resumed on the next free slot.
    sizes["v100-0"] = 20
    left["bytes"] = 1 * 10**9
    dispatcher.guard()
    dispatcher.tick()
    final = queue.read(victim)
    assert final["state"] == "failed" and final["legs"][0]["dead_lettered"] and final["legs"][0]["guard_trips"] == 2
    letter = json.loads((tmp_path / "queue" / "failed" / f"{victim}-leg0.json").read_text())
    assert len(letter["guard_notes"]) == 2 and "twice" in letter["reason"]


def test_the_harness_slices_cgroup_folder_follows_systemds_dashes() -> None:
    assert cgroup_dir("sma-harness.slice") == Path("/sys/fs/cgroup/sma.slice/sma-harness.slice")
    assert cgroup_dir("kernelbench.slice") == Path("/sys/fs/cgroup/kernelbench.slice")


# Step 12: queue drop.


def test_drop_takes_a_started_job_off_the_queue_and_refuses_one_a_live_worker_holds(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path / "queue")
    job = queue.add([{"repo": "LiquidAI/LFM3-1.7B", "quant": "bf16"}], kind="single")
    assert queue.claim("w", "spark") is not None
    queue.update(job["job_id"], lambda j: j["legs"][0].update(run_id="run-1", status="running"))
    with pytest.raises(SmaError, match="Stop that worker"):
        queue.drop(job["job_id"], stale_s=3600)
    dropped = queue.drop(job["job_id"], stale_s=0)
    assert dropped["removed_from"] == "claimed" and (tmp_path / "queue" / "removed" / f"{job['job_id']}.json").is_file()
    assert queue.read(job["job_id"]) is None
    with pytest.raises(SmaError, match="No queue job"):
        queue.drop(job["job_id"], stale_s=0)


# Turns: the speed steps (FifoLock).


def test_fifo_turns_go_in_arrival_order_and_skip_dead_waiters(tmp_path: Path) -> None:
    from sma.locks import FifoLock

    path = tmp_path / "speed.lock"
    order: list[str] = []
    first = FifoLock(path, poll_s=0.01)
    first.acquire()
    # A waiter whose process died left a ticket ahead of the others.
    (path.with_name("speed.lock.d") / f"{time.time_ns():020d}-{2**22 + 12345}").write_text("")

    def take(name: str) -> None:
        with FifoLock(path, poll_s=0.01):
            order.append(name)
            time.sleep(0.05)

    threads = []
    for name in ("b", "c"):
        threads.append(threading.Thread(target=take, args=(name,)))
        threads[-1].start()
        time.sleep(0.05)
    time.sleep(0.1)
    assert order == []  # Both wait while the first holds its turn.
    first.release()
    for thread in threads:
        thread.join(5)
    assert order == ["b", "c"] and list(path.with_name("speed.lock.d").iterdir()) == []


# BrowseComp-Plus: per-slot ports.


def test_two_slots_runs_get_their_own_browsecomp_ports_and_tags(tmp_path: Path) -> None:
    from sma.agentic import search_command

    cfg = _cfg(tmp_path)
    one, two = slot_config(cfg, "v100-0"), slot_config(cfg, "v100-1")
    first, second = search_command(one, "20261101T000000Z-aaaa"), search_command(two, "20261101T000001Z-bbbb")
    assert "--port 8791" in first and "--port 8792" in second
    assert "--tag runs/20261101T000000Z-aaaa" in first and "--tag runs/20261101T000001Z-bbbb" in second
    assert {one.browsecomp.local_port, two.browsecomp.local_port} == {8791, 8792}


def test_a_dispatchers_child_does_not_guard_memory_itself(tmp_path: Path) -> None:
    # Each run guarding the shared harness memory alone would stop all four at once.
    cfg = slot_config(_cfg(tmp_path), "v100-0")
    assert cfg.remote.slot == "v100-0" and cfg.memory.guard_free_gb == 4
    assert os.path.basename(str(cfg.lock_file())) == "v100-v100-0.lock"


# Step 7: KernelBench's FIFO, watched from SMA.


class QueuedService:
    """A KernelBench service whose queue is full once, then holds the job behind another before running it."""

    def __init__(self, started_at: str = "2026-11-01T10:00:00Z") -> None:
        self.submits = 0
        self.states = [
            {"status": "queued", "created_at": "2026-11-01T09:00:00Z"},
            {"status": "queued", "created_at": "2026-11-01T09:00:00Z"},
            {"status": "running", "created_at": "2026-11-01T09:00:00Z", "started_at": started_at},
            {"status": "done", "created_at": "2026-11-01T09:00:00Z", "started_at": started_at, "results": []},
        ]

    def submit(self, **kwargs: Any) -> dict[str, Any]:
        from sma.kbservice import QueueFull

        self.submits += 1
        if self.submits == 1:
            raise QueueFull("http://kb/v1/jobs returned 429: 50 jobs are already queued")
        return {"job_id": "kb-mine", "position": 2}

    def job(self, job_id: str) -> dict[str, Any]:
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def jobs(self) -> list[dict[str, Any]]:
        return [
            {"id": "kb-other", "status": "running", "created_at": "2026-11-01T08:00:00Z"},
            {"id": "kb-older", "status": "queued", "created_at": "2026-11-01T08:30:00Z"},
            {"id": "kb-mine", "status": "queued", "created_at": "2026-11-01T09:00:00Z"},
            {"id": "kb-newer", "status": "queued", "created_at": "2026-11-01T09:30:00Z"},
        ]


def _kb_ctx(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg.kernelbench.poll_s = 0
    return SimpleNamespace(cfg=cfg, run_id="20261101T000000Z-aaaa")


def test_kernelbench_waits_out_a_full_queue_and_times_out_only_from_when_its_job_started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sma import agentic

    ctx, service = _kb_ctx(tmp_path), QueuedService(started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    reply = agentic._submit(ctx, service, [{"problem_id": 1}])
    assert (service.submits, reply["job_id"], reply["position"]) == (2, "kb-mine", 2) and reply["submitted_at"]
    state = agentic._wait_for_job(ctx, service, {"job_id": "kb-mine", "position": 2, "submitted_at": "x"}, tmp_path)
    queue = state["queue"]
    # Hours queued behind other runs did not count against wait_s; the queued time and the most jobs ahead are kept.
    assert state["status"] == "done" and queue["max_position"] == 2 and queue["started_at"] and queue["queued_s"] > 3600
    assert json.loads((tmp_path / "queue.json").read_text())["position_at_submit"] == 2

    late = QueuedService(started_at="2026-11-01T00:00:00Z")
    late.states = late.states[2:3]
    ctx.cfg.kernelbench.wait_s = 60
    with pytest.raises(SmaError, match="within 60 s of starting"):
        agentic._wait_for_job(ctx, late, {"job_id": "kb-mine"}, tmp_path)


def test_a_jobs_position_counts_the_running_job_and_older_queued_ones() -> None:
    from sma.kbservice import queue_position

    jobs = QueuedService().jobs()
    assert queue_position(jobs, "kb-mine") == 2 and queue_position(jobs, "kb-other") is None and queue_position(jobs, "kb-gone") is None


# Step 8: LiveCodeBench's grading lock.


def test_two_livecodebench_containers_grade_one_at_a_time_on_the_timed_cpus(tmp_path: Path) -> None:
    import subprocess
    import sys

    from sma.commands import eval_factory_patch

    runner = tmp_path / "livecodebench" / "runner"
    runner.mkdir(parents=True)
    (tmp_path / "livecodebench" / "__init__.py").write_text("")
    (runner / "__init__.py").write_text("")
    (runner / "scenario_router.py").write_text(
        "import os, time\n"
        "def get_metrics(scenario, args, benchmark, results):\n"
        "    start = time.time(); time.sleep(0.4)\n"
        "    return f'{start} {time.time()} {sorted(os.sched_getaffinity(0))}'\n"
    )
    script = (
        "import argparse; from livecodebench.runner.scenario_router import get_metrics\n"
        "print(get_metrics('codegeneration', argparse.Namespace(timeout=14400, evaluation_timeout=6), [], []))\n"
    )
    cpu = min(os.sched_getaffinity(0))
    env = {"PYTHONPATH": f"{eval_factory_patch('nvcr.io/nvidia/eval-factory/livecodebench:26.03')}:{tmp_path}", "PATH": "/usr/bin:/bin",
           "SMA_GRADING_LOCK": str(tmp_path / "grading.lock"), "SMA_GRADING_CPUS": str(cpu)}  # fmt: skip
    procs = [subprocess.Popen([sys.executable, "-c", script], cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outs = [proc.communicate(timeout=60) for proc in procs]
    assert all(proc.returncode == 0 for proc in procs), outs
    spans = sorted((float(out.split()[0]), float(out.split()[1])) for out, _err in outs)
    assert spans[0][1] <= spans[1][0]  # The second container graded after the first finished.
    assert all(out.split(" ", 2)[2].strip() == f"[{cpu}]" for out, _err in outs)
    assert all("grading lock" in err and f"grading on CPUs {cpu}" in err for _out, err in outs)


def test_suite_containers_run_in_the_harness_slice_and_livecodebench_grades_on_the_timed_cpus(tmp_path: Path) -> None:
    from sma.pipeline import RunSpec, execute
    from tests.harness import make_config, make_services
    from tests.test_tool_suites import BFCL, LIVECODEBENCH, _run_dir

    cfg = make_config(tmp_path)
    cfg.quality.suites = [LIVECODEBENCH, BFCL]
    cfg.harness.cgroup_parent, cfg.harness.cpus_untimed, cfg.harness.cpus_timed = "sma-harness.slice", "0-4", "5-9"
    cfg.harness.lock_dir = str(tmp_path / "locks")
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    execute(RunSpec(repo="example/tiny", quant_flag="bf16", tool_parser="hermes"), cfg, services)
    lcb = json.loads((_run_dir(tmp_path) / "quality" / "livecodebench" / "command.json").read_text())
    bfcl = json.loads((_run_dir(tmp_path) / "quality" / "bfcl_v3" / "command.json").read_text())
    argv = lcb["argv"]
    assert argv[argv.index("--cgroup-parent") + 1] == "sma-harness.slice" and argv[argv.index("--cpuset-cpus") + 1] == "0-4,5-9"
    assert "SMA_GRADING_CPUS=5-9" in argv and "SMA_GRADING_LOCK=/sma-locks/livecodebench-grading.lock" in argv
    assert (lcb["cpus"], lcb["grading_cpus"], lcb["grading_lock"]) == ("0-4,5-9", "5-9", "livecodebench-grading.lock")
    assert bfcl["argv"][bfcl["argv"].index("--cpuset-cpus") + 1] == "0-4" and "SMA_GRADING_LOCK" not in " ".join(bfcl["argv"])
    assert bfcl["grading_cpus"] is None


def test_the_speed_client_runs_on_the_timed_cpus(tmp_path: Path) -> None:
    from sma.backends import BenchMounts, get_backend
    from sma.commands import bench_cells

    cfg = _cfg(tmp_path, harness={"cgroup_parent": "sma-harness.slice", "cpus_timed": "5-9", "lock_dir": str(tmp_path)})
    mounts = BenchMounts(model_dir=tmp_path, run_dir=tmp_path, image="vllm", uid=1, gid=1, tokenizer="/model", tokenizer_dir=None)
    argv = get_backend("llamacpp").bench_argv(cfg, container="sma-r-speed", served_name="m", cell=bench_cells(cfg)[0], result_dir="/x", mounts=mounts)
    assert argv[argv.index("--cpuset-cpus") + 1] == "5-9" and argv[argv.index("--cgroup-parent") + 1] == "sma-harness.slice"


def test_a_remote_runs_containers_here_never_take_this_gpu(tmp_path: Path) -> None:
    from sma.pipeline import _refuse_local_gpu

    ctx = SimpleNamespace(cfg=_cfg(tmp_path))
    _refuse_local_gpu(ctx, ["docker", "run", "--rm", "image"])
    with pytest.raises(SmaError, match="must not use this machine's GPU"):
        _refuse_local_gpu(ctx, ["docker", "run", "--gpus", "all", "image"])


# Step 9: the speed steps take turns.


def test_two_runs_reaching_speed_together_take_turns(tmp_path: Path) -> None:
    from sma.pipeline import Ctx, _speed_turn

    cfg = slot_config(_cfg(tmp_path), "v100-0")
    spans: list[tuple[str, float, float]] = []

    def speed(name: str) -> None:
        run_dir = tmp_path / name
        (run_dir / "steps").mkdir(parents=True)
        ctx = Ctx(cfg=cfg, services=None, run_dir=run_dir, manifest={"run_id": name})
        with _speed_turn(ctx):
            start = time.time()
            time.sleep(0.3)
            spans.append((name, start, time.time()))
        assert ctx.speed_turn and ctx.speed_turn["lock"] == "speed-v100.lock"

    threads = [threading.Thread(target=speed, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
        time.sleep(0.05)
    for thread in threads:
        thread.join(10)
    spans.sort(key=lambda row: row[1])
    assert [row[0] for row in spans] == ["a", "b"] and spans[0][2] <= spans[1][1]


def test_a_run_on_a_slot_refuses_a_server_another_run_left_on_its_gpu(tmp_path: Path) -> None:
    from sma.pipeline import Ctx, _guard_containers

    cfg = slot_config(_cfg(tmp_path), "v100-1")
    stopped: list[str] = []
    runtime = SimpleNamespace(
        list_sma=lambda: ["sma-me-quality", "sma-other-quality", "sma-x-quality-bfcl_v3"],
        servers_on=lambda: {"sma-me-quality": "1", "sma-other-quality": "0"},
        stop=stopped.append,
    )
    ctx = Ctx(cfg=cfg, services=SimpleNamespace(runtime=runtime), run_dir=tmp_path, manifest={"run_id": "me"})
    _guard_containers(ctx)  # Another slot's server and container are expected; its own leftover stops.
    assert stopped == ["sma-me-quality"]
    runtime.servers_on = lambda: {"sma-left-quality": "1"}
    with pytest.raises(SmaError, match="still running on GPU 1"):
        _guard_containers(ctx)


def test_each_slot_has_its_own_ssh_connection_to_the_serving_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # sshd allows 10 sessions per connection; four runs' log followers and telemetry on one would hit it.
    from sma.remote import remote_runner

    for name, value in {"V100_HOST": "pod", "V100_HOST_USER": "sma", "V100_HOST_SSH_PORT": "2222", "V100_HOST_SSH_KEY": "/k"}.items():
        monkeypatch.setenv(name, value)
    cfg = _cfg(tmp_path)
    dirs = {remote_runner(slot_config(cfg, name)).control_dir.name for name in ("v100-0", "v100-1")}
    assert dirs == {"v100-0", "v100-1"} and remote_runner(cfg).control_dir.name == "shared"


def test_a_unit_process_starts_detached_on_its_cpus_with_its_output_in_its_log(tmp_path: Path) -> None:
    import sys

    from sma.proc import start_detached

    cpu = min(os.sched_getaffinity(0))
    log = tmp_path / "logs" / "v100-0.log"
    proc = start_detached([sys.executable, "-c", "import os; print(sorted(os.sched_getaffinity(0)), os.getsid(0) == os.getpid())"], log, cpus={cpu}, what="test unit")
    assert proc.wait(30) == 0
    assert log.read_text().strip() == f"[{cpu}] True"  # Its own session: a stop reaches it only through the dispatcher.


def test_a_claimed_job_whose_legs_all_ended_is_closed_by_the_next_tick(tmp_path: Path) -> None:
    dispatcher, queue, _launcher, _ = _dispatcher(tmp_path)
    job = queue.add([{"repo": "r", "quant": "gguf", "gguf": "F16", "engine": "llamacpp"}], kind="single")
    queue.claim_job(job["job_id"], dispatcher.identity, "v100-pod")
    queue.update(job["job_id"], lambda j: j["legs"][0].update(status="done", run_id="run-1"))
    dispatcher.tick()
    assert queue.read(job["job_id"])["state"] == "done"


def test_the_speed_turn_waits_for_livecodebench_grading_and_holds_its_lock(tmp_path: Path) -> None:
    # Both are timed on harness.cpus_timed; grading on slower cores failed 2 of 105 tests at the 6 s limit.
    import fcntl

    from sma.commands import LIVECODEBENCH_GRADING_LOCK
    from sma.pipeline import Ctx, _speed_turn

    cfg = slot_config(_cfg(tmp_path), "v100-0")
    lock = cfg.lock_dir() / LIVECODEBENCH_GRADING_LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    held = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX)  # A container is grading.
    (tmp_path / "r" / "steps").mkdir(parents=True)
    ctx = Ctx(cfg=cfg, services=None, run_dir=tmp_path / "r", manifest={"run_id": "r"})
    inside: list[bool] = []

    def speed() -> None:
        with _speed_turn(ctx):
            probe = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                inside.append(False)
            except BlockingIOError:
                inside.append(True)  # Grading cannot start while speed measures.
            finally:
                os.close(probe)

    thread = threading.Thread(target=speed)
    thread.start()
    time.sleep(0.3)
    assert inside == [] and thread.is_alive()  # Speed waits for the grading to end.
    fcntl.flock(held, fcntl.LOCK_UN)
    os.close(held)
    thread.join(10)
    assert inside == [True] and ctx.speed_turn["grading_waited_s"] >= 0.2
