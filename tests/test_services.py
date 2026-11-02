"""The benchmark services in services/: the KernelBench evaluation service's request checks and how
KernelBench's outcomes become its results, its CPU pinning and host setup, and the pins of both
services. Offline: no torch, GPU, Docker, network or root."""
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "services" / "kernelbench"
SWE = ROOT / "services" / "swe-runner"
HOST_SETUP = KB / "host-setup.sh"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    level1 = tmp_path / "KernelBench" / "level1"
    level1.mkdir(parents=True)
    for name in ("1_Square_matrix_multiplication_.py", "19_ReLU.py", "README.md"):
        (level1 / name).write_text("import torch\n")
    monkeypatch.setenv("KERNELBENCH_DIR", str(tmp_path))
    mod = _load("kb_server", KB / "server.py")
    mod.STATE = tmp_path / "state"
    return mod


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(KB))       # worker.py imports sandbox.py from its folder
    return _load("kb_worker", KB / "worker.py")


def _result(server: ModuleType, raw, stopped=None) -> dict:
    return server.kernel_result(raw, "LOG TAIL", stopped, 300)


def test_problems_come_from_the_dataset_folder(server: ModuleType) -> None:
    assert server.problems(1) == {1: "1_Square_matrix_multiplication_", 19: "19_ReLU"}


def test_a_valid_job_gets_kernelbench_defaults_and_hides_its_sources(server: ModuleType) -> None:
    job = server.eval_job({"level": 1, "kernels": [{"problem_id": 19, "kernel_src": "x = 1", "label": "m/q4"}]})

    assert job["options"]["num_correct_trials"] == 5
    assert job["options"]["num_perf_trials"] == 100
    assert job["options"]["backend"] == "cuda"
    assert job["progress"] == {"done": 0, "total": 1}
    assert "kernels" not in server.public(job)
    assert "kernels" in server.public(job, sources=True)


@pytest.mark.parametrize("body", [
    {"level": 4, "kernels": [{"problem_id": 1, "kernel_src": "x"}]},                 # level 4: not offered
    {"level": 1, "kernels": []},
    {"level": 1, "kernels": [{"problem_id": 2, "kernel_src": "x"}]},                 # no such problem
    {"level": 1, "kernels": [{"problem_id": 1, "kernel_src": " "}]},
    {"level": 1, "kernels": [{"problem_id": 1, "kernel_src": "x", "shell": "rm"}]},   # unknown key
    {"level": 1, "kernels": [{"problem_id": 1, "kernel_src": "x"}], "options": {"backend": "hip"}},
    {"level": 1, "kernels": [{"problem_id": 1, "kernel_src": "x"}], "options": {"timeout_s": 5}},
    {"level": 1, "kernels": [{"problem_id": 1, "kernel_src": "x"}], "options": {"num_perf_trials": True}},
    {"level": 1, "kernels": [{"problem_id": 1, "kernel_src": "x" * (513 << 10)}]},
])
def test_a_bad_job_is_refused(server: ModuleType, body: dict) -> None:
    with pytest.raises(server.BadRequest):
        server.eval_job(body)


def test_a_baseline_covers_every_problem_of_its_levels_and_takes_only_timing_options(server: ModuleType) -> None:
    job = server.baseline_job({"levels": [1], "options": {"precision": "fp32"}})

    assert [w["problem_id"] for w in job["work"]] == [1, 19]
    with pytest.raises(server.BadRequest):
        server.baseline_job({"levels": [1], "options": {"backend": "cuda"}})


def test_a_correct_kernel_gets_its_speedup(server: ModuleType) -> None:
    ok = _result(server, {"problem_name": "19_ReLU", "compiled": True, "correctness": True, "runtime": 0.5,
                          "ref_runtime": 1.0, "runtime_stats": {"mean": 0.5}, "ref_runtime_stats": {"mean": 1.0},
                          "metadata": {"hardware": "NVIDIA GB10", "excessive_speedup": False}})

    assert (ok["correct"], ok["speedup"], ok["error_kind"], ok["error"]) == (True, 2.0, None, None)
    assert (ok["runtime_ms"], ok["ref_runtime_ms"]) == (0.5, 1.0)


@pytest.mark.parametrize("kind, raw", [
    ("compile", {"compiled": False, "correctness": False, "metadata": {"compilation_error": "nvcc: error"}}),
    ("runtime", {"compiled": True, "correctness": False, "metadata": {"runtime_error": "illegal memory access"}}),
    ("incorrect", {"compiled": True, "correctness": False,
                   "metadata": {"correctness_issue": "Output mismatch", "max_difference": ["0.5"]}}),
    ("timing", {"compiled": True, "correctness": True, "runtime": -1.0, "metadata": {"error_during_performance": "boom"}}),
])
def test_each_failure_names_its_kind_and_has_no_speedup(server: ModuleType, kind: str, raw: dict) -> None:
    r = _result(server, raw)

    assert r["error_kind"] == kind
    assert r["error"]
    assert r["speedup"] is None


def test_a_stopped_or_dead_worker_is_a_timeout_runtime_or_crash(server: ModuleType) -> None:
    server.MEMORY_BUDGET = 14 << 30
    over = _result(server, None, stopped="memory")

    assert _result(server, None, stopped="timeout")["error_kind"] == "timeout"
    assert over["error_kind"] == "runtime"
    assert "host-memory budget of 14.0 GiB" in over["error"]
    assert _result(server, None)["error_kind"] == "crash"
    assert "LOG TAIL" in _result(server, {"worker_error": "OSError: Landlock is not available"})["error"]


def test_an_excessive_speedup_is_flagged(server: ModuleType) -> None:
    r = _result(server, {"compiled": True, "correctness": True, "runtime": 0.01, "ref_runtime": 1.0,
                         "metadata": {"excessive_speedup": True}})

    assert r["excessive_speedup"] is True


def test_a_kernel_the_service_died_on_twice_is_not_run_again(server: ModuleType) -> None:
    # A 17 GB problem input once took a container over its memory limit, which killed the server
    # too, and every restart resumed on the same problem. One retry, then it is recorded and skipped.
    job = server.baseline_job({"levels": [1]})
    job["status"] = "running"
    server.jobs[job["id"]] = job
    for restarts in (0, 1):
        assert server.start_item(job, 1)
        assert job["in_flight"] == {"index": 1, "restarts": restarts}
        server.jobs.clear()
        server.load_jobs()                          # the service was killed while item 1 ran
        job = server.jobs[job["id"]]
        assert job["status"] == "queued"
        job["status"] = "running"
        server.save(job)

    assert not server.start_item(job, 1)
    assert server.start_item(job, 0)                # another item starts afresh
    assert "in_flight" not in server.public(job)


def test_the_memory_budget_comes_from_the_container_limit(server: ModuleType, tmp_path: Path) -> None:
    cg = tmp_path / "cgroup"
    cg.mkdir()
    server.CGROUP = cg
    (cg / "memory.max").write_text("17179869184\n")
    assert server.memory_budget() == 14 << 30
    (cg / "memory.max").write_text("max\n")
    assert server.memory_budget() is None
    (cg / "memory.stat").write_text("anon 1000\nfile 999999\nkernel 5\nshmem 20\n")
    assert server.memory_in_use() == 1020           # page cache is reclaimable: not counted


def test_unified_memory_stops_a_worker_when_the_machine_runs_low(server: ModuleType, tmp_path: Path) -> None:
    # On a GB10 the GPU's allocations come from the machine's RAM, outside the container's limit.
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       126000000 kB\nMemAvailable:    7340032 kB\n")
    server.MEMINFO = meminfo
    assert server.host_memory_available() == 7 << 30
    server.MEMINFO = tmp_path / "missing"
    assert server.host_memory_available() is None

    server.HOST_MEMORY_FLOOR = 8 << 30
    r = _result(server, None, stopped="host_memory")
    assert r["error_kind"] == "runtime"
    assert "available memory fell below 8.0 GiB" in r["error"]


def test_a_long_error_keeps_both_ends(server: ModuleType) -> None:
    out = server.clip("HEAD" + "x" * 50000 + "TAIL")

    assert len(out) <= server.MAX_ERROR + 100
    assert out.startswith("HEAD") and out.endswith("TAIL")


def test_the_fingerprint_follows_what_speedups_depend_on(server: ModuleType) -> None:
    env = {"torch": "2.9.1+cu130", "cuda_runtime": "13.0", "nvcc": "V13.0.88", "kernelbench_commit": "423217d",
           "clock_lock_mhz": "2200", "gpu": {"name": "NVIDIA GB10", "driver_version": "580.95.05"},
           "clocks_under_load": {"sm_mhz": "2184"}}
    fp = server.fingerprint(env)

    assert fp == server.fingerprint(env | {"clocks_under_load": {"sm_mhz": "2190"}})   # a sample, not a setting
    assert fp != server.fingerprint(env | {"clock_lock_mhz": None})
    assert fp != server.fingerprint(env | {"torch": "2.10.0"})
    assert server.fingerprint({"error": "probe failed"}) is None


def test_an_unpinned_environment_keeps_the_fingerprint_its_baselines_are_stored_under(server: ModuleType) -> None:
    # The old Spark's baselines were recorded before CPU pinning. Their stored fingerprints must
    # still come out the same, or the service would no longer find them.
    recorded = sorted((ROOT / "runs" / "gb10" / "baseline" / "kernelbench").glob("level*.json"))
    assert recorded
    for f in recorded:
        env = json.loads(f.read_text())["environment"]
        assert server.fingerprint(env) == env["fingerprint"], f.name
        if not (env.get("cpus") or {}).get("timing"):  # The harness GB10's baseline (2026-11-01) is pinned.
            assert server.fingerprint(env | {"cpus": {"timing": None, "compile": None, "allowed": "0-19"}}) == env["fingerprint"]
    assert any((json.loads(f.read_text())["environment"].get("cpus") or {}).get("timing") for f in recorded)


def test_pinned_timing_cpus_give_the_harness_gb10_its_own_fingerprint(server: ModuleType) -> None:
    # Same GPU, driver, CUDA, PyTorch and clock lock as the old Spark: only the timing CPUs differ.
    env = json.loads((ROOT / "runs/gb10/baseline/kernelbench/level1-fp32-cuda_event-100.2200mhz.json").read_text())["environment"]
    pinned = env | {"cpus": {"timing": "15-19", "compile": "10-14", "allowed": "15-19"}}

    assert server.fingerprint(pinned) != env["fingerprint"]
    assert server.fingerprint(pinned) != server.fingerprint(env | {"cpus": {"timing": "5-9", "compile": "10-14"}})
    assert server.fingerprint(pinned) == server.fingerprint(env | {"cpus": {"timing": "15-19", "compile": "0-4"}})


def test_the_probe_reports_the_memory_the_host_keeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {"torch": "2.9.1+cu130", "gpu": {"name": "NVIDIA GB10", "driver_version": "580.173.02", "compute_mode": "Exclusive_Process"},
           "cpus": {"timing": "15-19", "compile": "10-14", "allowed": "15-19"}, "worker_seconds": 3.0}
    monkeypatch.setenv("KERNELBENCH_DIR", str(tmp_path))
    monkeypatch.delenv("KERNELBENCH_MEMORY_RESERVED_GIB", raising=False)
    unset = _load("kb_server_unset", KB / "server.py")
    monkeypatch.setattr(unset, "run_worker", lambda req, timeout_s: (dict(raw), "", None))
    monkeypatch.setenv("KERNELBENCH_MEMORY_RESERVED_GIB", "80")
    reserved = _load("kb_server_reserved", KB / "server.py")
    monkeypatch.setattr(reserved, "run_worker", lambda req, timeout_s: (dict(raw), "", None))

    assert unset.probe()["memory_reserved_gib"] is None
    env = reserved.probe()
    assert env["memory_reserved_gib"] == 80.0
    assert env["fingerprint"] == reserved.fingerprint(raw)
    assert "worker_seconds" not in env


# ---- CPU pinning (worker.py)

def test_a_cpu_list_reads_and_writes_the_linux_form(worker: ModuleType) -> None:
    assert worker.cpu_list("10-14,16") == {10, 11, 12, 13, 14, 16}
    assert worker.cpu_list(" 3 , 1-2") == {1, 2, 3}
    assert worker.cpu_list(None) == worker.cpu_list("") == set()
    assert worker.cpu_list_text({16, 12, 10, 11, 0}) == "0,10-12,16"
    assert worker.cpu_list_text(worker.cpu_list("15-19")) == "15-19"
    assert worker.cpu_list_text(set()) == ""


@pytest.mark.parametrize("text", ["14-10", "a", "5-", "-5", "1;2", "1-2-3"])
def test_a_bad_cpu_list_is_refused(worker: ModuleType, text: str) -> None:
    with pytest.raises(ValueError):
        worker.cpu_list(text)


def test_a_child_is_pinned_before_the_callers_own_preexec_runs(worker: ModuleType) -> None:
    calls = []
    preexec = worker.pin_first({10, 11}, lambda: calls.append("caller"),
                               setaffinity=lambda pid, cpus: calls.append(("pin", pid, cpus)))
    preexec()
    worker.pin_first({12}, setaffinity=lambda pid, cpus: calls.append(("pin", pid, cpus)))()

    assert calls == [("pin", 0, {10, 11}), "caller", ("pin", 0, {12})]


def test_a_pinned_popen_starts_its_child_on_the_given_cpu(worker: ModuleType) -> None:
    mine = os.sched_getaffinity(0)
    cpu = min(mine)
    popen = worker.pinned_popen(subprocess.Popen, {cpu})
    with popen([sys.executable, "-I", "-c", "import os; print(sorted(os.sched_getaffinity(0)))"],
               stdout=subprocess.PIPE, text=True) as p:
        out, _ = p.communicate(timeout=60)

    assert out.strip() == str([cpu])
    assert isinstance(p, subprocess.Popen)
    assert os.sched_getaffinity(0) == mine          # only the child moved


def _cpuset(monkeypatch: pytest.MonkeyPatch, worker: ModuleType, allowed: set[int]) -> dict:
    """A container whose cpuset is `allowed`: the kernel drops other CPUs, or refuses when none is left."""
    state = {"now": set(allowed)}

    def setaffinity(pid, cpus):
        if not set(cpus) & allowed:
            raise OSError(22, "Invalid argument")
        state["now"] = set(cpus) & allowed
    monkeypatch.setattr(worker.os, "sched_setaffinity", setaffinity)
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda pid: set(state["now"]))
    monkeypatch.setattr(worker.subprocess, "Popen", worker.subprocess.Popen)     # restored after the test
    return state


def test_the_worker_times_on_the_timing_cpus_and_compiles_on_the_compile_cpus(
        worker: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    original = worker.subprocess.Popen
    state = _cpuset(monkeypatch, worker, set(range(10, 20)))
    monkeypatch.setenv("KERNELBENCH_TIMING_CPUS", "15-19")
    monkeypatch.setenv("KERNELBENCH_COMPILE_CPUS", "10-14")

    assert worker.pin_cpus() == {"timing": "15-19", "compile": "10-14"}
    assert state["now"] == {15, 16, 17, 18, 19}
    assert issubclass(worker.subprocess.Popen, original) and worker.subprocess.Popen is not original


def test_unset_cpu_variables_leave_the_worker_as_it_was(worker: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    original = worker.subprocess.Popen
    state = _cpuset(monkeypatch, worker, set(range(10, 20)))
    monkeypatch.delenv("KERNELBENCH_TIMING_CPUS", raising=False)
    monkeypatch.delenv("KERNELBENCH_COMPILE_CPUS", raising=False)

    assert worker.pin_cpus() == {"timing": None, "compile": None}
    assert state["now"] == set(range(10, 20))
    assert worker.subprocess.Popen is original


@pytest.mark.parametrize("name, cpus", [("KERNELBENCH_TIMING_CPUS", "15-21"), ("KERNELBENCH_COMPILE_CPUS", "8-14"),
                                        ("KERNELBENCH_TIMING_CPUS", "0-4")])
def test_cpus_outside_the_container_are_refused_not_narrowed(
        worker: ModuleType, monkeypatch: pytest.MonkeyPatch, name: str, cpus: str) -> None:
    # The kernel would quietly time on fewer CPUs than the environment names.
    _cpuset(monkeypatch, worker, set(range(10, 20)))
    monkeypatch.delenv("KERNELBENCH_TIMING_CPUS", raising=False)
    monkeypatch.delenv("KERNELBENCH_COMPILE_CPUS", raising=False)
    monkeypatch.setenv(name, cpus)

    with pytest.raises(OSError, match="--cpuset-cpus"):
        worker.pin_cpus()


def test_a_compile_error_that_mentions_lock_is_a_compile_error(worker: ModuleType) -> None:
    # KernelBench returns None for any build error containing "lock" ("block", blockIdx) and only
    # prints it; the worker reads the error back.
    printed = ("[Eval] Start\nFailed to compile custom CUDA kernel: Record as compilation failure. \n"
               "Error: Error building extension 'k':\ncuda.cu(9): error: identifier \"blockIdy\" is undefined\n"
               "[Eval] Lock file error during compilation, Please retry. Error: Error building extension 'k'\n")

    assert worker.printed_compile_error(printed) == (
        "Error building extension 'k':\ncuda.cu(9): error: identifier \"blockIdy\" is undefined")


def test_a_baseline_without_a_time_reports_what_kernelbench_printed(worker: ModuleType) -> None:
    printed = ("Using PyTorch Eager Execution on 19_ReLU.py\n[Eval] Error in Measuring Performance: "
               "CUDA out of memory. Tried to allocate 6.00 GiB.\n")

    assert worker.printed_timing_error(printed) == "CUDA out of memory. Tried to allocate 6.00 GiB."
    assert worker.printed_timing_error("nothing") == "KernelBench recorded no time"


def test_the_worker_sandboxes_itself_before_it_runs_anything() -> None:
    src = (KB / "worker.py").read_text()
    main = src[src.index("def main"):]

    assert main.index("sandbox.restrict") < main.index('"eval":')
    assert main.index("pin_cpus()") < main.index('"eval":')      # pinned before torch starts its threads
    assert re.search(r"^import torch", src, re.M) is None          # torch is imported only after the sandbox


# ---- Host setup (host-setup.sh). Nothing here applies it: that needs root.

def _host_setup(*args: str, **env: str) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items()
            if not k.startswith(("KB_", "HARNESS_", "SYSTEM_MEMORY", "USER_MEMORY", "SMA_", "INSTALL_DISPATCHER"))
            and k not in ("GPU", "CLOCK_MHZ")}
    full.update({"SMA_USER": "tester", "SMA_REPO": "/srv/sma", **env})
    return subprocess.run(["bash", str(HOST_SETUP), *args], capture_output=True, text=True, env=full, timeout=120)


def _printed_files(out: str) -> dict[str, str]:
    parts = re.split(r"^==> (.+?) <==\n", out, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


def test_host_setup_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(HOST_SETUP)], capture_output=True, timeout=60).returncode == 0


def test_host_setup_gives_kernelbench_its_cpus_and_caps_everything_else() -> None:
    run = _host_setup("--print")
    files = _printed_files(run.stdout)
    unit = "/etc/systemd/system/"

    assert run.returncode == 0, run.stderr
    assert "AllowedCPUs=10-19\nMemoryMin=80G\n" in files[unit + "kernelbench.slice"]
    assert "AllowedCPUs=0-9\nMemoryHigh=30G\nMemoryMax=32G\n" in files[unit + "sma-harness.slice"]
    assert "[Slice]\nAllowedCPUs=0-9\nMemoryMax=3G\n" in files[unit + "system.slice.d/50-kernelbench-isolation.conf"]
    assert "[Slice]\nAllowedCPUs=0-9\nMemoryMax=3G\n" in files[unit + "user.slice.d/50-kernelbench-isolation.conf"]
    assert "[Scope]\nAllowedCPUs=0-9\n" in files[unit + "init.scope.d/50-kernelbench-isolation.conf"]
    clocks = files[unit + "kernelbench-gpu-clocks.service"]
    assert "Environment=GPU=0 CLOCK_MHZ=2200\n" in clocks
    assert "ExecStart=/usr/bin/nvidia-smi -i ${GPU} -c EXCLUSIVE_PROCESS\n" in clocks
    assert "ExecStop=/usr/bin/nvidia-smi -i ${GPU} -c DEFAULT\n" in clocks
    assert "ExecStart=/usr/local/sbin/kernelbench-host-setup --irqs" in files[unit + "kernelbench-irq-affinity.service"]
    assert files["/proc/irq/default_smp_affinity"].strip() == "3ff"
    assert "IRQBALANCE_BANNED_CPULIST=10-19" in "".join(v for k, v in files.items() if k.startswith("/etc/default/irqbalance"))


def test_host_setup_installs_the_dispatcher_with_its_values_filled_in() -> None:
    dispatcher = _printed_files(_host_setup("--print").stdout)["/etc/systemd/system/sma-dispatcher.service"]

    assert "User=tester\n" in dispatcher
    assert "WorkingDirectory=/srv/sma\n" in dispatcher
    assert "ExecStart=/srv/sma/.venv/bin/sma worker --slots --config .sma/v100/sma-v100.yaml\n" in dispatcher
    assert "Slice=sma-harness.slice\n" in dispatcher
    assert "KillMode=mixed\n" in dispatcher and "TimeoutStopSec=300\n" in dispatcher
    assert "Restart=on-failure\n" in dispatcher and "Environment=PYTHONUNBUFFERED=1\n" in dispatcher
    assert re.search(r"@[A-Z_]+@", dispatcher) is None
    assert "@SMA_USER@" in (KB / "sma-dispatcher.service").read_text()


def test_host_setup_parameters_reach_the_files() -> None:
    files = _printed_files(_host_setup("--print", GPU="1", CLOCK_MHZ="1980", KB_MEMORY_MIN="64G", INSTALL_DISPATCHER="0",
                                       SMA_DISPATCHER_CONFIG="x.yaml").stdout)

    assert "Environment=GPU=1 CLOCK_MHZ=1980\n" in files["/etc/systemd/system/kernelbench-gpu-clocks.service"]
    assert "MemoryMin=64G\n" in files["/etc/systemd/system/kernelbench.slice"]
    assert "/etc/systemd/system/sma-dispatcher.service" not in files


@pytest.mark.parametrize("env, says", [
    ({"KB_CPUS": "5-19"}, "overlap"),
    ({"KB_COMPILE_CPUS": "12-16"}, "overlap"),
    ({"KB_TIMING_CPUS": "15-21"}, "not inside KB_CPUS"),
    ({"KB_CPUS": "10-19;reboot"}, "not a CPU list"),
    ({"HARNESS_MEMORY_MAX": "lots"}, "not a size"),
    ({"HARNESS_MEMORY_HIGH": "40G"}, "above HARNESS_MEMORY_MAX"),
    ({"SMA_USER": "root"}, "must not run as root"),
    ({"SMA_REPO": "/srv/my sma"}, "absolute path"),
    ({"GPU": "0/x"}, "nvidia-smi -i"),
])
def test_host_setup_refuses_bad_parameters_before_writing_anything(env: dict, says: str) -> None:
    run = _host_setup("--print", **env)

    assert run.returncode == 2
    assert says in run.stderr
    assert run.stdout == ""


def test_host_setup_will_not_apply_without_root() -> None:
    if os.geteuid() == 0:
        pytest.skip("running as root would apply the isolation to this machine")
    run = _host_setup()

    assert run.returncode == 2
    assert "sudo" in run.stderr


def test_host_setup_check_reports_every_item_without_root() -> None:
    # This machine may or may not be set up: the check must report, not crash.
    text = _host_setup("--check")
    as_json = _host_setup("--check", "--json")
    items = json.loads(as_json.stdout)

    assert text.returncode in (0, 1), text.stderr
    lines = text.stdout.splitlines()
    assert lines and all(re.match(r"(ok  |fail) \S+ \S", line) for line in lines)
    assert {"kernelbench.slice/AllowedCPUs", "processes", "irqs", "gpu/compute_mode", "gpu/clock_lock", "memory",
            "container"} <= {i["name"] for i in items}
    assert all(set(i) == {"name", "ok", "detail"} and isinstance(i["ok"], bool) for i in items)
    assert as_json.returncode == (0 if all(i["ok"] for i in items) else 1)
    assert len(items) == len(lines)


def test_the_kernelbench_image_is_pinned_and_installs_no_model_client() -> None:
    cf = (KB / "Containerfile").read_text()
    installs = " ".join(line for line in cf.splitlines() if "pip install" in line or line.strip().startswith('"'))

    assert re.search(r"ARG BASE_IMAGE=\S+@sha256:[0-9a-f]{64}\nFROM \$BASE_IMAGE\n", cf)
    assert "ARG KERNELBENCH_COMMIT=423217d9fda91e0c2d67e4a43bf62f96f6d104f1" in cf
    assert '"torch==2.9.1"' in cf
    assert '"einops==' in cf                       # Level 3's Mamba2 problems (48, 49) import it
    assert not re.search(r"litellm|openai", installs.lower())
    for stub in ("openai", "litellm"):
        assert "raise RuntimeError" in (KB / "stubs" / stub / "__init__.py").read_text()


def test_the_swe_runner_pins_every_package_and_holds_litellm() -> None:
    lines = (SWE / "requirements.txt").read_text().splitlines()
    pins = [line.split(" ")[0] for line in lines if line and not line.startswith((" ", "#"))]

    assert pins and all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[^=\s]+", p) for p in pins)
    assert all(lines[i + 1].strip().startswith("--hash=sha256:") for i, line in enumerate(lines)
               if line and not line.startswith((" ", "#")))
    assert "litellm==1.106.2" in pins               # 1.107.0 carried a credential stealer; 1.106.2 precedes it
    assert {"swebench==5.0.2", "mini-swe-agent==2.4.6"} <= set(pins)


def test_the_swe_runner_pulls_each_task_image_by_digest() -> None:
    refs = (SWE / "images.txt").read_text().split()
    pattern = r"docker\.io/swebench/sweb\.eval\.x86_64\.[a-z0-9_.\-]+:latest@sha256:[0-9a-f]{64}"

    assert len(refs) == len(set(refs)) == 50
    assert all(re.fullmatch(pattern, ref) for ref in refs)


def test_host_setup_adds_the_dispatcher_sudo_rule_only_when_asked() -> None:
    import os
    import subprocess

    script = Path(__file__).resolve().parents[1] / "services" / "kernelbench" / "host-setup.sh"
    env = {**os.environ, "SMA_USER": "sma-test"}
    plain = subprocess.run(["bash", str(script), "--print"], capture_output=True, text=True, env=env, timeout=60)
    asked = subprocess.run(["bash", str(script), "--print"], capture_output=True, text=True, env={**env, "ALLOW_DISPATCHER_CONTROL": "1"}, timeout=60)
    assert plain.returncode == asked.returncode == 0, plain.stderr + asked.stderr
    assert "sudoers" not in plain.stdout
    rules = [line for line in asked.stdout.splitlines() if line.startswith("sma-test ALL=(root) NOPASSWD:")]
    assert [rule.rsplit(" ", 2)[-2] for rule in rules] == ["start", "stop", "restart"]
    assert all(rule.endswith(" sma-dispatcher.service") for rule in rules)
