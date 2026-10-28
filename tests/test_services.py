"""The benchmark services in services/: the KernelBench evaluation service's request checks and how
KernelBench's outcomes become its results, and the pins of both services. Offline: no torch, GPU,
Docker or network."""
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "services" / "kernelbench"
SWE = ROOT / "services" / "swe-runner"


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
    assert re.search(r"^import torch", src, re.M) is None          # torch is imported only after the sandbox


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
