"""One KernelBench evaluation in its own process (README.md).

server.py starts one for each kernel, each baseline problem and each environment probe. It
reads one JSON request on stdin, sandboxes itself (sandbox.py), runs KernelBench and writes
one JSON result to the request's result_path. A fresh process per kernel means a crash, a hung
kernel or a sticky CUDA error ends with that process and never reaches the next kernel.
"""
import contextlib
import io
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
import traceback

import sandbox

KB_DIR = os.environ.get("KERNELBENCH_DIR", "/opt/kernelbench")


def limits(timeout_s: int):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (timeout_s * 16, timeout_s * 16))  # nvcc runs in parallel


def problem(level: int, problem_id: int):
    from kernelbench.dataset import construct_kernelbench_dataset
    ds = construct_kernelbench_dataset(level, source="local", base_path=os.path.join(KB_DIR, "KernelBench"),
                                       problem_ids=[problem_id])
    return ds.get_problem_by_id(problem_id)


def stringify(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))


class Tee(io.TextIOBase):
    """Copies what KernelBench prints, to read its compile error back (it prints, then returns None)."""

    def __init__(self, out):
        self.out, self.buf = out, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.out.write(s)

    def flush(self):
        self.out.flush()


def printed_compile_error(text: str) -> str:
    """The error KernelBench printed before giving up on a build (eval.py, its "lock file" case)."""
    m = re.search(r"Failed to compile custom CUDA kernel: Record as compilation failure\. \nError: (.*?)"
                  r"\n\[Eval\] Lock file error during compilation", text, re.S)
    return m.group(1) if m else text[-16000:]


def run_eval(req: dict, scratch: str) -> dict:
    import torch
    from kernelbench import eval as kb_eval
    p = problem(req["level"], req["problem_id"])
    opts = req["options"]
    device = torch.device("cuda:0")
    tee = Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            r = kb_eval.eval_kernel_against_ref(
                original_model_src=p.code,
                custom_model_src=req["kernel_src"],
                seed_num=opts["seed"],
                num_correct_trials=opts["num_correct_trials"],
                num_perf_trials=opts["num_perf_trials"],
                measure_performance=True,
                timing_method=opts["timing_method"],
                build_dir=os.path.join(scratch, "build"),
                device=device,
                backend=opts["backend"],
                precision=kb_eval.get_torch_dtype_from_string(opts["precision"]),
                check_for_excessive_speedup=True,
                excessive_speedup_threshold=opts["excessive_speedup_threshold"],
            )
    except Exception as e:  # what escapes KernelBench: CUDA errors (illegal access, launch failure), the rest
        return {"problem_name": p.name, "compiled": "CUDA error" not in str(e), "correctness": False,
                "metadata": {"runtime_error": f"{type(e).__name__}: {e}",
                             "runtime_error_traceback": traceback.format_exc()}}
    if r is None:
        # KernelBench returns None when a compile error's text contains "lock" (meant for a lock file
        # left by a concurrent build), which "block" and blockIdx also match. Every kernel builds alone
        # in a fresh folder here, so it is the compile error itself: read it back from what was printed.
        return {"problem_name": p.name, "compiled": False, "correctness": False,
                "metadata": {"compilation_error": printed_compile_error(tee.buf.getvalue())}}
    return {"problem_name": p.name, "compiled": r.compiled, "correctness": r.correctness,
            "runtime": r.runtime, "runtime_stats": r.runtime_stats,
            "ref_runtime": r.ref_runtime, "ref_runtime_stats": r.ref_runtime_stats,
            "metadata": stringify(r.metadata)}


def printed_timing_error(text: str) -> str:
    """The error KernelBench printed before returning no time (timing.py catches everything)."""
    m = re.search(r"\[Eval\] Error in Measuring Performance: (.*)", text, re.S)
    return m.group(1).strip() if m else "KernelBench recorded no time"


def run_baseline(req: dict) -> dict:
    import torch
    from kernelbench import timing as kb_timing
    p = problem(req["level"], req["problem_id"])
    opts = req["options"]
    tee = Tee(sys.stdout)
    # What scripts/generate_baseline_time.py records for eager PyTorch, at the eval's precision.
    try:
        with contextlib.redirect_stdout(tee):
            stats = kb_timing.measure_ref_program_time(
                ref_arch_name=p.name, ref_arch_src=p.code, num_trials=opts["num_perf_trials"],
                timing_method=opts["timing_method"], use_torch_compile=False,
                device=torch.device("cuda:0"), precision=opts["precision"])
    except Exception as e:
        return {"problem_name": p.name, "error": f"{type(e).__name__}: {e}"}
    peak = torch.cuda.max_memory_allocated(0)
    if not stats:
        return {"problem_name": p.name, "error": printed_timing_error(tee.buf.getvalue()), "peak_bytes": peak}
    return {"problem_name": p.name, "stats": stats, "peak_bytes": peak}


def smi(fields: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30)
    return [v.strip() for v in out.stdout.strip().split(",")] if out.returncode == 0 else []


def run_probe(box: dict) -> dict:
    import platform
    import torch
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(), "kernelbench_commit": os.environ.get("KERNELBENCH_COMMIT"),
           "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"), "sandbox": box}
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    env["nvcc"] = (nvcc.stdout.strip().splitlines() or ["?"])[-1]
    keys = ["name", "driver_version", "pci_bus_id", "memory_total_mib", "power_limit_w", "clocks_max_sm_mhz",
            "persistence_mode"]
    env["gpu"] = dict(zip(keys, smi("name,driver_version,pci.bus_id,memory.total,power.limit,clocks.max.sm,"
                                    "persistence_mode")))
    env["gpu"]["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    # The clocks while the GPU is busy: with the host's clock lock (README.md) the SM clock must
    # sit at it, not boost or throttle.
    stop = threading.Event()

    def load():
        a = torch.randn(4096, 4096, device="cuda")
        while not stop.is_set():
            a = a @ a
            a = a / a.norm()
            torch.cuda.synchronize()
    t = threading.Thread(target=load)
    t.start()
    time.sleep(2.0)
    sample = smi("clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active")
    stop.set()
    t.join()
    env["clocks_under_load"] = dict(zip(["sm_mhz", "mem_mhz", "power_w", "temp_c", "event_reasons"], sample))
    env["clock_lock_mhz"] = os.environ.get("KERNELBENCH_GPU_CLOCK_LOCK_MHZ")
    return env


def main():
    req = json.load(sys.stdin)
    scratch = req["scratch"]
    limits(req.get("timeout_s", 600))
    # TMPDIR, HOME and every cache are inside scratch. CUDA writes its own /proc entry when it starts
    # (error 304 without it); /proc/self resolves to this process alone, never the server's.
    box = sandbox.restrict([scratch, "/dev", "/proc/self"])
    started = time.monotonic()
    try:
        out = {"eval": lambda: run_eval(req, scratch), "baseline": lambda: run_baseline(req),
               "probe": lambda: run_probe(box)}[req["mode"]]()
    except Exception as e:
        out = {"worker_error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    out["worker_seconds"] = round(time.monotonic() - started, 2)
    with open(req["result_path"], "w") as f:
        json.dump(out, f, default=str)


if __name__ == "__main__":
    main()
