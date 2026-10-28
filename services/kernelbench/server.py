"""KernelBench evaluation service (README.md).

SMA submits a batch of model-written kernels; one worker thread evaluates them one at a time on
the container's GPU, each in its own sandboxed process (worker.py), and the client polls for
results. Jobs, results and baselines live on the state volume and survive a restart; a job that
was running resumes where it stopped, and a kernel the service died on twice is recorded as a
crash. A worker that passes the container's host-memory budget is stopped before the container's
limit, which can kill the whole container, and on unified memory (GB10) also when the machine runs
low. Standard library only, so the API runs without torch.

  GET    /v1/health                       liveness, queue length
  GET    /v1/environment                  GPU, driver, CUDA, PyTorch, KernelBench, clocks, sandbox
  GET    /v1/problems?level=1             problem ids and names
  POST   /v1/jobs                         {level, kernels: [{problem_id, kernel_src, label?}], options?, name?}
  GET    /v1/jobs                         recent jobs (no sources or results)
  GET    /v1/jobs/<id>[?sources=1]        status, environment, one result per kernel
  DELETE /v1/jobs/<id>                    cancel (a running job stops after its current kernel)
  POST   /v1/baselines                    {levels: [1], options?}: eager PyTorch time of every problem
  GET    /v1/baselines                    baselines recorded, by environment
  GET    /v1/baselines/<file>             one baseline
"""
import calendar
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATE = Path(os.environ.get("KERNELBENCH_STATE", "/var/lib/kernelbench"))
SCRATCH = Path(os.environ.get("KERNELBENCH_SCRATCH", "/tmp/kernelbench"))   # emptied after every worker
KB_DIR = Path(os.environ.get("KERNELBENCH_DIR", "/opt/kernelbench"))
WORKER = Path(__file__).with_name("worker.py")
PORT = int(os.environ.get("KERNELBENCH_PORT", "8000"))
KEEP_DAYS = int(os.environ.get("KERNELBENCH_KEEP_DAYS", "30"))

LEVELS = (1, 2, 3)                      # level 4 needs Hugging Face models: not offered
MAX_BODY = 64 << 20
MAX_KERNELS = 1000
MAX_SRC = 512 << 10
MAX_QUEUED = 50
MAX_ERROR = 16000
# The container's memory limit can kill every process in it, the server too (Kubernetes sets
# memory.oom.group), so a worker is stopped while the container's anonymous memory is still this
# far below the limit.
MEMORY_MARGIN = 2 << 30
CGROUP = Path("/sys/fs/cgroup")
# On unified memory (the DGX Spark's GB10) the GPU's allocations come from the machine's RAM and are
# not charged to the container, so a worker is also stopped when the machine's available memory
# falls below this floor (0: off, for a card with its own memory).
HOST_MEMORY_FLOOR = int(float(os.environ.get("KERNELBENCH_HOST_MEMORY_FLOOR_GIB", "0")) * 2**30)
MEMINFO = Path("/proc/meminfo")
# KernelBench's eval_from_generations.py defaults, except a longer timeout (compiles share the
# machine's CPUs with its other services).
DEFAULTS = {"num_correct_trials": 5, "num_perf_trials": 100, "timing_method": "cuda_event", "backend": "cuda",
            "precision": "fp32", "timeout_s": 300, "seed": 42, "excessive_speedup_threshold": 10}
CHOICES = {"timing_method": {"cuda_event"}, "backend": {"cuda"}, "precision": {"fp32", "fp16", "bf16"}}
RANGES = {"num_correct_trials": (1, 20), "num_perf_trials": (10, 1000), "timeout_s": (60, 1800),
          "seed": (0, 2**31 - 1), "excessive_speedup_threshold": (2, 1000)}
BASELINE_OPTIONS = ("num_perf_trials", "timing_method", "precision")

lock = threading.Lock()
wake = threading.Event()
jobs: dict[str, dict] = {}
current = {"job": None, "environment": None}


class BadRequest(Exception):
    pass


# ---- Problems (the dataset folders in the image; no torch needed)

def problems(level: int) -> dict[int, str]:
    out = {}
    for f in (KB_DIR / "KernelBench" / f"level{level}").glob("*.py"):
        m = re.match(r"^(\d+)_", f.name)
        if m:
            out[int(m.group(1))] = f.stem
    return out


# ---- Requests

def options(raw, keys=None) -> dict:
    keys = keys or tuple(DEFAULTS)
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - set(keys):
        raise BadRequest(f"options takes {', '.join(keys)}")
    out = {k: DEFAULTS[k] for k in keys}
    for k, v in raw.items():
        if k in CHOICES:
            if v not in CHOICES[k]:
                raise BadRequest(f"options.{k} is one of {sorted(CHOICES[k])}")
        elif not isinstance(v, int) or isinstance(v, bool) or not RANGES[k][0] <= v <= RANGES[k][1]:
            raise BadRequest(f"options.{k} is an integer from {RANGES[k][0]} to {RANGES[k][1]}")
        out[k] = v
    return out


def eval_job(body: dict) -> dict:
    if not isinstance(body, dict) or set(body) - {"level", "kernels", "options", "name"}:
        raise BadRequest("a job takes level, kernels, options and name")
    level = body.get("level")
    if level not in LEVELS:
        raise BadRequest(f"level is one of {list(LEVELS)}")
    known = problems(level)
    kernels = body.get("kernels")
    if not isinstance(kernels, list) or not 1 <= len(kernels) <= MAX_KERNELS:
        raise BadRequest(f"kernels is a list of 1 to {MAX_KERNELS}")
    clean = []
    for i, k in enumerate(kernels):
        if not isinstance(k, dict) or set(k) - {"problem_id", "kernel_src", "label"}:
            raise BadRequest(f"kernels[{i}] takes problem_id, kernel_src and label")
        if k.get("problem_id") not in known:
            raise BadRequest(f"kernels[{i}].problem_id {k.get('problem_id')!r} is not a level {level} problem")
        src = k.get("kernel_src")
        if not isinstance(src, str) or not src.strip() or len(src.encode()) > MAX_SRC:
            raise BadRequest(f"kernels[{i}].kernel_src is source text up to {MAX_SRC} bytes")
        label = k.get("label")
        if label is not None and (not isinstance(label, str) or len(label) > 200):
            raise BadRequest(f"kernels[{i}].label is text up to 200 characters")
        clean.append({"problem_id": k["problem_id"], "kernel_src": src, "label": label})
    name = body.get("name")
    if name is not None and (not isinstance(name, str) or len(name) > 200):
        raise BadRequest("name is text up to 200 characters")
    return new_job("eval", name, {"level": level, "kernels": clean, "options": options(body.get("options"))},
                   len(clean))


def baseline_job(body: dict) -> dict:
    if not isinstance(body, dict) or set(body) - {"levels", "options", "name"}:
        raise BadRequest("a baseline takes levels, options and name")
    levels = body.get("levels", [1])
    if not isinstance(levels, list) or not levels or any(l not in LEVELS for l in levels) or len(set(levels)) != len(levels):
        raise BadRequest(f"levels is a list from {list(LEVELS)}")
    work = [{"level": l, "problem_id": p} for l in levels for p in sorted(problems(l))]
    return new_job("baseline", body.get("name"),
                   {"levels": levels, "work": work, "options": options(body.get("options"), BASELINE_OPTIONS)},
                   len(work))


def new_job(kind: str, name, spec: dict, total: int) -> dict:
    return {"id": f"kb-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}", "kind": kind,
            "name": name, "status": "queued", "created_at": now(), "started_at": None, "finished_at": None,
            "progress": {"done": 0, "total": total}, "environment": None, "results": [], "error": None,
            "cancel": False, **spec}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- Results

def clip(text, limit=MAX_ERROR):
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    return text[:head] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-(limit - head):]


def memory_error(stopped: str = "memory") -> str:
    if stopped == "host_memory":
        return (f"stopped: the machine's available memory fell below {HOST_MEMORY_FLOOR / 2**30:.1f} GiB "
                "(on unified memory the GPU's allocations and KernelBench's host copies share the same RAM)")
    return (f"stopped: the process passed the container's host-memory budget of {MEMORY_BUDGET / 2**30:.1f} GiB "
            "(KernelBench builds a problem's inputs in host memory first; some are larger than this machine)")


def kernel_result(raw: dict | None, log_tail: str, stopped: str | None, timeout_s: int) -> dict:
    """One KernelBench outcome as the API reports it: error_kind is compile, runtime, incorrect,
    timing, timeout or crash (null when the kernel compiled, was correct and was timed). stopped is
    why the server ended the worker: "timeout", "memory", "host_memory" or None."""
    out = {"problem_name": None, "compiled": False, "correct": False, "runtime_ms": None, "ref_runtime_ms": None,
           "speedup": None, "runtime_stats": None, "ref_runtime_stats": None, "excessive_speedup": False,
           "error_kind": None, "error": None, "metadata": {}}
    if stopped == "timeout":
        return {**out, "error_kind": "timeout", "error": clip(f"no result within {timeout_s} s\n{log_tail}")}
    if stopped in ("memory", "host_memory"):
        return {**out, "error_kind": "runtime", "error": clip(f"{memory_error(stopped)}\n{log_tail}")}
    if raw is None or "worker_error" in raw or raw.get("retry"):
        why = (raw or {}).get("worker_error") or ("the build was interrupted (KernelBench lock file)" if raw else
                                                  "the worker exited without a result")
        return {**out, "error_kind": "crash", "error": clip(f"{why}\n{(raw or {}).get('traceback', '')}{log_tail}")}
    md = dict(raw.get("metadata") or {})
    out.update(problem_name=raw.get("problem_name"), compiled=bool(raw.get("compiled")),
               correct=bool(raw.get("correctness")), excessive_speedup=bool(md.pop("excessive_speedup", False)))
    if not out["compiled"]:
        if "compilation_error" in md:
            out["error_kind"], out["error"] = "compile", md.pop("compilation_error")
        else:
            out["error_kind"] = "runtime"
            out["error"] = md.pop("runtime_error", None) or md.pop("cuda_error", None) or md.pop("other_error", None)
    elif not out["correct"]:
        if "runtime_error" in md:
            out["error_kind"] = "runtime"
            out["error"] = f"{md.pop('runtime_error')}\n{md.pop('runtime_error_traceback', '')}".strip()
        else:
            out["error_kind"] = "incorrect"
            out["error"] = md.get("correctness_issue") or "output does not match the reference"
    else:
        rt, ref = raw.get("runtime"), raw.get("ref_runtime")
        out.update(runtime_stats=raw.get("runtime_stats") or None, ref_runtime_stats=raw.get("ref_runtime_stats") or None)
        if isinstance(rt, (int, float)) and rt > 0:
            out["runtime_ms"] = rt
        if isinstance(ref, (int, float)) and ref > 0:
            out["ref_runtime_ms"] = ref
        if out["runtime_ms"] and out["ref_runtime_ms"]:
            out["speedup"] = round(out["ref_runtime_ms"] / out["runtime_ms"], 4)
        if "error_during_performance" in md or not out["runtime_ms"]:
            out["error_kind"] = "timing"
            out["error"] = md.pop("error_during_performance", None) or "the kernel was correct but not timed"
    out["error"] = clip(out["error"])
    out["metadata"] = {k: clip(v, 2000) for k, v in md.items()}
    return out


def fingerprint(env: dict | None) -> str | None:
    """What a speedup depends on: the same fingerprint means comparable numbers."""
    if not env or "gpu" not in env:
        return None
    key = {k: env.get(k) for k in ("torch", "cuda_runtime", "nvcc", "kernelbench_commit", "clock_lock_mhz")}
    key.update(gpu=env["gpu"].get("name"), driver=env["gpu"].get("driver_version"))
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def baseline_name(level: int, opts: dict) -> str:
    return f"level{level}-{opts['precision']}-{opts['timing_method']}-{opts['num_perf_trials']}.json"


def baseline_times(env: dict, level: int, opts: dict) -> dict:
    f = STATE / "baselines" / (fingerprint(env) or "none") / baseline_name(level, opts)
    try:
        data = json.loads(f.read_text())
    except (OSError, ValueError):
        return {}
    return {p["problem_id"]: p["stats"]["mean"] for p in data.get("problems", []) if p.get("stats")}


# ---- Storage

def save(job: dict):
    d = STATE / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{job['id']}.tmp"
    tmp.write_text(json.dumps(job))
    tmp.replace(d / f"{job['id']}.json")


def load_jobs():
    for f in sorted((STATE / "jobs").glob("kb-*.json")):
        try:
            job = json.loads(f.read_text())
        except ValueError:
            continue
        if job["status"] in ("done", "failed", "cancelled") and job.get("finished_at") and \
                time.time() - calendar.timegm(time.strptime(job["finished_at"], "%Y-%m-%dT%H:%M:%SZ")) > KEEP_DAYS * 86400:
            f.unlink()
            continue
        if job["status"] == "running":      # the service restarted mid-job: carry on where it stopped
            job["status"] = "queued"
            job.setdefault("notes", []).append(f"{now()}: resumed after a service restart")
            if job.get("in_flight"):
                job["in_flight"]["restarts"] = job["in_flight"].get("restarts", 0) + 1
        jobs[job["id"]] = job


def public(job: dict, sources=False) -> dict:
    out = {k: v for k, v in job.items() if k not in ("kernels", "work", "cancel", "in_flight")}
    if job["kind"] == "eval" and sources:
        out["kernels"] = job["kernels"]
    return out


def summary(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "kind", "name", "status", "created_at", "started_at", "finished_at",
                                    "progress", "error")}


# ---- The worker thread

def wipe_scratch():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for p in SCRATCH.iterdir():
        try:
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        except OSError:
            pass


def memory_budget() -> int | None:
    """The container's memory limit less the margin; None without a limit (or outside cgroup v2)."""
    try:
        limit = (CGROUP / "memory.max").read_text().strip()
    except OSError:
        return None
    return int(limit) - MEMORY_MARGIN if limit.isdigit() else None


def memory_in_use() -> int:
    """The container's anonymous and shared memory: what the limit can't reclaim (page cache it can)."""
    try:
        stat = dict(line.split() for line in (CGROUP / "memory.stat").read_text().splitlines())
        return int(stat["anon"]) + int(stat.get("shmem", 0))
    except (OSError, KeyError, ValueError):
        return 0


def host_memory_available() -> int | None:
    """The machine's MemAvailable (/proc/meminfo is the host's in a container); None if unreadable."""
    try:
        for line in MEMINFO.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


MEMORY_BUDGET = memory_budget()


def start_item(job: dict, i: int) -> bool:
    """Marks item i as running (saved first), or False if the service died twice while it ran."""
    inf = job.get("in_flight") or {}
    restarts = inf.get("restarts", 0) if inf.get("index") == i else 0
    if restarts >= 2:
        return False
    with lock:
        job["in_flight"] = {"index": i, "restarts": restarts}
        save(job)
    return True


DIED_ON = ("the service was killed twice while this ran (the container went over its memory limit, or the "
           "machine restarted); not tried again")


def run_worker(req: dict, timeout_s: int) -> tuple[dict | None, str, str | None]:
    """One sandboxed worker process: (its result or None, the end of its log, why the server
    stopped it: "timeout", "memory", "host_memory" or None)."""
    work = SCRATCH / f"kb-{uuid.uuid4().hex[:12]}"
    for sub in ("", "tmp", "build", "cache"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    req = {**req, "scratch": str(work), "result_path": str(work / "result.json"), "timeout_s": timeout_s}
    env = {k: v for k, v in os.environ.items() if not k.startswith("KUBERNETES")}
    env.update(HOME=str(work), TMPDIR=str(work / "tmp"), TORCH_EXTENSIONS_DIR=str(work / "build"),
               TRITON_CACHE_DIR=str(work / "cache" / "triton"), XDG_CACHE_HOME=str(work / "cache"),
               PYTHONDONTWRITEBYTECODE="1")
    stopped = None
    try:
        with open(work / "log", "wb") as log:
            p = subprocess.Popen([sys.executable, "-B", str(WORKER)], stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, env=env, cwd=work, start_new_session=True)
            p.stdin.write(json.dumps(req).encode())
            p.stdin.close()
            deadline = time.monotonic() + timeout_s
            while p.poll() is None:
                if time.monotonic() > deadline:
                    stopped = "timeout"
                    break
                if MEMORY_BUDGET and memory_in_use() > MEMORY_BUDGET:
                    stopped = "memory"
                    break
                if HOST_MEMORY_FLOOR and (host_memory_available() or HOST_MEMORY_FLOOR) < HOST_MEMORY_FLOOR:
                    stopped = "host_memory"
                    break
                time.sleep(0.02)
            try:
                os.killpg(p.pid, signal.SIGKILL)    # the worker and anything it left running
            except ProcessLookupError:
                pass
            p.wait()
        raw = None
        rp = work / "result.json"
        if rp.exists() and rp.stat().st_size < (16 << 20):
            try:
                raw = json.loads(rp.read_text())
            except ValueError:
                raw = None
        with open(work / "log", "rb") as f:
            f.seek(max(0, (work / "log").stat().st_size - 8000))
            tail = f.read().decode(errors="replace")
        return raw, tail, stopped
    finally:
        wipe_scratch()


def probe() -> dict:
    raw, tail, _ = run_worker({"mode": "probe"}, 300)
    if not raw or "worker_error" in raw:
        raise RuntimeError(clip(f"environment probe failed: {(raw or {}).get('worker_error', 'no result')}\n{tail}"))
    raw.pop("worker_seconds", None)
    raw["fingerprint"] = fingerprint(raw)
    raw["host_memory_budget_gib"] = round(MEMORY_BUDGET / 2**30, 1) if MEMORY_BUDGET else None
    raw["host_memory_floor_gib"] = round(HOST_MEMORY_FLOOR / 2**30, 1) if HOST_MEMORY_FLOOR else None
    raw["probed_at"] = now()
    return raw


def run_eval_job(job: dict):
    opts = job["options"]
    base = baseline_times(job["environment"], job["level"], opts)
    while len(job["results"]) < len(job["kernels"]):
        if job["cancel"]:
            return
        i = len(job["results"])
        k = job["kernels"][i]
        started = time.monotonic()
        attempt = 0
        if not start_item(job, i):
            r = {**kernel_result(None, "", None, opts["timeout_s"]), "error": DIED_ON}
        else:
            for attempt in (1, 2):      # once more only for KernelBench's interrupted-build case
                raw, tail, stopped = run_worker({"mode": "eval", "level": job["level"], "problem_id": k["problem_id"],
                                                 "kernel_src": k["kernel_src"], "options": opts}, opts["timeout_s"])
                if not (raw and raw.get("retry")):
                    break
            r = kernel_result(raw, tail, stopped, opts["timeout_s"])
        r = {"index": i, "problem_id": k["problem_id"], "label": k["label"],
             "kernel_sha256": hashlib.sha256(k["kernel_src"].encode()).hexdigest(), **r,
             "baseline_ref_runtime_ms": base.get(k["problem_id"]), "attempts": attempt,
             "eval_seconds": round(time.monotonic() - started, 1)}
        if r["problem_name"] is None:
            r["problem_name"] = problems(job["level"]).get(k["problem_id"])
        with lock:
            job["results"].append(r)
            job["progress"]["done"] = len(job["results"])
            job.pop("in_flight", None)
            save(job)


def run_baseline_job(job: dict):
    opts = job["options"]
    while len(job["results"]) < len(job["work"]):
        if job["cancel"]:
            return
        i = len(job["results"])
        w = job["work"][i]
        raw, tail, stopped = None, "", None
        ran = start_item(job, i)
        if ran:
            raw, tail, stopped = run_worker({"mode": "baseline", "level": w["level"], "problem_id": w["problem_id"],
                                             "options": opts}, 600)
        r = {"level": w["level"], "problem_id": w["problem_id"],
             "problem_name": (raw or {}).get("problem_name") or problems(w["level"]).get(w["problem_id"])}
        if raw and raw.get("stats"):
            r.update(stats=raw["stats"], peak_bytes=raw.get("peak_bytes"))
        else:
            r["error"] = clip(DIED_ON if not ran else
                              "timed out" if stopped == "timeout" else
                              memory_error(stopped) if stopped in ("memory", "host_memory") else
                              (raw or {}).get("error") or (raw or {}).get("worker_error") or f"no result\n{tail}", 4000)
        with lock:
            job["results"].append(r)
            job["progress"]["done"] = len(job["results"])
            job.pop("in_flight", None)
            save(job)
    fp = job["environment"]["fingerprint"]
    d = STATE / "baselines" / fp
    d.mkdir(parents=True, exist_ok=True)
    for level in job["levels"]:
        rows = [r for r in job["results"] if r["level"] == level]
        (d / baseline_name(level, opts)).write_text(json.dumps(
            {"level": level, "options": opts, "environment": job["environment"], "job": job["id"],
             "recorded_at": now(), "problems": rows}, indent=1))


def worker_loop():
    while True:
        with lock:
            queued = sorted((j for j in jobs.values() if j["status"] == "queued"), key=lambda j: j["created_at"])
            job = queued[0] if queued else None
            if job:
                job["status"], job["started_at"] = "running", job["started_at"] or now()
                current["job"] = job["id"]
                save(job)
        if not job:
            wake.wait(30)
            wake.clear()
            continue
        try:
            env = probe()
            with lock:
                if job["environment"] and job["environment"].get("fingerprint") != env["fingerprint"]:
                    job.setdefault("notes", []).append(f"{now()}: environment changed while the job waited")
                job["environment"] = env
                current["environment"] = env
            (run_eval_job if job["kind"] == "eval" else run_baseline_job)(job)
            status, error = ("cancelled", None) if job["cancel"] else ("done", None)
        except Exception as e:
            status, error = "failed", clip(f"{type(e).__name__}: {e}")
        with lock:
            job.update(status=status, error=error, finished_at=now())
            current["job"] = None
            save(job)


def startup_probe():
    try:
        env = probe()
        with lock:
            current["environment"] = current["environment"] or env
    except Exception as e:
        with lock:
            current["environment"] = current["environment"] or {"error": clip(str(e))}


# ---- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "kernelbench-eval"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{now()} {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}\n")

    def send(self, code: int, body):
        data = json.dumps(body, indent=1).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY:
            raise BadRequest(f"send a JSON body up to {MAX_BODY >> 20} MiB")
        try:
            return json.loads(self.rfile.read(n))
        except ValueError:
            raise BadRequest("the body is not JSON")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        parts = [p for p in u.path.split("/") if p]
        with lock:
            if parts == ["v1", "health"]:
                return self.send(200, {"ok": True, "running": current["job"],
                                       "queued": sum(j["status"] == "queued" for j in jobs.values())})
            if parts == ["v1", "environment"]:
                return self.send(200, current["environment"] or {"pending": "probing the GPU"})
            if parts == ["v1", "problems"]:
                raw = (q.get("level") or ["1"])[0]
                level = int(raw) if raw.isdigit() else 0
                if level not in LEVELS:
                    return self.send(400, {"error": f"level is one of {list(LEVELS)}"})
                return self.send(200, {"level": level, "problems": [{"problem_id": i, "name": n}
                                                                    for i, n in sorted(problems(level).items())]})
            if parts == ["v1", "jobs"]:
                return self.send(200, {"jobs": [summary(j) for j in sorted(jobs.values(), key=lambda j: j["created_at"],
                                                                           reverse=True)[:100]]})
            if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
                job = jobs.get(parts[2])
                if not job:
                    return self.send(404, {"error": "no such job"})
                return self.send(200, public(job, sources=(q.get("sources") or ["0"])[0] == "1"))
            if parts == ["v1", "baselines"]:
                out = [{"fingerprint": d.name, "file": f.name, "current": d.name == (current["environment"] or {}).get("fingerprint")}
                       for d in sorted((STATE / "baselines").glob("*")) for f in sorted(d.glob("*.json"))]
                return self.send(200, {"baselines": out})
            if len(parts) == 3 and parts[:2] == ["v1", "baselines"] and re.match(r"^level\d-[a-z0-9_-]+\.json$", parts[2]):
                fp = (q.get("fingerprint") or [(current["environment"] or {}).get("fingerprint") or ""])[0]
                f = STATE / "baselines" / re.sub(r"[^0-9a-f]", "", fp) / parts[2]
                if not f.is_file():
                    return self.send(404, {"error": "no such baseline for this environment"})
                return self.send(200, json.loads(f.read_text()))
        self.send(404, {"error": "not found"})

    def do_POST(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            if parts == ["v1", "jobs"]:
                job = eval_job(self.body())
            elif parts == ["v1", "baselines"]:
                job = baseline_job(self.body())
            else:
                return self.send(404, {"error": "not found"})
        except BadRequest as e:
            return self.send(400, {"error": str(e)})
        with lock:
            queued = [j for j in jobs.values() if j["status"] == "queued"]
            if len(queued) >= MAX_QUEUED:
                return self.send(429, {"error": f"{len(queued)} jobs are already queued"})
            jobs[job["id"]] = job
            save(job)
            position = len(queued) + (1 if current["job"] else 0)
        wake.set()
        self.send(202, {"job_id": job["id"], "status": "queued", "position": position,
                        "total": job["progress"]["total"], "poll": f"/v1/jobs/{job['id']}"})

    def do_DELETE(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) != 3 or parts[:2] != ["v1", "jobs"]:
            return self.send(404, {"error": "not found"})
        with lock:
            job = jobs.get(parts[2])
            if not job:
                return self.send(404, {"error": "no such job"})
            if job["status"] == "queued":
                job.update(status="cancelled", finished_at=now())
            elif job["status"] == "running":
                job["cancel"] = True
            save(job)
            return self.send(200, summary(job) | {"cancel_requested": job["cancel"]})


def main():
    (STATE / "jobs").mkdir(parents=True, exist_ok=True)
    (STATE / "baselines").mkdir(parents=True, exist_ok=True)
    load_jobs()
    threading.Thread(target=lambda: (startup_probe(), worker_loop()), daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
