import os
import socket
import subprocess
import sys
from pathlib import Path


def test_dry_run_prints_the_plan_without_a_gpu() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env.pop("HF_TOKEN", None)
    # The dry run looks up the repo's config and file sizes; offline, it says why it has no numbers.
    env["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "sma", "run", "tiny-model", "--bf16", "--dry-run"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    stdout = completed.stdout
    quality = next(line for line in stdout.splitlines() if line.startswith("quality-server:"))
    speed = next(line for line in stdout.splitlines() if line.startswith("speed-server:"))
    assert "VLLM_BATCH_INVARIANT=1" in quality
    assert "--no-enable-flashinfer-autotune" in quality
    assert "--tensor-parallel-size 1" in quality
    assert "--no-async-scheduling" in quality
    assert "--moe-backend" not in quality
    assert "VLLM_BATCH_INVARIANT" not in speed
    assert "vllm bench serve" in stdout
    assert "--percentile-metrics ttft,tpot,itl,e2el" in stdout
    assert "--metric-percentiles 50,90,99" in stdout
    assert "--save-result" in stdout
    assert "nel eval run nel.yaml -o attempt-1" in stdout
    assert "musique: sma:musique, 100 questions (seeded shuffle)" in stdout
    assert "6c browsecomp_plus: BrowseComp-Plus 04694903, 100 of 830 questions" in stdout
    assert "search_server.py --index indexes/bm25" in stdout and "6d browsecomp_plus_results" in stdout
    assert "host_alias: gpu-host-1" in stdout
    assert "quantize: no" in stdout
    assert "kv cache: fp8 (always 8-bit)" in stdout
    assert "--kv-cache-dtype fp8" in quality and "--kv-cache-dtype fp8" in speed
    assert "memory plan: computed at fetch" in stdout and "Not looked up here" in stdout
    assert socket.gethostname() not in stdout
    assert "runs" not in completed.stderr or "error" not in completed.stderr.lower()


def test_assume_moe_dry_run_includes_the_backend() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    env["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sma",
            "run",
            "org/model",
            "--nvfp4",
            "--dry-run",
            "--assume-moe",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    quality = next(line for line in completed.stdout.splitlines() if line.startswith("quality-server:"))
    assert "--moe-backend flashinfer_b12x" in quality
    assert "--tensor-parallel-size 1" in quality
    # A quantized checkpoint runs its real kernels: the dry-run prints what the run starts.
    assert "VLLM_BATCH_INVARIANT" not in quality
    assert "batch invariance off: a nvfp4 checkpoint runs its real kernels" in completed.stdout
