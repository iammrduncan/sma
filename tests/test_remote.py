"""Remote serving (sma.remote): llama-server on the V100 pod, started over SSH, reached through a forward."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sma.config import config_from_mapping
from sma.errors import SmaError
from sma.remote import RemoteRuntime, remote_serve_args

BUILD = {"tag": "b11960", "commit": "8642541b255bfb5dc39cd351a178be09013a55d6", "cuda": "12.8", "sha256": {}}


class FakeRunner:
    """The serving machine: remembers every script, start, forward, and copy."""

    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.started: list[tuple[str, str, str]] = []
        self.forwards: list[tuple[int, int]] = []
        self.target = "sma@pod"

    def shell(self, script: str, *, what: str, timeout_s: float = 300.0) -> str:
        self.scripts.append(script)
        if script.startswith("nvidia-smi"):
            return "Tesla PG500-216, 32768 MiB, 580.178.04\n"
        return "alive\n" if "kill -0" in script and "echo alive" in script else ""

    def read_text(self, rel: str) -> str | None:
        return json.dumps(BUILD) if rel.endswith("build.json") else None

    def start(self, rel_dir: str, name: str, command: str) -> None:
        self.started.append((rel_dir, name, command))

    def status(self, rel_dir: str, name: str) -> tuple[int | None, bool]:
        return None, True

    def forward(self, local_port: int, remote_port: int, log_path: Path) -> Any:
        self.forwards.append((local_port, remote_port))
        return SimpleNamespace(stop=lambda timeout_s=0.1: None)

    def command(self, script: str) -> list[str]:
        return ["ssh", self.target, script]

    def prefix(self) -> list[str]:
        return ["ssh", self.target]


class FakeLocal:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    def stop(self, name: str) -> None:
        self.stopped.append(name)

    def list_sma(self) -> list[str]:
        return ["sma-r1-quality-bfcl_v3"]

    def ensure_image(self, image: str | None = None) -> str | None:
        return "sha256:local"


def _cfg(**remote: Any):
    return config_from_mapping({"remote": {"enabled": True, "gpus": "1", "remote_port": 8002, **remote}, "engine": {"port": 18002}})


def test_the_servers_flags_point_at_the_model_and_the_loopback_on_the_serving_machine() -> None:
    argv = ["-m", "/model/LFM3-1.7B-F16.gguf", "--host", "0.0.0.0", "--port", "18002", "-ngl", "all", "-c", "526777"]
    assert remote_serve_args(argv, model_dir="/work/sma/models/x@abc", port=8002) == [
        "-m", "/work/sma/models/x@abc/LFM3-1.7B-F16.gguf", "--host", "127.0.0.1", "--port", "8002", "-ngl", "all", "-c", "526777",
    ]  # fmt: skip


def test_a_server_starts_on_its_gpu_from_the_pinned_build_and_the_harness_reaches_it_through_a_forward() -> None:
    cfg = _cfg()
    runner, local = FakeRunner(), FakeLocal()
    runtime = RemoteRuntime(cfg, runner=runner, local=local)
    spec = SimpleNamespace(
        name="sma-r1-quality", mode="quality", env={}, port=18002, remote_model_dir="/work/sma/models/x@abc",
        serve_argv=["-m", "/model/m.gguf", "--host", "0.0.0.0", "--port", "18002", "-np", "8"],
    )
    runtime.start(spec)

    (folder, name, command), = runner.started
    assert (folder, name) == ("servers/sma-r1-quality", "server")
    assert "CUDA_VISIBLE_DEVICES=1" in command and "/work/sma/llama.cpp-b11960/build/bin/llama-server" in command
    assert "/work/sma/models/x@abc/m.gguf" in command and "--port 8002" in command and "echo $$ > servers/sma-r1-quality/llama.pid" in command
    assert runner.forwards == [(18002, 8002)]
    assert runtime.ensure_image(cfg.engine.llamacpp_image) == BUILD["commit"]
    assert runtime.ensure_image("nvcr.io/nvidia/eval-factory/bfcl:26.03") == "sha256:local"  # Pulled here.

    runtime.stop("sma-r1-quality")
    assert any("kill -TERM $(cat servers/sma-r1-quality/llama.pid)" in script for script in runner.scripts)
    # The eval-factory suites stay in Docker here; teardown sees both.
    runtime.stop("sma-r1-quality-bfcl_v3")
    assert local.stopped == ["sma-r1-quality-bfcl_v3"]
    assert "sma-r1-quality-bfcl_v3" in runtime.list_sma()
    facts = runtime.facts()
    assert facts["hardware"] == "v100" and facts["gpu_names"] == ["Tesla PG500-216"] and facts["build"]["commit"] == BUILD["commit"]


def test_the_model_is_downloaded_on_the_serving_machine_from_the_pinned_revision_and_checked() -> None:
    runner = FakeRunner()
    runtime = RemoteRuntime(_cfg(), runner=runner, local=FakeLocal())
    manifest = {
        "model_repo": "LiquidAI/LFM3-1.7B-GGUF", "revision_sha": "bad5729bba61d076fff20d2548e62526665af215",
        "gguf": {"files": ["LFM3-1.7B-F16.gguf"]}, "file_hashes": {"LFM3-1.7B-F16.gguf": "sha256:" + "b7" * 32},
    }  # fmt: skip
    folder = runtime.ensure_model(manifest)
    assert folder == "/work/sma/models/LiquidAI_LFM3-1.7B-GGUF@bad5729bba61"
    (script,) = runner.scripts
    assert "https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/bad5729bba61d076fff20d2548e62526665af215/LFM3-1.7B-F16.gguf" in script
    assert "sha256sum" in script and "b7" * 32 in script and "exit 4" in script
    with pytest.raises(SmaError, match="no sha256"):
        runtime.ensure_model({**manifest, "file_hashes": {}})


def test_a_remote_run_is_llama_cpp_only_and_its_folder_names_the_hardware(tmp_path: Path) -> None:
    from sma.pipeline import RunSpec, execute
    from tests.harness import make_config, make_services

    cfg = make_config(tmp_path)
    cfg.remote.enabled = True
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    with pytest.raises(SmaError, match="runs llama.cpp only"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16", engine="vllm"), cfg, services)
    assert not list((tmp_path / "runs").glob("*/vllm*"))


def test_a_remote_worker_claims_only_gguf_jobs(tmp_path: Path) -> None:
    from sma.queue import JobQueue

    queue = JobQueue(tmp_path / "queue")
    queue.add([{"repo": "LiquidAI/LFM3-VL-2B", "quant": "bf16"}], kind="single")
    queue.add([{"repo": "LiquidAI/LFM3-1.7B-GGUF", "quant": "gguf", "gguf": "F16", "engine": "llamacpp"}], kind="single")
    claimed = queue.claim("w", "v100-0", gguf_only=True)
    assert claimed is not None and claimed[1]["legs"][0]["repo"] == "LiquidAI/LFM3-1.7B-GGUF"
    assert queue.claim("w", "v100-0", gguf_only=True) is None  # The safetensors job stays for another worker.


def test_telemetry_reads_the_remote_gpu_over_ssh(tmp_path: Path) -> None:
    from sma.telemetry import TelemetryRecorder

    recorder = TelemetryRecorder(tmp_path / "telemetry.csv", _cfg(gpus="2,3"), query_prefix=["ssh", "sma@pod"])
    assert recorder._query[:2] == ["ssh", "sma@pod"] and "nvidia-smi" in recorder._query[2] and "-i 2" in recorder._query[2]


def test_command_output_that_is_not_utf8_is_kept_with_replacement_characters() -> None:
    # The serving machine's log tail (tail -c) cut a character in half, run_command raised UnicodeDecodeError,
    # and the run recorded its KV cache type as unconfirmed (V100, 2026-11-01).
    from sma.proc import run_command

    result = run_command(["printf", "\\304llama_kv_cache: size = 1 MiB, K (q8_0): 1, V (q8_0): 1"], timeout_s=10, what="printf")
    assert result.ok and result.stdout.startswith("�") and "K (q8_0)" in result.stdout
