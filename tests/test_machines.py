"""What sma watch says each machine is: from its own facts, never its name."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from sma.errors import SmaError
from sma.machines import Machine, describe_host, judge_machine, kernelbench_machine, parse_facts, runner_machine
from sma.watch import View, render
from tests.harness import make_config

# Real output of machines.FACTS_SCRIPT on a DGX Spark and on the x86_64 runner (a QEMU VM), 2026-10-28.
SPARK = (
    "arch=aarch64\nproduct=NVIDIA_DGX_Spark\nvendor=NVIDIA\ncpu=Cortex-X925\ncpu=Cortex-A725\n"
    "cores=20\nmem_kib=127606784\ngpu=NVIDIA GB10\n"
)
QEMU = (
    "arch=x86_64\nproduct=Standard PC (i440FX + PIIX, 1996)\nvendor=QEMU\ncpu=QEMU Virtual CPU version 2.5+\n"
    "cores=10\nmem_kib=24615120\n"
)


def test_a_spark_and_a_qemu_runner_are_described_by_kind_and_chip() -> None:
    assert describe_host("model", parse_facts(SPARK)) == Machine("model", True, "DGX Spark · GB10 GPU · 20× Cortex-X925/A725 · 121.7 GiB")
    assert describe_host("swe", parse_facts(QEMU)) == Machine("swe", True, "QEMU VM · no GPU · 10× x86_64 vCPU · 23.5 GiB")


def test_services_report_what_their_apis_give_or_say_they_are_down_or_unset(monkeypatch) -> None:
    monkeypatch.delenv("KERNELBENCH_URL", raising=False)
    monkeypatch.delenv("SMA_JUDGE_URL", raising=False)
    assert kernelbench_machine() == Machine("kernel", None)
    assert judge_machine() == Machine("judge", None)

    monkeypatch.setenv("KERNELBENCH_URL", "http://kb.example:9000")
    monkeypatch.setenv("SMA_JUDGE_URL", "http://judge.example:8000/v1")
    monkeypatch.setenv("SMA_JUDGE_API_KEY", "secret-key")
    asked = []

    def reply(method, url, *, timeout_s, attempts, headers=None):
        asked.append((url, headers))
        if url.endswith("/v1/environment"):
            return {"gpu": {"name": "NVIDIA GB10", "compute_capability": "12.1"}, "cuda_runtime": "13.0", "torch": "2.9.1+cu130"}
        return {"data": [{"id": "ornith-1.5-122b"}]}

    monkeypatch.setattr("sma.machines.request_json", reply)
    assert kernelbench_machine() == Machine("kernel", True, "GB10 GPU (sm 12.1) · CUDA 13.0 · torch 2.9.1 · type not reported")
    assert judge_machine() == Machine("judge", True, "serves ornith-1.5-122b · hardware not reported")
    assert asked[1] == ("http://judge.example:8000/v1/models", {"Authorization": "Bearer secret-key"})

    def down(*args, **kwargs):
        raise SmaError("connection refused")

    monkeypatch.setattr("sma.machines.request_json", down)
    assert kernelbench_machine().up is False and judge_machine().up is False


def test_the_runner_is_asked_over_ssh_once_and_may_be_down(tmp_path: Path, monkeypatch) -> None:
    cfg = make_config(tmp_path)

    class Runner:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def machine_facts(self, script: str, *, timeout_s: float) -> str:
            if self.text is None:
                raise SmaError("ssh: connect timed out")
            assert "uname -m" in script
            return self.text

    monkeypatch.setattr("sma.swebench.runner_from_env", lambda cfg, **kw: Runner(QEMU))
    assert runner_machine(cfg).summary.startswith("QEMU VM")
    monkeypatch.setattr("sma.swebench.runner_from_env", lambda cfg, **kw: Runner(None))
    assert runner_machine(cfg) == Machine("swe", False)
    monkeypatch.setattr("sma.swebench.runner_from_env", lambda cfg, **kw: None)
    assert runner_machine(cfg) == Machine("swe", None)


def test_watch_lists_each_machine_with_its_state(tmp_path: Path) -> None:
    machines = [
        describe_host("model", parse_facts(SPARK)),
        Machine("kernel", True, "GB10 GPU (sm 12.1) · CUDA 13.0"),
        Machine("swe", False),
        Machine("judge", None),
    ]
    now = datetime(2026, 10, 29, 3, 0, tzinfo=timezone.utc)
    screen = render(make_config(tmp_path), View(["sma-x-quality"], (84 * 2**30, 121 * 2**30), now, machines=machines, interval_s=5))
    top = screen.split("QUEUE")[0]
    assert "redraws every 5 s" in top
    # One line per machine, so the screen fits.
    assert "● model   DGX Spark · GB10 GPU · 20× Cortex-X925/A725 · 121.7 GiB, 84.0 free" in top
    assert "● kernel  GB10 GPU (sm 12.1) · CUDA 13.0" in top
    assert "● swe     does not answer" in top
    assert "○ judge   not set in .env" in top
    assert len(top.split("MACHINES")[1].strip().splitlines()) == 4
