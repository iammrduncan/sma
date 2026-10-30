"""KernelBench and SWE-bench as run steps (owner decision 2026-10-27), on fakes of both lab machines.

The fake SWE-bench runner keeps its 'remote' folder in tmp_path and runs SMA's own remote Python
snippets there, so pruning and status reading are tested as they run on the real runner.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sma.pipeline import RunSpec, execute
from sma.schemautil import validate_instance
from tests.harness import make_config, make_services

ROOT = Path(__file__).resolve().parents[1]
QWEN3_TEMPLATE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}"
)


class FakeKernelBench:
    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.polls = 0

    def environment(self) -> dict[str, Any]:
        return {"fingerprint": "fp-test", "gpu": {"name": "NVIDIA GB10"}}

    def submit(self, *, level: int, name: str, kernels: list[dict[str, Any]], options: dict[str, Any]) -> str:
        self.submitted.append({"level": level, "name": name, "kernels": kernels, "options": options})
        return "kb-test"

    def job(self, job_id: str) -> dict[str, Any]:
        self.polls += 1
        if self.polls == 1:
            return {"status": "running", "progress": {"done": 0, "total": 100}}
        kernels = self.submitted[0]["kernels"]
        # Even problems are correct; problems divisible by 4 are twice as fast as the baseline.
        results = [
            {
                "problem_id": k["problem_id"],
                "compiled": True,
                "correct": k["problem_id"] % 2 == 0,
                "runtime_ms": 1.0 if k["problem_id"] % 4 == 0 else 3.0,
                "baseline_ref_runtime_ms": None if k["problem_id"] == 95 else 2.0,
                "excessive_speedup": False,
                "error_kind": None if k["problem_id"] % 2 == 0 else "incorrect",
            }
            for k in kernels
        ]
        return {"status": "done", "options": {"precision": "fp32", "timing_method": "cuda_event", "num_perf_trials": 100},
                "environment": {"fingerprint": "fp-test"}, "results": results}

    def baseline_ids(self, *, level: int, options: dict[str, Any], fingerprint: str) -> set[int]:
        assert (level, fingerprint) == (1, "fp-test")
        return set(range(1, 101)) - {95}


class FakeTunnel:
    class process:  # noqa: N801 - the shape of proc.Following
        @staticmethod
        def poll() -> None:
            return None

    stopped = False

    def stop(self, timeout_s: float = 10.0) -> None:
        FakeTunnel.stopped = True


class FakeRunner:
    """The runner's folder in tmp_path. The agent's first launch loses the model on one task."""

    def __init__(self, root: Path, tasks: list[str], *, resolved: set[str], lost: set[str]) -> None:
        self.root, self.tasks, self.resolved, self.lost = root, tasks, resolved, lost
        self.started: list[tuple[str, str]] = []
        self.written: dict[str, str] = {}
        (root / "data").mkdir(parents=True)
        (root / "data" / "verified-mini.jsonl").write_text("".join(json.dumps({"instance_id": t}) + "\n" for t in tasks))

    def check(self, cfg: Any) -> None:
        return None

    def write_text(self, rel: str, text: str) -> None:
        self.written[rel] = text
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_text(text)

    def start(self, rel_dir: str, name: str, command: str) -> None:
        self.started.append((name, command))
        folder = self.root / rel_dir
        if name == "agent":
            preds_path = folder / "preds.json"
            preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}
            first = sum(1 for started, _ in self.started if started == "agent") == 1
            for task in self.tasks:
                if task in preds:
                    continue
                status = "APIConnectionError" if first and task in self.lost else "Submitted"
                (folder / task).mkdir(parents=True, exist_ok=True)
                traj = {"info": {"exit_status": status, "model_stats": {"api_calls": 3}}, "messages": [], "instance_id": task}
                (folder / task / f"{task}.traj.json").write_text(json.dumps(traj))
                preds[task] = {"model_name_or_path": "x", "instance_id": task, "model_patch": "" if status != "Submitted" else "diff"}
            preds_path.write_text(json.dumps(preds))
        else:
            (folder / "report.json").write_text(json.dumps({"total_instances": len(self.tasks), "resolved_ids": sorted(self.resolved)}))
        (folder / f"{name}.log").write_text(f"{name} ran\n")
        (folder / f"{name}.exit").write_text("0\n")

    def status(self, rel_dir: str, name: str) -> tuple[int | None, bool]:
        done = self.root / rel_dir / f"{name}.exit"
        return (int(done.read_text()), False) if done.exists() else (None, False)

    def stop(self, run_id: str) -> None:
        return None

    def python(self, rel_dir: str, code: str, *args: str, what: str, timeout_s: float = 300.0) -> str:
        folder = self.root / rel_dir
        folder.mkdir(parents=True, exist_ok=True)
        return subprocess.run([sys.executable, "-c", code, *args], cwd=folder, capture_output=True, text=True, check=True).stdout

    def read_text(self, rel: str) -> str | None:
        path = self.root / rel
        return path.read_text() if path.is_file() else None

    def tunnel(self, port: int, log_path: Path) -> FakeTunnel:
        return FakeTunnel()


def _setup(tmp_path: Path, *, kernelbench: bool, swebench: bool, template: str | None = None):
    cfg = make_config(tmp_path)
    cfg.kernelbench.enabled, cfg.swebench.enabled = kernelbench, swebench
    cfg.kernelbench.concurrency, cfg.kernelbench.poll_s = 4, 0
    cfg.swebench.poll_s = 0
    cfg.engine.tool_call_parser = "hermes"
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    if template is not None:
        (tmp_path / "snapshot" / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    return cfg, services


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_kernelbench_asks_for_every_level1_kernel_with_the_models_defaults_and_scores_fast_1(tmp_path: Path, monkeypatch) -> None:
    import tests.fake_vllm as fake_vllm

    seen: list[dict[str, Any]] = []
    real = fake_vllm.FakeVLLM.start

    def start(self):  # keep a handle on the fake server's recorded chat requests
        seen.append(self.chat_requests)  # type: ignore[arg-type]
        return real(self)

    monkeypatch.setattr(fake_vllm.FakeVLLM, "start", start)
    cfg, services = _setup(tmp_path, kernelbench=True, swebench=False, template=QWEN3_TEMPLATE)
    service = FakeKernelBench()
    services.kernelbench = service
    run_dir = _run_dir(tmp_path, execute(RunSpec(repo="example/tiny", quant_flag="bf16", thinking="on"), cfg, services))

    requests = [request for batch in seen for request in batch]
    assert len(requests) == 100
    # The model's own sampling: no sampling field. The run's thinking mode: on (owner decision 2026-10-29).
    assert not any(key in requests[0] for key in ("temperature", "top_p", "top_k"))
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
    # A thinking-on run is its own subject, beside the quant's thinking-off runs.
    assert run_dir.parent.name == "bf16-thinking" and _json(run_dir / "manifest.json")["thinking"] == "on"
    # No kernelbench.max_tokens: the context per stream minus room for the prompt.
    context = _json(run_dir / "manifest.json")["memory_plan"]["context"]
    assert requests[0]["max_tokens"] == context - 4096
    assert requests[0]["messages"][0] == {"role": "system", "content": "You are a helpful assistant"}

    suite = run_dir / "quality" / "kernelbench_l1"
    assert len(service.submitted[0]["kernels"]) == 100
    # The reasoning block is not taken for the kernel.
    assert service.submitted[0]["kernels"][0]["kernel_src"].startswith("class ModelNew")
    scores = _json(suite / "scores.json")
    # Correct and faster: problems divisible by 4 (25 of them), none of them #95. n = 99 with a baseline.
    assert scores["primary_metric"] == "fast_1" and scores["n"] == 99
    assert scores["primary_score"] == pytest.approx(25 / 99)
    assert scores["kernelbench"]["correctness_rate"] == pytest.approx(0.5)
    assert scores["context_per_stream"] is not None
    settings = _json(suite / "settings.json")
    assert settings["thinking"]["enable_thinking"] is True
    assert settings["thinking"]["template_default"]["enable_thinking"] is True

    summary = _json(run_dir / "summary.json")
    validate_instance(summary, "summary.schema.json")
    validate_instance(_json(run_dir / "manifest.json"), "manifest.schema.json")
    assert summary["quality"]["kernelbench_l1"]["score"] == pytest.approx(25 / 99)
    assert _json(run_dir / "manifest.json")["steps"]["kernelbench_results"]["status"] == "done"


def test_swebench_retries_tasks_that_lost_the_model_then_grades_and_scores(tmp_path: Path) -> None:
    cfg, services = _setup(tmp_path, kernelbench=False, swebench=True)
    tasks = ["django__a", "django__b", "sphinx__c", "sphinx__d"]
    runner = FakeRunner(tmp_path / "runner", tasks, resolved={"django__a", "sphinx__c", "sphinx__d"}, lost={"django__b"})
    services.swe_runner = runner
    run_dir = _run_dir(tmp_path, execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services))

    # Launch 1 lost the model on django__b; SMA dropped it and launched again; then the harness graded.
    assert [name for name, _ in runner.started] == ["agent", "agent", "grade"]
    overlay = runner.written[next(rel for rel in runner.written if rel.endswith("sma-agent.yaml"))]
    assert "step_limit: 250" in overlay and "temperature" not in overlay
    # LiteLLM has no price for a local checkpoint; without this every task fails on its first call.
    assert "cost_tracking: ignore_errors" in overlay
    assert f"127.0.0.1:{cfg.engine.port}/v1" in overlay

    suite = run_dir / "quality" / "swebench_verified_mini"
    scores = _json(suite / "scores.json")
    assert (scores["primary_score"], scores["n"], scores["error"]) == (0.75, 4, None)
    assert scores["exit_statuses"] == {"Submitted": 4}
    assert set(_json(suite / "preds.json")) == set(tasks)
    with gzip.open(suite / "trajectories.jsonl.gz", "rt") as handle:
        assert len(handle.read().splitlines()) == 4
    manifest = _json(run_dir / "manifest.json")
    assert "quality/swebench_verified_mini/trajectories.jsonl.gz" in manifest["large_files"]
    assert FakeTunnel.stopped
    validate_instance(_json(run_dir / "summary.json"), "summary.schema.json")


def test_benchmarks_that_cannot_reach_their_machines_score_null_and_say_why(tmp_path: Path, monkeypatch) -> None:
    for name in ("KERNELBENCH_URL", "SWE_RUNNER_HOST", "SWE_RUNNER_USER", "SWE_RUNNER_SSH_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg, services = _setup(tmp_path, kernelbench=True, swebench=True)
    run_dir = _run_dir(tmp_path, execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services))
    summary = _json(run_dir / "summary.json")
    assert summary["quality"]["kernelbench_l1"]["score"] is None
    assert "KERNELBENCH_URL is not set" in summary["quality"]["kernelbench_l1"]["error"]
    assert "SWE_RUNNER_HOST" in summary["quality"]["swebench_verified_mini"]["error"]
    warnings = _json(run_dir / "manifest.json")["warnings"]
    assert any("KERNELBENCH_URL" in warning for warning in warnings)


def test_every_committed_run_still_validates_against_the_schemas() -> None:
    manifests = sorted((ROOT / "runs").glob("*/*/*/*/manifest.json"))
    assert manifests
    for path in manifests:
        validate_instance(_json(path), "manifest.schema.json")
        validate_instance(_json(path.parent / "summary.json"), "summary.schema.json")


def test_a_started_command_that_changes_folder_still_writes_its_log_and_exit_status(tmp_path: Path) -> None:
    # Grading runs `cd eval && ... && cp report ../runs/<id>/report.json`. Without a subshell the cd moved the
    # log and exit paths into eval/, so the copy and the exit status were lost though the harness had graded.
    import subprocess
    import time

    from sma.swebench import start_script

    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "report.json").write_text("{}")
    command = "cd eval && echo graded && cp -- report.json ../runs/r1/report.json"
    subprocess.run(["sh", "-c", start_script("runs/r1", "grade", command)], cwd=tmp_path, check=True, timeout=30)
    done = tmp_path / "runs" / "r1" / "grade.exit"
    deadline = time.monotonic() + 20
    while not done.is_file() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert done.read_text().strip() == "0"
    assert "graded" in (tmp_path / "runs" / "r1" / "grade.log").read_text()
    assert (tmp_path / "runs" / "r1" / "report.json").is_file()
