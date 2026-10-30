"""nel's model client waits quality.request_timeout_s for an answer, not nemo-evaluator 0.3.0's fixed 120 s.

On a Spark, LFM3-1.7B's 4,096-token MuSiQue answers took about 123 s. nel's client gave up at 120 s, retried
the same way, and 14 of 100 questions had no answer, so the suite's score was null.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sma.commands import nel_with_client_timeout, script_interpreter
from sma.pipeline import CommandEvaluator, RunSpec, execute
from tests.harness import make_config, make_services

# A stand-in for nemo-evaluator's model client: the field and the hook the launcher relies on.
FAKE_CLIENT = """
from dataclasses import dataclass

@dataclass
class ModelClient:
    base_url: str
    timeout: float = 120.0

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
"""

# A stand-in for nel's console script: builds a client the way nel's simple solver does, without a timeout.
FAKE_NEL = """
import sys
from nemo_evaluator.engine.model_client import ModelClient
print(sys.argv[1:], ModelClient(base_url="http://x/").timeout)
sys.exit(3)
"""


def _fake_nel(tmp_path: Path, client: str = FAKE_CLIENT) -> tuple[Path, dict[str, str]]:
    package = tmp_path / "site" / "nemo_evaluator" / "engine"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "model_client.py").write_text(client)
    script = tmp_path / "bin" / "nel"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n" + FAKE_NEL)
    script.chmod(0o755)
    return script, {"PYTHONPATH": str(tmp_path / "site")}


def test_the_launcher_raises_the_model_clients_timeout_and_keeps_nels_argv_and_exit_code(tmp_path: Path) -> None:
    script, env = _fake_nel(tmp_path)
    argv = nel_with_client_timeout(["nel", "eval", "run", "nel.yaml"], interpreter=sys.executable, script=str(script), timeout_s=900)
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 3
    assert result.stdout.strip() == "['eval', 'run', 'nel.yaml'] 900.0"


def test_a_nel_whose_client_changed_fails_instead_of_keeping_two_minutes(tmp_path: Path) -> None:
    script, env = _fake_nel(tmp_path, client="class Other:\n    pass\n")
    argv = nel_with_client_timeout(["nel", "eval"], interpreter=sys.executable, script=str(script), timeout_s=900)
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode != 0 and "ModelClient" in result.stderr


@pytest.mark.parametrize(
    ("first_line", "expected"),
    [
        ("#!/opt/venv/bin/python3", "/opt/venv/bin/python3"),
        ("#!/usr/bin/env python3", "python3"),
        ("#!/bin/sh", None),
        ("print('no shebang')", None),
    ],
)
def test_the_interpreter_comes_from_the_scripts_first_line(tmp_path: Path, first_line: str, expected: str | None) -> None:
    script = tmp_path / "nel"
    script.write_text(first_line + "\nimport sys\n")
    assert script_interpreter(script) == expected


def test_the_evaluator_refuses_a_nel_it_cannot_raise_the_timeout_for(tmp_path: Path, monkeypatch) -> None:
    from sma.errors import SmaError

    script = tmp_path / "nel"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SmaError, match="cannot raise nemo-evaluator's 120 s model-client timeout to quality.request_timeout_s"):
        CommandEvaluator().run(["nel", "eval"], tmp_path / "out", 60, tmp_path / "log.txt", client_timeout_s=900)


def test_every_nel_suite_gets_the_configured_request_timeout(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.quality.request_timeout_s = 777
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    nel_calls = [timeout for argv, timeout in zip(services.evaluator.calls, services.evaluator.client_timeouts) if argv[:1] == ["nel"]]
    assert nel_calls and set(nel_calls) == {777.0}
    command = next(Path(cfg.runs_dir).glob(f"*/*/*/{run_id}/quality/ifeval/command.json"))
    assert json.loads(command.read_text())["client_timeout_s"] == 777


def test_nel_is_found_beside_smas_python_when_it_is_not_on_path(tmp_path, monkeypatch) -> None:
    # 2026-10-30: `.venv/bin/sma worker` started without .venv/bin on PATH, and every nel suite failed.
    import sys

    from sma.commands import find_executable

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    nel = venv_bin / "nel"
    nel.write_text("#!/bin/sh\n")
    nel.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert find_executable("nel") == str(nel)
    assert find_executable("not-installed") is None
