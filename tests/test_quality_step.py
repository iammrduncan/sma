"""The quality step: nel config and grader in the suite folder, fresh attempt folders, no hostname."""

import json
import socket
from pathlib import Path

import pytest
import yaml

from sma.config import QualitySuite
from sma.errors import SmaError
from sma.pipeline import RunSpec, execute
from tests.harness import make_config, make_services


def _run_dir(tmp_path: Path) -> Path:
    return next((tmp_path / "runs").glob("*/*/*/*/"))


def test_a_packaged_grader_is_copied_into_the_suite_folder_and_named_relatively(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.quality.suites = [QualitySuite("gsm8k", 3, bench="sma:gsm8k")]
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)

    suite_dir = _run_dir(tmp_path) / "quality" / "gsm8k"
    shipped = Path(__file__).resolve().parents[1] / "src" / "sma" / "nel_benchmarks" / "gsm8k.py"
    assert (suite_dir / "gsm8k.py").read_text(encoding="utf-8") == shipped.read_text(encoding="utf-8")
    config = yaml.safe_load((suite_dir / "nel.yaml").read_text(encoding="utf-8"))
    assert config["benchmarks"][0]["name"] == "gsm8k.py"
    command = json.loads((suite_dir / "command.json").read_text(encoding="utf-8"))
    assert command["argv"] == ["nel", "eval", "run", "nel.yaml", "-o", "attempt-1"]
    scores = json.loads((suite_dir / "scores.json").read_text(encoding="utf-8"))
    assert (scores["bench"], scores["n"], scores["primary_score"]) == ("sma:gsm8k", 3, 0.5)
    assert len((suite_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_a_suite_that_failed_is_rerun_into_a_fresh_attempt_folder(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    write_output = services.evaluator.run
    calls: list[str] = []

    def fails_once_after_writing(argv, output_dir, timeout_s, log_path):
        calls.append(argv[-1])
        write_output(argv, output_dir, timeout_s, log_path)
        if len(calls) == 1:
            raise SmaError("nemo-evaluator: connection reset")

    services.evaluator.run = fails_once_after_writing
    with pytest.raises(SmaError, match="connection reset"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = _run_dir(tmp_path)
    execute(RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_dir.name), cfg, services)

    assert calls == ["attempt-1", "attempt-2"]
    scores = json.loads((run_dir / "quality" / "ifeval" / "scores.json").read_text(encoding="utf-8"))
    assert scores["bundle"].startswith("attempt-2/")


def test_no_file_under_quality_holds_the_hostname_even_when_nel_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: "secret-host")
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    write_output = services.evaluator.run

    def writes_then_fails(argv, output_dir, timeout_s, log_path):
        write_output(argv, output_dir, timeout_s, log_path)
        raise SmaError("nemo-evaluator failed")

    services.evaluator.run = writes_then_fails
    with pytest.raises(SmaError):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)

    quality = _run_dir(tmp_path) / "quality"
    written = [path for path in quality.rglob("*") if path.is_file()]
    assert any(path.name == "nel_run.json" for path in written)
    assert [path for path in written if "secret-host" in path.read_text(encoding="utf-8")] == []


def test_nel_runs_in_the_suite_folder_and_python_writes_no_bytecode_there(tmp_path: Path, monkeypatch) -> None:
    # Bytecode stores the absolute source path, and the run folder's path can hold the hostname.
    import sys

    from sma.pipeline import CommandEvaluator

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "nel"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import importlib.util, os, pathlib\n"
        "spec = importlib.util.spec_from_file_location('grader', 'grader.py')\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "pathlib.Path('seen.txt').write_text(os.environ.get('PYTHONDONTWRITEBYTECODE', ''))\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    suite_dir = tmp_path / "quality" / "gsm8k"
    suite_dir.mkdir(parents=True)
    (suite_dir / "grader.py").write_text("VALUE = 1\n", encoding="utf-8")

    CommandEvaluator().run(["nel", "eval", "run", "nel.yaml", "-o", "attempt-1"], suite_dir, 30.0, tmp_path / "log.txt")

    assert (suite_dir / "seen.txt").read_text(encoding="utf-8") == "1"
    assert not (suite_dir / "__pycache__").exists()
