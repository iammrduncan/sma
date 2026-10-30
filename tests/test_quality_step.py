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

    def fails_once_after_writing(argv, output_dir, timeout_s, log_path, client_timeout_s=None):
        calls.append(argv[-1])
        write_output(argv, output_dir, timeout_s, log_path)
        if len(calls) == 1:
            raise SmaError("nemo-evaluator: connection reset")

    services.evaluator.run = fails_once_after_writing
    with pytest.raises(SmaError, match="these failed: quality"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = _run_dir(tmp_path)
    assert "connection reset" in json.loads((run_dir / "quality" / "ifeval" / "scores.json").read_text())["error"]
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

    def writes_then_fails(argv, output_dir, timeout_s, log_path, client_timeout_s=None):
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


def test_rescore_brings_new_fields_to_an_old_record_from_its_saved_output(tmp_path: Path) -> None:
    # Owner, 2026-10-29: records change only through SMA. Fields SMA records now (answers_cut_off, IFEval's
    # measures) reach older runs by `sma rescore`, which asks nothing of a model.
    from sma.pipeline import rescore
    from sma.records import load_manifest
    from sma.schemautil import validate_instance

    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    suite_dir = run_dir / "quality" / cfg.quality.suites[0].name
    scores_path = suite_dir / "scores.json"
    old = json.loads(scores_path.read_text())
    # An old record: no cut-off count. nel's traffic log says one answer ran out of room.
    old.pop("answers_cut_off", None)
    scores_path.write_text(json.dumps(old))
    bench_dir = next(path for path in (suite_dir / "attempt-1").iterdir() if path.is_dir() and (path / "results.jsonl").is_file())
    (bench_dir / "model_traffic.jsonl").write_text(json.dumps({"problem_idx": 0, "repeat": 0, "status_code": 200, "finish_reason": "length"}) + "\n")

    assert cfg.quality.suites[0].name in rescore(cfg, run_id)

    new = json.loads(scores_path.read_text())
    assert new["answers_cut_off"] == 1
    assert (new["context_per_stream"], new["concurrency"]) == (old["context_per_stream"], old["concurrency"])
    manifest = load_manifest(run_dir)
    assert cfg.quality.suites[0].name in manifest["rescored"][-1]["suites"]
    validate_instance(manifest, "manifest.schema.json")
    summary = json.loads((run_dir / "summary.json").read_text())
    validate_instance(summary, "summary.schema.json")
    assert summary["quality"][cfg.quality.suites[0].name]["answers_cut_off"] == 1


def test_rescore_refuses_a_run_that_is_still_running(tmp_path: Path) -> None:
    from sma.pipeline import SimulatedKill, rescore

    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    services.hooks.kill_step = "quality"
    with pytest.raises(SimulatedKill):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_id = next((tmp_path / "runs").glob("*/*/*/*")).name
    with pytest.raises(SmaError, match="has a step running \\(quality\\)"):
        rescore(cfg, run_id)
