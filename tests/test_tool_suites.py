"""Suites that call tools (BFCL): the tool-call parser, the eval-factory container, and its scores."""

import json
import socket
from pathlib import Path

import pytest

from sma.config import QualitySuite, config_from_mapping, validate_config
from sma.errors import SmaError
from sma.pipeline import RunSpec, execute
from tests.harness import make_config, make_services

BFCL = QualitySuite("bfcl_v3", image="nvcr.io/nvidia/eval-factory/bfcl:26.03", task="bfclv3_ast", tools=True)


def _run_dir(tmp_path: Path) -> Path:
    return next((tmp_path / "runs").glob("*/*/*/*/"))


def test_bfcl_runs_its_container_on_the_host_network_and_keeps_one_row_per_question(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.quality.suites = [BFCL]
    services, _fetcher, runtime = make_services(tmp_path, cfg)

    execute(RunSpec(repo="example/tiny", quant_flag="bf16", tool_parser="hermes"), cfg, services)

    quality_flags = runtime.specs[0].serve_argv
    assert quality_flags[quality_flags.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-auto-tool-choice" in quality_flags
    assert json.loads(quality_flags[quality_flags.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": False}
    speed_flags = runtime.specs[-1].serve_argv
    assert "--tool-call-parser" not in speed_flags and "--default-chat-template-kwargs" not in speed_flags

    run_dir = _run_dir(tmp_path)
    assert json.loads((run_dir / "manifest.json").read_text())["engine"]["tool_call_parser"] == "hermes"
    suite_dir = run_dir / "quality" / "bfcl_v3"
    argv = json.loads((suite_dir / "command.json").read_text())["argv"]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--network") + 1] == "host"
    assert "NEMO_EVALUATOR_TELEMETRY_LEVEL=0" in argv
    assert f"nvcr.io/nvidia/eval-factory/bfcl:26.03@{runtime.digest}" in argv
    scores = json.loads((suite_dir / "scores.json").read_text())
    assert (scores["primary_metric"], scores["primary_score"], scores["n"]) == ("accuracy", 0.8, 5)
    assert scores["categories"] == {"multi_turn_base": {"correct": 2, "total": 2}, "simple": {"correct": 2, "total": 3}}
    assert scores["harness_metrics"] == {"bfcl/multi_turn_overall_acc": 100.0}
    rows = [json.loads(line) for line in (suite_dir / "samples.jsonl").read_text().splitlines()]
    assert {row["id"]: row["correct"] for row in rows} == {
        "base_0": True,
        "base_1": True,
        "simple_0": True,
        "simple_1": False,
        "simple_2": True,
    }
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["quality"]["bfcl_v3"]["score"] == 0.8
    assert summary["quality"]["bfcl_v3"]["scorer"] == "eval-factory bfclv3_ast"


def test_a_tool_suite_without_a_parser_records_a_null_score_and_the_run_goes_on(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.quality.suites = [BFCL, QualitySuite("ifeval", 2, bench="lm-eval://ifeval")]
    services, _fetcher, runtime = make_services(tmp_path, cfg)

    execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)

    assert "--tool-call-parser" not in runtime.specs[0].serve_argv
    summary = json.loads((_run_dir(tmp_path) / "summary.json").read_text())
    assert summary["quality"]["bfcl_v3"]["score"] is None
    assert "Pass --tool-parser" in summary["quality"]["bfcl_v3"]["error"]
    assert summary["quality"]["ifeval"]["score"] == 0.5
    assert any("calls tools" in warning for warning in summary["warnings"])
    assert all(call[0] != "docker" for call in services.evaluator.calls)


def test_a_resumed_run_keeps_its_tool_parser(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    services.hooks.kill_step = "determinism"
    with pytest.raises(RuntimeError):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16", tool_parser="hermes"), cfg, services)
    run_id = _run_dir(tmp_path).name

    with pytest.raises(SmaError, match="started with tool parser hermes, not llama3_json"):
        execute(
            RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_id, tool_parser="llama3_json"),
            cfg,
            services,
        )


def test_the_command_record_has_no_hostname_from_the_mount_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: "secret-host")
    from sma.commands import eval_factory_argv
    from sma.scrub import scrub_obj

    argv = eval_factory_argv(
        name="sma-x-quality-bfcl_v3",
        image="img",
        attempt_dir=Path("/home/secret-host/runs/x/attempt-1"),
        config_path=Path("/home/secret-host/runs/x/run_config.yaml"),
        uid=1000,
        gid=1000,
    )
    assert "secret-host" not in json.dumps(scrub_obj(argv))
    assert "/home/<host>/runs/x/attempt-1:/results" in scrub_obj(argv)


@pytest.mark.parametrize(
    ("suite", "message"),
    [
        ({"name": "x", "image": "img"}, "image and task go together"),
        ({"name": "x", "image": "img", "task": "bfclv3_ast", "limit": 5}, "takes no bench or limit"),
    ],
)
def test_an_eval_factory_suite_is_checked_before_any_download(suite: dict, message: str) -> None:
    with pytest.raises(SmaError, match=message):
        validate_config(config_from_mapping({"quality": {"suites": [suite]}}))


def test_a_queued_job_carries_its_tool_parser(tmp_path: Path) -> None:
    from sma.queue import JobQueue

    job = JobQueue(tmp_path / "queue").add(
        [{"repo": "org/m", "quant": "bf16", "revision": None, "engine": "vllm"}], kind="single", tool_parser="hermes"
    )
    assert json.loads((tmp_path / "queue" / "pending" / f"{job['job_id']}.json").read_text())["tool_parser"] == "hermes"


def test_a_cold_image_pull_gets_the_long_pull_timeout(monkeypatch) -> None:
    # The eval-factory BFCL image is 2.2 GB. The short docker timeout (300 s) is for inspect and stop.
    from sma import runtime as runtime_module
    from sma.config import default_config
    from sma.proc import CommandResult

    calls: list[tuple[list[str], float]] = []
    pulled = {"done": False}

    def fake_run(argv, *, timeout_s, what, **_kwargs):
        calls.append((list(argv), timeout_s))
        if argv[:2] == ["docker", "pull"]:
            pulled["done"] = True
            return CommandResult(argv=list(argv), returncode=0, stdout="", stderr="")
        if pulled["done"]:
            return CommandResult(argv=list(argv), returncode=0, stdout='["img@sha256:' + "c" * 64 + '"]', stderr="")
        return CommandResult(argv=list(argv), returncode=1, stdout="", stderr="No such image")

    monkeypatch.setattr(runtime_module, "run_command", fake_run)
    cfg = default_config()
    digest = runtime_module.DockerRuntime(cfg).ensure_image("img:1")

    assert digest == "sha256:" + "c" * 64
    pull_timeouts = [timeout for argv, timeout in calls if argv[:2] == ["docker", "pull"]]
    assert pull_timeouts == [cfg.timeouts.image_pull_s]
