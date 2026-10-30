"""Suites that call tools (BFCL): the tool-call parser, the eval-factory container, and its scores."""

import json
import socket
from pathlib import Path

import pytest

from sma.config import QualitySuite, config_from_mapping, validate_config
from sma.errors import SmaError
from sma.pipeline import RunSpec, SimulatedKill, execute
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
    with pytest.raises(SimulatedKill):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16", tool_parser="hermes"), cfg, services)
    run_id = _run_dir(tmp_path).name

    with pytest.raises(SmaError, match="started with tool parser hermes, not llama3_json"):
        execute(
            RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_id, tool_parser="llama3_json"),
            cfg,
            services,
        )


def test_a_thinking_on_run_serves_quality_with_thinking_on_and_resumes_only_as_thinking_on(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    services.hooks.kill_step = "determinism"
    with pytest.raises(SimulatedKill):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16", thinking="on"), cfg, services)
    run_dir = _run_dir(tmp_path)
    assert run_dir.parent.name == "bf16-thinking"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    # The quality server takes thinking on as its default, for suites that cannot send it (BFCL).
    flags = manifest["engine"]["flags_quality"]
    assert json.loads(flags[flags.index("--default-chat-template-kwargs") + 1]) == {"enable_thinking": True}
    assert manifest["config_snapshot"]["quality"]["temperature"] is None

    with pytest.raises(SmaError, match="started with thinking on, not off"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_dir.name), cfg, services)


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
        hf_home=Path("/home/secret-host/.cache/sma/hf/eval-factory"),
    )
    assert "secret-host" not in json.dumps(scrub_obj(argv))
    # Datasets a container downloads are kept between runs (LiveCodeBench's is 4.5 GB); no token goes in.
    assert "/home/<host>/.cache/sma/hf/eval-factory:/hf-home" in scrub_obj(argv) and "HF_HOME=/hf-home" in argv
    assert not any("HF_TOKEN" in arg for arg in argv)
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


LIVECODEBENCH = QualitySuite(
    "livecodebench",
    image="nvcr.io/nvidia/eval-factory/livecodebench:26.03",
    task="codegeneration_release_v6",
    params={"extra": {"n_samples": 1, "start_date": "2024-08-01", "end_date": "2025-05-01"}},
    sampling=True,
)


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_livecodebench_runs_one_answer_per_problem_in_the_runs_thinking_mode(tmp_path: Path, thinking: str) -> None:
    # Owner decision 2026-10-29: SWE-bench is too slow; LiveCodeBench replaces it. Its container sends
    # temperature, top_p, and an answer limit itself, so SMA sets them for the run.
    cfg = make_config(tmp_path)
    cfg.quality.suites = [LIVECODEBENCH]
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    execute(RunSpec(repo="example/tiny", quant_flag="bf16", thinking=thinking), cfg, services)

    [run_config] = services.evaluator.eval_factory_configs
    params = run_config["config"]["params"]
    run_dir = _run_dir(tmp_path)
    context = json.loads((run_dir / "manifest.json").read_text())["memory_plan"]["benchmarks"]["quality"]
    # The answer may use the stream's context, less room for the prompt; one answer per problem, in the window.
    assert params["max_new_tokens"] == context - 4096 and params["request_timeout"] == int(cfg.quality.request_timeout_s)
    assert params["extra"] == {"n_samples": 1, "start_date": "2024-08-01", "end_date": "2025-05-01"}
    assert params["parallelism"] == cfg.quality.concurrency
    sampling = json.loads((run_dir / "quality" / "livecodebench" / "command.json").read_text())["sampling"]
    if thinking == "off":
        assert (params["temperature"], params["top_p"]) == (0.0, 1.0) and sampling["source"] == "thinking off: greedy"
    else:
        # vLLM applies the checkpoint's generation_config.json; this one sets neither, so 1.0, as vLLM does.
        assert (params["temperature"], params["top_p"]) == (1.0, 1.0) and "generation_config" in sampling["source"]

    scores = json.loads((run_dir / "quality" / "livecodebench" / "scores.json").read_text())
    assert (scores["primary_metric"], scores["n"]) == ("pass@1", 3)
    assert scores["primary_score"] == pytest.approx(2 / 3)
    assert scores["categories"] == {"easy": {"correct": 1, "total": 1}, "hard": {"correct": 0, "total": 1}, "medium": {"correct": 1, "total": 1}}
    rows = [json.loads(line) for line in (run_dir / "quality" / "livecodebench" / "samples.jsonl").read_text().splitlines()]
    assert {row["id"]: row["correct"] for row in rows} == {"abc390_a": True, "abc390_f": False, "3781": True}
    assert rows[0]["code"] and rows[0]["answer"]


def test_container_settings_belong_to_eval_factory_suites() -> None:
    with pytest.raises(SmaError, match="params and sampling are for an eval-factory container"):
        validate_config(config_from_mapping({"quality": {"suites": [{"name": "x", "sampling": True}]}}))


def test_teardown_stops_a_suite_container_left_running(tmp_path: Path) -> None:
    # 2026-10-30: a worker stopped mid-BFCL; teardown stopped the servers but not BFCL's container, then
    # failed with "Containers still present after teardown".
    cfg = make_config(tmp_path)
    cfg.quality.suites = [BFCL]
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    services.hooks.kill_step = "determinism"
    with pytest.raises(SimulatedKill):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16", tool_parser="hermes"), cfg, services)
    run_id = _run_dir(tmp_path).name
    leftover = f"{cfg.engine.container_prefix}-{run_id}-quality-bfcl_v3"
    runtime.running.add(leftover)  # as if the eval-factory container were still up

    from sma.pipeline import _load, _teardown_containers

    ctx = _load(RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_id), cfg, services)
    _teardown_containers(ctx)
    assert leftover not in runtime.list_sma()
