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
    assert json.loads((suite_dir / "command.json").read_text())["patch"] is None and "PYTHONPATH=/sma-patch" not in argv
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
        ({"name": "x", "image": "img", "task": "bfclv3_ast", "limit": 5}, "cannot run a share"),
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
    # The answer limit is quality.max_tokens (the harness's 16), and never more than the stream's context less
    # 4,096 for the prompt; one answer per problem, in the window.
    assert params["max_new_tokens"] == min(cfg.quality.max_tokens, context - 4096) == 16
    assert params["request_timeout"] == int(cfg.quality.request_timeout_s)
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
    # Grading gets LiveCodeBench's own per-test limit, not the request timeout (src/sma/patches/livecodebench).
    command = json.loads((run_dir / "quality" / "livecodebench" / "command.json").read_text())
    assert command["patch"]["name"] == "livecodebench" and len(command["patch"]["files"]["sitecustomize.py"]) == 64
    argv = command["argv"]
    assert "PYTHONPATH=/sma-patch" in argv and any(arg.endswith("patches/livecodebench:/sma-patch:ro") for arg in argv)


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


def test_the_livecodebench_patch_grades_with_the_evaluation_timeout_and_keeps_the_request_timeout(tmp_path: Path) -> None:
    # livecodebench:26.03 grades each test with --timeout, the request timeout SMA sets to hours, and never
    # reads --evaluation_timeout (6 s). LFM3-1.7B's grading waited 4 hours on answers that never stop (2026-10-31).
    import subprocess
    import sys

    from sma.commands import eval_factory_patch

    runner = tmp_path / "livecodebench" / "runner"
    runner.mkdir(parents=True)
    (tmp_path / "livecodebench" / "__init__.py").write_text("")
    (runner / "__init__.py").write_text("")
    (runner / "scenario_router.py").write_text("def get_metrics(scenario, args, benchmark, results):\n    return args.timeout\n")
    (runner / "main.py").write_text("from livecodebench.runner.scenario_router import get_metrics\n")
    script = (
        "import argparse; from livecodebench.runner.main import get_metrics\n"
        "args = argparse.Namespace(timeout=14400, evaluation_timeout=6)\n"
        "print(get_metrics('codegeneration', args, [], []), args.timeout)\n"
    )
    patch = eval_factory_patch("nvcr.io/nvidia/eval-factory/livecodebench:26.03@sha256:00")
    assert patch is not None and eval_factory_patch("nvcr.io/nvidia/eval-factory/simple-evals:26.03") is None
    done = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env={"PYTHONPATH": f"{patch}:{tmp_path}", "PATH": "/usr/bin:/bin"},
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["6", "14400"]
    assert "SMA patch: grading with --evaluation_timeout 6 s per test" in done.stderr


def test_bfcl_with_a_limit_answers_an_even_share_of_every_category(tmp_path: Path) -> None:
    # Owner decision 2026-10-31: 2,000 of BFCL's 4,441. The container's own --limit keeps the first questions
    # by category name, so SMA's patch picks an even share of every category and the scorer keeps the same ones.
    cfg = make_config(tmp_path)
    cfg.quality.suites = [QualitySuite("bfcl_v3", 3, image="nvcr.io/nvidia/eval-factory/bfcl:26.03", task="bfclv3_ast", tools=True)]
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    execute(RunSpec(repo="example/tiny", quant_flag="bf16", tool_parser="hermes"), cfg, services)

    suite_dir = _run_dir(tmp_path) / "quality" / "bfcl_v3"
    command = json.loads((suite_dir / "command.json").read_text())
    assert command["limit"] == 3 and command["patch"]["name"] == "bfcl"
    assert set(command["patch"]["files"]) == {"sitecustomize.py", "sma_subset.py"}
    assert "SMA_QUESTION_LIMIT=3" in command["argv"] and "PYTHONPATH=/sma-patch" in command["argv"]
    # The fake container answered all 5; the scorer keeps the share: 2 of simple's 3, 1 of the other's 2.
    scores = json.loads((suite_dir / "scores.json").read_text())
    assert (scores["n"], scores["limit"]) == (3, 3)
    rows = [json.loads(line) for line in (suite_dir / "samples.jsonl").read_text().splitlines()]
    assert sorted(row["id"] for row in rows) == ["base_1", "simple_0", "simple_2"]


def test_the_even_share_takes_every_category_s_share_spread_through_it() -> None:
    from collections import Counter

    from sma.patches.bfcl.sma_subset import category_and_index, even_share

    sizes = {"simple": 400, "multiple": 200, "live_multiple": 1053, "live_parallel": 16, "multi_turn_base": 200, "javascript": 50}
    ids = [f"{name}_{index}" for name, size in sizes.items() for index in range(size)]
    ids[ids.index("live_multiple_5")] = "live_multiple_5-2-0"  # Live ids carry sub-indexes.
    keep = even_share(ids, 860)
    kept = Counter(category_and_index(item)[0] for item in keep)
    assert len(keep) == 860 and kept == {"simple": 179, "multiple": 90, "live_multiple": 472, "live_parallel": 7, "multi_turn_base": 90, "javascript": 22}
    assert keep == even_share(list(reversed(ids)), 860)  # The order SMA reads them in does not matter.
    simple = sorted(category_and_index(item)[1] for item in keep if item.startswith("simple_"))
    assert simple[0] < 5 and simple[-1] > 395  # Spread through the category, not its first questions.
    assert even_share(ids, 10_000) == set(ids)


def test_the_bfcl_patch_answers_the_share_and_scores_each_answer_against_its_own_question(tmp_path: Path) -> None:
    # The evaluator pairs answers with questions and ground truths by position, which only holds for a whole category.
    import subprocess
    import sys

    from sma.commands import eval_factory_patch, eval_factory_takes_limit

    for folder in ("bfcl", "bfcl/eval_checker"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / "__init__.py").write_text("")
    (tmp_path / "bfcl" / "_llm_response_generation.py").write_text(
        "def collect_test_cases(args, model_name, categories, paths, entries):\n    return list(entries)\n"
    )
    runner = "    return [e['id'] for e in prompt], [e['id'] for e in answers]\n"
    (tmp_path / "bfcl" / "eval_checker" / "eval_runner.py").write_text(
        "def ast_file_runner(handler, model_result, prompt, answers, language, category, name, score_dir):\n" + runner
        + "def multi_turn_runner(handler, model_result, prompt, answers, name, category, score_dir):\n" + runner
        + "def relevance_file_runner(handler, model_result, prompt, name, category, score_dir):\n    return [e['id'] for e in prompt]\n"
        + "def executable_file_runner(handler, model_result, prompt, name, category, score_dir):\n    return [e['id'] for e in prompt]\n"
    )  # fmt: skip
    script = (
        "from bfcl._llm_response_generation import collect_test_cases\n"
        "from bfcl.eval_checker import eval_runner\n"
        "entries = [{'id': f'simple_{i}'} for i in range(10)]\n"
        "kept = collect_test_cases(None, 'm', [], [], entries)\n"
        "print(' '.join(e['id'] for e in kept))\n"
        "print(eval_runner.ast_file_runner(None, kept, entries, entries, 'Python', 'simple', 'm', '.'))\n"
    )
    patch = eval_factory_patch("nvcr.io/nvidia/eval-factory/bfcl:26.03")
    assert patch is not None and eval_factory_takes_limit("nvcr.io/nvidia/eval-factory/bfcl:26.03")
    assert not eval_factory_takes_limit("nvcr.io/nvidia/eval-factory/livecodebench:26.03")
    env = {"PYTHONPATH": f"{patch}:{tmp_path}", "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60, env={**env, "SMA_QUESTION_LIMIT": "4"})
    assert done.returncode == 0, done.stderr
    kept, scored = done.stdout.splitlines()
    assert kept == "simple_1 simple_3 simple_6 simple_8"
    assert scored == str((kept.split(), kept.split()))
    # Without a limit the patch does nothing.
    plain = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)
    assert plain.stdout.splitlines()[0] == " ".join(f"simple_{i}" for i in range(10)) and "SMA patch" not in plain.stderr


def test_only_a_container_sma_can_share_takes_a_limit() -> None:
    validate_config(config_from_mapping({"quality": {"suites": [
        {"name": "bfcl_v3", "image": "nvcr.io/nvidia/eval-factory/bfcl:26.03", "task": "bfclv3_ast", "limit": 2000}]}}))
    with pytest.raises(SmaError, match="cannot run a share"):
        validate_config(config_from_mapping({"quality": {"suites": [
            {"name": "livecodebench", "image": "nvcr.io/nvidia/eval-factory/livecodebench:26.03", "task": "codegeneration_release_v6", "limit": 50}]}}))
