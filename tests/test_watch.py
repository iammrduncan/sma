"""`sma watch`: the screen it draws from run folders, the queue, and a plan file."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sma.catalog import Catalog
from sma.cli import main
from sma.errors import SmaError
from sma.queue import JobQueue
from sma.records import STEP_NAMES
from sma.watch import View, render
from tests.harness import make_config

NOW = datetime(2026, 10, 29, 3, 0, tzinfo=timezone.utc)
SUITES = [{"name": "ifeval", "limit": 50}, {"name": "mmlu_pro", "limit": 700}, {"name": "bfcl_v3", "limit": None}]


def _run(cfg, run_id: str, repo: str, quant: str, *, steps: dict[str, str], finished: bool = False) -> Path:
    run_dir = Path(cfg.runs_dir) / repo.replace("/", "_") / "llamacpp" / f"gguf-{quant}" / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "model_repo": repo,
        "model_slug": repo.replace("/", "_"),
        "quant_flag": "gguf",
        "quant_label": quant,
        "engine": {"name": "llamacpp"},
        "host_alias": "gpu-host-1",
        "timestamps": {"created": "2026-10-29T02:05:34Z", "started": "2026-10-29T02:05:34Z", "finished": "2026-10-29T02:50:00Z" if finished else None},
        "steps": {name: {"status": steps.get(name, "pending"), "started": "2026-10-29T02:10:00Z" if name in steps else None, "finished": None, "error": None} for name in STEP_NAMES},
        "config_snapshot": {"quality": {"suites": SUITES}, "browsecomp": {"limit": 100}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def _cell(screen: str, row: str, column: str) -> str:
    """One SERVING table cell: rows are labels of 20 characters after the indent, columns 12 wide."""
    serving = screen.split("SERVING")[1].split("FINISHED")[0].splitlines()
    columns = ["since start", "5 min", "1 hour", "24 hours"]
    line = next(text for text in serving if text.strip().startswith(row))
    index = columns.index(column)
    return line[22 + 12 * index : 22 + 12 * (index + 1)].strip()


def _screen(cfg, containers: list[str], plan: Path | None = None) -> str:
    Catalog(cfg.runs_path()).rebuild()
    return render(cfg, View(containers, (90 * 2**30, 121 * 2**30), NOW), plan=plan)


def test_watch_shows_the_current_runs_steps_scores_and_answered_counts(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16",
                   steps={"fetch": "done", "determinism": "done", "logprobs": "done", "quality": "running"})
    (run_dir / "determinism.json").write_text(json.dumps({"passed": False, "results": [
        {"batch_size": 1, "matches_batch_1": True}, {"batch_size": 8, "matches_batch_1": False}, {"batch_size": 64, "matches_batch_1": False}]}))
    (run_dir / "logprobs").mkdir()
    (run_dir / "logprobs" / "perplexity.json").write_text(json.dumps({"perplexity": 24.019}))
    ifeval = run_dir / "quality" / "ifeval"
    ifeval.mkdir(parents=True)
    (ifeval / "scores.json").write_text(json.dumps({"primary_score": 0.72, "primary_metric": "pass@1", "n": 50}))
    (ifeval / ".done").write_text("{}")
    answers = run_dir / "quality" / "mmlu_pro" / "attempt-1" / "mmlu_pro"
    answers.mkdir(parents=True)
    (answers / "inference_log.jsonl").write_text("{}\n{}\n{}\n")

    screen = _screen(cfg, ["sma-20261029T020534Z-aaaa-quality"])

    assert "CURRENT RUN  20261029T020534Z-aaaa  LiquidAI/LFM3-1.7B-GGUF BF16 on llamacpp" in screen
    run = screen.split("CURRENT RUN")[1].split("SERVING")[0]
    # One line per benchmark, in the order the run does them. No setup or bookkeeping steps.
    names = [line.split()[1] for line in run.splitlines()[1:] if line.strip()]
    assert names == ["determinism", "perplexity", "ifeval", "mmlu_pro", "bfcl_v3", "kernelbench", "swebench", "browsecomp", "speed"]
    # A failed check is marked failed, with the batch sizes that differ; scores are percentages.
    assert "✗ determinism      failed: batch 8, 64 differ from batch 1" in run and "perplexity       24.019" in run
    assert "ifeval           72.0% pass@1  n=50" in run
    assert "mmlu_pro         3/700 answered" in run
    assert "bfcl_v3          queued" in run and "fetch" not in run
    assert "memory 90.0 of 121.0 GiB free" in screen


def test_watch_prefers_the_run_with_a_container_over_a_record_left_running_by_a_crash(tmp_path: Path) -> None:
    # A Spark that went down mid-step leaves its run "running" in the record with no container.
    cfg = make_config(tmp_path)
    _run(cfg, "20261029T020534Z-live", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"quality": "running"})
    _run(cfg, "20261029T030000Z-dead", "LiquidAI/LFM3-VL-2B-GGUF", "BF16", steps={"serve_quality": "running"})

    screen = _screen(cfg, ["sma-20261029T020534Z-live-quality"])

    assert "CURRENT RUN  20261029T020534Z-live" in screen


def test_watch_plan_counts_only_runs_since_its_start(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    repo = "LiquidAI/LFM3-1.7B-GGUF"
    _run(cfg, "20261028T212249Z-old1", repo, "QAD-Q4_0", steps={"serve_quality": "failed"})
    _run(cfg, "20261029T020534Z-bf16", repo, "BF16", steps={"fetch": "done"}, finished=True)
    _run(cfg, "20261029T040000Z-q4aa", repo, "QAD-Q4_0", steps={"quality": "running"})
    plan = tmp_path / "plan.txt"
    plan.write_text(f"# order\nsince 2026-10-29T02:05:00Z\n{repo} BF16\n{repo} qad-q4_0\nLiquidAI/LFM3-VL-2B-GGUF BF16\n")

    screen = _screen(cfg, ["sma-20261029T040000Z-q4aa-quality"], plan)

    lines = screen.splitlines()
    assert any("LFM3-1.7B-GGUF BF16" in line and "done" in line and "20261029T020534Z-bf16" in line for line in lines)
    assert any("LFM3-1.7B-GGUF qad-q4_0" in line and "running" in line and "20261029T040000Z-q4aa" in line for line in lines)
    assert any("LFM3-VL-2B-GGUF BF16" in line and "queued" in line for line in lines)
    assert "old1" not in screen.split("FINISHED")[0]


def test_watch_lists_pending_queue_jobs(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    JobQueue(cfg.queue_path()).add([{"repo": "Qwen/Qwen3-8B", "quant": "bf16"}], kind="single")

    screen = _screen(cfg, [])

    assert "QUEUE  1 job in sma queue" in screen
    assert "Qwen/Qwen3-8B bf16" in screen and "pending" in screen
    assert "CURRENT RUN  none" in screen


def test_watch_refuses_a_plan_line_it_cannot_read(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    plan = tmp_path / "plan.txt"
    plan.write_text("LiquidAI/LFM3-1.7B-GGUF BF16 extra\n")
    with pytest.raises(SmaError, match="is not 'REPO QUANT' or 'since TIME'"):
        _screen(cfg, [], plan)


def test_sma_watch_once_prints_one_screen(tmp_path: Path, monkeypatch, capsys) -> None:
    class NoDocker:
        def __init__(self, cfg) -> None:
            pass

        def list_sma(self) -> list[str]:
            return []

    monkeypatch.setattr("sma.cli.DockerRuntime", NoDocker)
    # main() loads the repo's .env; the machines' facts would reach the lab. Tests stay offline.
    monkeypatch.setattr("sma.machines.gather", lambda cfg: [])
    config = tmp_path / "sma.yaml"
    config.write_text(f"runs_dir: {tmp_path / 'runs'}\nqueue:\n  dir: {tmp_path / 'queue'}\n", encoding="utf-8")

    assert main(["watch", "--once", "--config", str(config)]) == 0
    assert "CURRENT RUN  none" in capsys.readouterr().out
    assert main(["watch", "--once", "--interval", "0", "--config", str(config)]) == 2


def test_watch_clips_lines_to_the_terminal_width_and_keeps_colors_closed(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    JobQueue(cfg.queue_path()).add([{"repo": "Org/" + "M" * 120, "quant": "bf16"}], kind="single")
    Catalog(cfg.runs_path()).rebuild()
    for color in (False, True):
        screen = render(cfg, View([], None, NOW, color=color, width=60))
        plain = [re.sub(r"\033\[[0-9;]*m", "", line) for line in screen.splitlines()]
        assert max(len(line) for line in plain) <= 60
        # A color still open where a line is cut is closed.
        assert all(re.findall(r"\033\[[0-9;]*m", line)[-1:] in ([], ["\033[0m"]) for line in screen.splitlines())


def test_watch_masks_the_hostname_and_lab_machines_so_it_can_be_shared(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sma.scrub.host_names", lambda: ["sparkbox"])
    monkeypatch.setenv("SWE_RUNNER_HOST", "runner-lab-07")
    cfg = make_config(tmp_path)
    cfg.queue.dir = str(tmp_path / "sparkbox" / "queue")
    # A step on its second attempt shows the first attempt's error.
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"serve_quality": "running"})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["steps"]["serve_quality"]["error"] = "sparkbox could not reach runner-lab-07 over SSH"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    Catalog(cfg.runs_path()).rebuild()

    for color in (False, True):
        screen = render(cfg, View(["sma-20261029T020534Z-aaaa-quality"], None, NOW, color=color))
        assert "sparkbox" not in screen and "runner-lab-07" not in screen
        assert "<host> could not reach ${SWE_RUNNER_HOST}" in screen


def test_watch_serving_shows_each_windows_averages_and_dashes_what_the_engine_lacks(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"quality": "running"})
    end = NOW.timestamp()

    def row(age_s: float, prompt: float, predicted: float) -> str:
        values = {
            "llamacpp:prompt_tokens_total": prompt,
            "llamacpp:prompt_seconds_total": prompt / 2000,
            "llamacpp:tokens_predicted_total": predicted,
            "llamacpp:tokens_predicted_seconds_total": predicted / 40,
            "llamacpp:requests_processing": 8,
        }
        return json.dumps({"ts": end - age_s, "values": values})

    # Two samples 2 hours ago, then three in the last minute: 1 hour and 5 min see only the recent ones.
    rows = [row(7215, 0, 0), row(7200, 3000, 4500), row(45, 6000, 9000), row(30, 9000, 13500), row(15, 12000, 18000)]
    (run_dir / "engine_metrics.jsonl").write_text("\n".join(rows) + "\n")
    os.utime(run_dir / "engine_metrics.jsonl", (end, end))

    screen = _screen(cfg, ["sma-20261029T020534Z-aaaa-quality"])

    rows = ("prefill, per stream", "decode, per stream", "decode, all streams", "ITL ms")
    # Per stream: one request's rate while sharing. All streams: the server's total over the window.
    assert [_cell(screen, row, "5 min") for row in rows] == ["2,000", "40.0", "300", "25.0"]
    # Prompt tokens over wall-clock time are not a prefill speed, so there is no such row.
    assert "prefill, all streams" not in screen
    # The day also counts the two samples from 2 hours ago, at the same rates; the gap between is skipped.
    assert [_cell(screen, row, "24 hours") for row in rows] == ["2,000", "40.0", "300", "25.0"]
    assert _cell(screen, "prefill, per stream", "since start") == "—"  # No live read in this test.
    serving = screen.split("SERVING")[1].split("FINISHED")[0]
    assert "TTFT" not in serving and "covered" not in serving  # Only prefill, decode, and ITL.


def test_watch_serving_says_when_no_run_recorded_samples(tmp_path: Path) -> None:
    screen = _screen(make_config(tmp_path), [])
    assert "no engine counters in the last 24 hours, and no server up" in screen


LLAMA_SCRAPE = (
    "llamacpp:prompt_tokens_total {prompt}\nllamacpp:prompt_seconds_total {prompt_s}\n"
    "llamacpp:tokens_predicted_total {tokens}\nllamacpp:tokens_predicted_seconds_total {tokens_s}\n"
    "llamacpp:requests_processing 8\n"
)


def test_watch_reads_the_live_server_for_a_run_that_records_no_engine_counters(tmp_path: Path) -> None:
    # A run started before SMA recorded engine counters: sma watch reads its server itself.
    from datetime import timedelta

    from sma.watch import Live, read_live

    cfg = make_config(tmp_path)
    _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"quality": "running"})
    Catalog(cfg.runs_path()).rebuild()
    up = ["sma-20261029T020534Z-aaaa-quality"]
    reads = iter([
        LLAMA_SCRAPE.format(prompt=20000, prompt_s=10, tokens=30000, tokens_s=1000),
        LLAMA_SCRAPE.format(prompt=26000, prompt_s=13, tokens=36000, tokens_s=1150),
    ])
    asked: list[tuple[str, str]] = []

    def scrape(base_url: str, path: str) -> str:
        asked.append((base_url, path))
        return next(reads)

    started: list[str] = []

    def started_at(name: str):
        started.append(name)
        return NOW - timedelta(seconds=1000)

    live = Live(interval_s=5)
    read_live(cfg, live, up + ["sma-20261029T020534Z-aaaa-logprobs"], NOW - timedelta(seconds=15), scrape, started_at)
    read_live(cfg, live, up, NOW, scrape, started_at)
    assert started[-1] == "sma-20261029T020534Z-aaaa-quality"  # The server, not the logprobs container.
    screen = render(cfg, View(up, None, NOW, live=live))

    assert asked[0] == (f"http://127.0.0.1:{cfg.engine.port}", "/metrics")
    rows = ("prefill, per stream", "decode, per stream", "decode, all streams", "ITL ms")
    # Since start: the counters over the container's 1,000 s of uptime.
    assert [_cell(screen, row, "since start") for row in rows] == ["2,000", "31.3", "36.0", "31.9"]
    assert [_cell(screen, row, "5 min") for row in rows] == ["2,000", "40.0", "400", "25.0"]
    assert "this run records no counters" in screen


def test_watch_does_not_count_a_recording_run_twice(tmp_path: Path) -> None:
    from sma.watch import Live, read_live

    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"quality": "running"})
    (run_dir / "engine_metrics.jsonl").write_text("")
    Catalog(cfg.runs_path()).rebuild()
    live = Live()
    text = LLAMA_SCRAPE.format(prompt=20000, prompt_s=10, tokens=30000, tokens_s=1000)
    for _ in range(3):
        read_live(cfg, live, ["sma-20261029T020534Z-aaaa-quality"], NOW, lambda url, path: text)
    assert live.samples == [] and live.latest is not None
    # No container up: no read at all.
    read_live(cfg, live, [], NOW, lambda url, path: pytest.fail("read with no server up"))
    assert live.latest is None


def test_watch_shows_a_failed_benchmarks_error_on_its_own_line(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"swebench": "failed", "browsecomp_plus": "running"})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["steps"]["swebench"]["error"] = "The runner has no Docker images for the tasks."
    # A failed step makes the run failed; keep it current with a container up and started.
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    Catalog(cfg.runs_path()).rebuild()
    row = {"run_id": "20261029T020534Z-aaaa", "model_repo": "LiquidAI/LFM3-1.7B-GGUF", "quant": "BF16", "engine": "llamacpp",
           "run_dir": str(run_dir), "manifest": manifest}
    from sma.watch import _run_lines

    lines = _run_lines(row, NOW, lambda text, style: text)
    assert "  ✗ swebench         The runner has no Docker images for the tasks." in lines
    assert any(line.startswith("  ▶ browsecomp       0/100 answered") for line in lines)


def test_docker_start_time_is_read_from_inspect_with_nanoseconds(monkeypatch) -> None:
    from sma.config import default_config
    from sma.proc import CommandResult
    from sma.runtime import DockerRuntime

    answers = iter([
        CommandResult([], 0, "2026-10-29T02:07:36.38284748Z\n", ""),
        CommandResult([], 1, "", "Error: No such object: gone"),
    ])
    monkeypatch.setattr("sma.runtime.run_command", lambda argv, **kw: next(answers))
    runtime = DockerRuntime(default_config())
    assert runtime.started_at("sma-x-quality") == datetime(2026, 10, 29, 2, 7, 36, 382847, tzinfo=timezone.utc)
    assert runtime.started_at("gone") is None


def test_watch_counts_bfcl_answers_from_the_containers_result_files(tmp_path: Path) -> None:
    # BFCL runs in NVIDIA's eval-factory container, not nel: one JSON line per answered question, a file per category.
    from sma.watch import _suite_row

    run_dir = tmp_path / "run"
    results = run_dir / "quality" / "bfcl_v3" / "attempt-1" / "result" / "Org_Model"
    results.mkdir(parents=True)
    (results / "BFCL_v3_simple_result.json").write_text('{"id": "simple_0"}\n{"id": "simple_1"}\n')
    (results / "BFCL_v3_multi_turn_base_result.json").write_text('{"id": "multi_turn_base_0"}\n')
    name, state, detail = _suite_row(run_dir, {"name": "bfcl_v3", "limit": None, "image": "nvcr.io/nvidia/eval-factory/bfcl:26.03"}, NOW)
    assert (name, state) == ("bfcl_v3", "running")
    assert detail.startswith("3 answered · 2 categories")
    _name, _state, limited = _suite_row(run_dir, {"name": "bfcl_v3", "limit": 1000, "image": "x"}, NOW)
    assert limited.startswith("3/1,000 answered")


def test_a_suites_question_total_comes_from_its_config_even_for_an_older_runs_record(tmp_path: Path) -> None:
    from sma.config import config_from_mapping
    from sma.watch import _run_lines

    with pytest.raises(SmaError, match=r"suites\[0\].questions must be a whole number"):
        config_from_mapping({"quality": {"suites": [{"name": "bfcl_v3", "questions": 0}]}})
    cfg = config_from_mapping({"quality": {"suites": [{"name": "bfcl_v3", "image": "x", "task": "t", "questions": 4441}]}})
    assert cfg.quality.suites[0].questions == 4441

    run_dir = tmp_path / "run"
    results = run_dir / "quality" / "bfcl_v3" / "attempt-1" / "result" / "Org_Model"
    results.mkdir(parents=True)
    (results / "BFCL_v3_simple_result.json").write_text("{}\n" * 2872)
    # The run's own config was saved before the field existed.
    manifest = {"steps": {"quality": {"status": "running"}}, "config_snapshot": {"quality": {"suites": [{"name": "bfcl_v3", "image": "x", "limit": None}]}}}
    row = {"run_id": "r", "model_repo": "m", "quant": "q", "engine": "llamacpp", "run_dir": str(run_dir), "manifest": manifest}
    lines = _run_lines(row, NOW, lambda text, style: text, {"bfcl_v3": 4441})
    assert any("bfcl_v3          2,872/4,441 answered · 1 category" in line for line in lines)


def test_watch_finds_a_run_whose_catalog_folder_was_scrubbed(tmp_path: Path, monkeypatch) -> None:
    # On a machine whose user name is its hostname, the catalog stores /home/<host>/... for a run's folder.
    cfg = make_config(tmp_path)
    _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"quality": "running"})
    _run(cfg, "20261028T223359Z-dead", "LiquidAI/LFM3-VL-2B-GGUF", "BF16", steps={"serve_quality": "running"})
    Catalog(cfg.runs_path()).rebuild()
    original = Catalog.list_runs

    def scrubbed(self):
        return [{**row, "run_dir": "/home/<host>/runs/" + row["run_id"]} if row["run_id"].endswith("aaaa") else row for row in original(self)]

    monkeypatch.setattr(Catalog, "list_runs", scrubbed)
    screen = render(cfg, View(["sma-20261029T020534Z-aaaa-quality"], None, NOW))
    assert "CURRENT RUN  20261029T020534Z-aaaa" in screen


def test_watch_follows_a_resumed_run_whose_record_still_says_failed(tmp_path: Path) -> None:
    # The BF16 run's speed step failed; its resume redid SWE-bench while the record still said "failed".
    cfg = make_config(tmp_path)
    _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"speed": "failed", "swebench": "running"})
    _run(cfg, "20261028T223359Z-dead", "LiquidAI/LFM3-VL-2B-GGUF", "BF16", steps={"serve_quality": "running"})
    Catalog(cfg.runs_path()).rebuild()
    screen = render(cfg, View(["sma-20261029T020534Z-aaaa-quality"], None, NOW))
    assert "CURRENT RUN  20261029T020534Z-aaaa" in screen


def test_a_step_being_redone_shows_progress_not_its_last_attempts_error(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"swebench": "running"})
    folder = run_dir / "quality" / "swebench_verified_mini"
    folder.mkdir(parents=True)
    (folder / "scores.json").write_text(json.dumps({"primary_score": None, "error": "50 of 50 tasks got no answer"}))
    Catalog(cfg.runs_path()).rebuild()
    screen = render(cfg, View(["sma-20261029T020534Z-aaaa-quality"], None, NOW))
    line = next(text for text in screen.splitlines() if " swebench " in text)
    assert "▶" in line and "agents on the runner" in line and "no answer" not in line



def test_scores_are_percentages_colored_from_red_to_green() -> None:
    from sma.watch import _painter, _percent

    plain = _painter(False)
    assert _percent(0.704, plain) == "70.4%"
    colored = _painter(True)
    assert _percent(0.01, colored) == "\033[38;5;196m1.0%\033[0m"  # red
    assert _percent(0.35, colored) == "\033[38;5;214m35.0%\033[0m"  # orange
    assert _percent(0.55, colored) == "\033[38;5;226m55.0%\033[0m"  # yellow
    assert _percent(1.0, colored) == "\033[38;5;46m100.0%\033[0m"  # green


def test_kernelbench_shows_its_correct_and_compiled_counts(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"kernelbench_results": "done"})
    folder = run_dir / "quality" / "kernelbench_l1"
    folder.mkdir(parents=True)
    (folder / "scores.json").write_text(json.dumps({"primary_score": 1 / 99, "primary_metric": "fast_1", "n": 99,
                                                    "kernelbench": {"total": 100, "compiled": 11, "correct": 5}}))
    Catalog(cfg.runs_path()).rebuild()
    screen = render(cfg, View(["sma-20261029T020534Z-aaaa-quality"], None, NOW))
    assert "kernelbench      1.0% fast_1  n=99  (5 of 100 correct, 11 compiled)" in screen
