import json
import socket
from pathlib import Path

import pytest

from sma.errors import SmaError
from sma.qualityout import normalize_suite, scrub_output


def _bundle(path: Path, *, value: float, stamp: str = "20261027T045340Z", failures: int = 1) -> Path:
    # The shape nemo-evaluator 0.3.0 wrote for gpqa on a Spark.
    path.mkdir(parents=True, exist_ok=True)
    bundle = path / f"eval-{stamp}-gpqa.json"
    bundle.write_text(
        json.dumps(
            {
                "sdk_version": "0.3.0",
                "config": {"benchmark": "gpqa", "shuffle": {"seed": 0, "applied": True, "ds_full_size": 198}},
                "benchmark": {
                    "name": "gpqa",
                    "samples": 20,
                    "repeats": 1,
                    "scores": {
                        "pass@1": {"value": value, "bootstrap_ci_lower": 0.0, "bootstrap_ci_upper": 0.25},
                        "failures": {"total_steps": 20, "total_failures": failures},
                    },
                    "categories": {"Physics": {"mean_reward": 0.0, "n": 7}},
                },
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_a_bundle_becomes_scores_with_its_interval_and_rows_without_trajectories(tmp_path: Path) -> None:
    bench_dir = tmp_path / "attempt-1" / "gpqa"
    _bundle(bench_dir, value=0.1)
    (bench_dir / "results.jsonl").write_text(
        '{"problem_idx": 7, "reward": 1.0, "trajectory": [{"source": "user"}]}\n\n{"problem_idx": 3, "reward": 0.0}\n',
        encoding="utf-8",
    )

    scores = normalize_suite(tmp_path, tmp_path / "attempt-1", "gpqa_diamond", bench="gpqa", limit=20)

    assert scores["scorer"] == "nemo-evaluator 0.3.0"
    # One of 20 requests got no answer. nel scores it as wrong; SMA does not fill a missing score with zero.
    assert (scores["primary_metric"], scores["primary_score"], scores["ci_lower"]) == ("pass@1", None, None)
    assert (scores["pass_at_1_failures_as_wrong"], scores["n"], scores["failures"]) == (0.1, 20, 1)
    assert scores["error"].startswith("1 of 20 requests got no answer")
    assert (scores["shuffle_seed"], scores["dataset_size"], scores["limit"]) == (0, 198, 20)
    assert scores["bundle"] == "attempt-1/gpqa/eval-20261027T045340Z-gpqa.json"
    assert json.loads((tmp_path / "scores.json").read_text(encoding="utf-8")) == scores
    rows = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"problem_idx": 7, "reward": 1.0}, {"problem_idx": 3, "reward": 0.0}]


def test_only_the_named_attempt_is_read(tmp_path: Path) -> None:
    _bundle(tmp_path / "attempt-1" / "gpqa", value=0.9, failures=0)
    _bundle(tmp_path / "attempt-2" / "gpqa", value=0.2, failures=0)

    assert normalize_suite(tmp_path, tmp_path / "attempt-2", "gpqa_diamond", bench="gpqa", limit=20)["primary_score"] == 0.2


def test_an_attempt_without_a_bundle_fails_with_the_suite_name(tmp_path: Path) -> None:
    (tmp_path / "attempt-1").mkdir()
    with pytest.raises(SmaError, match="did not write a result bundle for gsm8k"):
        normalize_suite(tmp_path, tmp_path / "attempt-1", "gsm8k", bench="sma:gsm8k", limit=50)


def test_scrubbing_removes_the_hostname_from_every_text_file_nel_wrote(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: "secret-host")
    (tmp_path / "gpqa").mkdir()
    (tmp_path / "nel_run.json").write_text('{"output_dir": "/home/secret-host/runs/x"}', encoding="utf-8")
    (tmp_path / "gpqa" / "model_traffic.jsonl").write_text('{"note": "secret-host"}\n', encoding="utf-8")
    (tmp_path / "gpqa" / "clean.json").write_text('{"ok": true}', encoding="utf-8")

    scrub_output(tmp_path)
    scrub_output(tmp_path / "missing")  # a run that failed before nel wrote anything

    assert (tmp_path / "nel_run.json").read_text(encoding="utf-8") == '{"output_dir": "/home/<host>/runs/x"}'
    assert "secret-host" not in (tmp_path / "gpqa" / "model_traffic.jsonl").read_text(encoding="utf-8")
    assert (tmp_path / "gpqa" / "clean.json").read_text(encoding="utf-8") == '{"ok": true}'


def test_a_suite_where_every_request_was_answered_keeps_its_score_and_interval(tmp_path: Path) -> None:
    _bundle(tmp_path / "attempt-1" / "gpqa", value=0.3, failures=0)
    scores = normalize_suite(tmp_path, tmp_path / "attempt-1", "gpqa_diamond", bench="gpqa", limit=20)
    assert (scores["primary_score"], scores["ci_lower"], scores["ci_upper"]) == (0.3, 0.0, 0.25)
    assert (scores["error"], scores["pass_at_1_failures_as_wrong"]) == (None, None)


def test_a_refusal_is_an_answer_and_stays_in_the_score(tmp_path: Path) -> None:
    # The shape nel 0.3.0 wrote on a Spark: one IFEval reply categorized as a refusal.
    bundle = _bundle(tmp_path / "attempt-1" / "gpqa", value=0.6)
    data = json.loads(bundle.read_text(encoding="utf-8"))
    data["benchmark"]["scores"]["failures"] = {
        "total_steps": 10, "total_failures": 1, "failure_rate": 0.1,
        "categories": {"refusal": {"count": 1, "percentage": 0.1}},
    }
    bundle.write_text(json.dumps(data), encoding="utf-8")
    scores = normalize_suite(tmp_path, tmp_path / "attempt-1", "ifeval", bench="lm-eval://ifeval", limit=10)
    assert (scores["primary_score"], scores["error"]) == (0.6, None)


def test_timeouts_and_failed_calls_are_unanswered() -> None:
    from sma.qualityout import unanswered_count

    spark = {"total_failures": 8, "categories": {"infra_error": {"count": 8}}}
    assert unanswered_count(spark) == 8
    assert unanswered_count({"total_failures": 3, "categories": {"graceful": {"count": 2}, "refusal": {"count": 1}}}) == 2
    assert unanswered_count({"total_failures": 2}) == 2
