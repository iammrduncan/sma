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


def test_a_partial_credit_scorer_reports_its_mean_with_the_suites_name_and_true_or_false_shares(tmp_path: Path) -> None:
    # nel reports mean_reward, not pass@1, when any reward is between 0 and 1 (MuSiQue's answer F1).
    bench_dir = tmp_path / "attempt-1" / "sma_musique"
    bench_dir.mkdir(parents=True)
    (bench_dir / "eval-20261028T000000Z-sma_musique.json").write_text(
        json.dumps(
            {
                "sdk_version": "0.3.0",
                "config": {"shuffle": {"seed": 0, "applied": True, "ds_full_size": 2417}},
                "benchmark": {
                    "samples": 2,
                    "repeats": 1,
                    "scores": {"mean_reward": {"value": 0.7, "ci_lower": 0.4, "ci_upper": 1.0}, "failures": {"total_failures": 0}},
                },
            }
        ),
        encoding="utf-8",
    )
    (bench_dir / "results.jsonl").write_text(
        '{"problem_idx": 1, "reward": 1.0, "scoring_details": {"answer_f1": 1.0, "exact_match": true, "extracted": "x"}}\n'
        '{"problem_idx": 2, "reward": 0.4, "scoring_details": {"answer_f1": 0.4, "exact_match": false, "extracted": "y"}}\n',
        encoding="utf-8",
    )
    scores = normalize_suite(tmp_path, tmp_path / "attempt-1", "musique", bench="sma:musique", limit=100, metric="answer_f1")
    assert (scores["primary_metric"], scores["primary_score"], scores["ci_lower"], scores["ci_upper"]) == ("answer_f1", 0.7, 0.4, 1.0)
    assert scores["scorer_shares"] == {"exact_match": 0.5}
    assert (scores["dataset_size"], scores["error"]) == (2417, None)


def test_ifeval_records_strict_and_loose_by_prompt_and_by_instruction(tmp_path: Path) -> None:
    # LFM3-1.7B's IFEval on a Spark scored 72% (strict, by prompt) while its card quotes a higher number;
    # the other three measures were in results.jsonl but not in scores.json. The rows' shape is nel 0.3.0's.
    _bundle(tmp_path / "attempt-1" / "lm-eval___ifeval", value=0.5, failures=0)
    rows = [
        {"prompt_level_strict_acc": True, "inst_level_strict_acc": [True, True], "prompt_level_loose_acc": True, "inst_level_loose_acc": [True, True]},
        {"prompt_level_strict_acc": False, "inst_level_strict_acc": [True, False, False], "prompt_level_loose_acc": True, "inst_level_loose_acc": [True, True, True]},
    ]
    (tmp_path / "attempt-1" / "lm-eval___ifeval" / "results.jsonl").write_text(
        "".join(
            json.dumps({"problem_idx": i, "reward": float(r["prompt_level_strict_acc"]), "scoring_details": {"method": "lm-eval://ifeval", "lm_eval_metrics": r}}) + "\n"
            for i, r in enumerate(rows)
        ),
        encoding="utf-8",
    )
    scores = normalize_suite(tmp_path, tmp_path / "attempt-1", "ifeval", bench="lm-eval://ifeval", limit=2)
    # By prompt: a share of the 2 prompts. By instruction: pooled over all 5 instructions.
    assert scores["measures"] == {
        "inst_level_loose_acc": 1.0,
        "inst_level_strict_acc": 0.6,
        "prompt_level_loose_acc": 1.0,
        "prompt_level_strict_acc": 0.5,
    }
    # A suite whose rows report no lm-eval metrics records none.
    _bundle(tmp_path / "gpqa" / "attempt-1" / "gpqa", value=0.3, failures=0)
    assert normalize_suite(tmp_path / "gpqa", tmp_path / "gpqa" / "attempt-1", "gpqa_diamond", bench="gpqa", limit=20)["measures"] is None


def test_answers_that_ran_out_of_room_are_counted_once_per_question(tmp_path: Path) -> None:
    # With quality.max_tokens 4096, LFM3-1.7B ran out on 213 of 700 MMLU-Pro answers and nothing said so.
    bench_dir = tmp_path / "attempt-1" / "gpqa"
    _bundle(bench_dir, value=0.3, failures=0)
    traffic = [
        {"problem_idx": 1, "repeat": 0, "status_code": 200, "finish_reason": "length"},
        {"problem_idx": 2, "repeat": 0, "status_code": 200, "finish_reason": "stop"},
        # A retry replaces its first attempt; a failed request is not an answer.
        {"problem_idx": 3, "repeat": 0, "status_code": 200, "finish_reason": "length"},
        {"problem_idx": 3, "repeat": 0, "status_code": 200, "finish_reason": "stop"},
        {"problem_idx": 4, "repeat": 0, "status_code": 500},
    ]
    (bench_dir / "model_traffic.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traffic), encoding="utf-8")
    scores = normalize_suite(tmp_path, tmp_path / "attempt-1", "gpqa_diamond", bench="gpqa", limit=20)
    assert scores["answers_cut_off"] == 1
    (bench_dir / "model_traffic.jsonl").unlink()
    assert normalize_suite(tmp_path, tmp_path / "attempt-1", "gpqa_diamond", bench="gpqa", limit=20)["answers_cut_off"] is None
