"""Turn NeMo Evaluator and eval-factory container output into scores.json and samples.jsonl."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from sma.atomic import atomic_write_json, atomic_write_text
from sma.errors import SmaError
from sma.scrub import scrub_text

# nel writes the absolute output path into nel_run.json and checkpoint.json. Answers can quote anything.
_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".log", ".yaml", ".txt"}


def scrub_output(root: Path) -> None:
    """Remove the hostname and any token from every text file nel wrote under root."""
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        cleaned = scrub_text(text)
        if cleaned != text:
            atomic_write_text(path, cleaned)


def newest_bundle(attempt_dir: Path) -> Path | None:
    """nel names each result bundle eval-<UTC stamp>-<benchmark>.json."""
    found = sorted(attempt_dir.rglob("eval-*.json"), key=lambda path: path.name)
    return found[-1] if found else None


# nel's failure categories for a request whose model call failed, so it has no answer to score
# (nemo_evaluator/engine/eval_loop.py, "solve_failed"). Other categories, such as refusal, are answers.
_UNANSWERED = ("infra_error", "graceful")


def unanswered_count(failures: dict[str, Any]) -> int:
    """Requests with no answer, from nel's failure block. Without a category breakdown, every failure counts."""
    categories = failures.get("categories")
    if isinstance(categories, dict):
        total = 0
        for name in _UNANSWERED:
            entry = categories.get(name)
            count = entry.get("count") if isinstance(entry, dict) else None
            if isinstance(count, int) and not isinstance(count, bool):
                total += count
        return total
    failed = failures.get("total_failures")
    return failed if isinstance(failed, int) and not isinstance(failed, bool) else 0


def normalize_suite(
    suite_dir: Path, attempt_dir: Path, suite_name: str, *, bench: str, limit: int | None, metric: str | None = None
) -> dict[str, Any]:
    """Read one suite's nel bundle into scores.json, and its per-question rows into samples.jsonl.

    The headline is nel's pass@1 for right-or-wrong scorers, or its mean_reward when a scorer gives
    partial credit (MuSiQue's answer F1). metric names it in the record when the suite config does.
    """
    bundle_path = newest_bundle(attempt_dir)
    if bundle_path is None:
        raise SmaError(f"nemo-evaluator did not write a result bundle for {suite_name} under {attempt_dir}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    benchmark = bundle.get("benchmark") if isinstance(bundle, dict) else None
    if not isinstance(benchmark, dict):
        raise SmaError(f"{bundle_path} has no benchmark block")
    scores = benchmark.get("scores") if isinstance(benchmark.get("scores"), dict) else {}
    pass_at_1 = scores.get("pass@1") if isinstance(scores.get("pass@1"), dict) else {}
    # nel reports mean_reward instead when any reward is between 0 and 1.
    mean_reward = scores.get("mean_reward") if isinstance(scores.get("mean_reward"), dict) else {}
    headline = pass_at_1 or mean_reward
    if pass_at_1:
        lower, upper = pass_at_1.get("bootstrap_ci_lower"), pass_at_1.get("bootstrap_ci_upper")
    else:
        lower, upper = mean_reward.get("ci_lower"), mean_reward.get("ci_upper")
    failures = scores.get("failures") if isinstance(scores.get("failures"), dict) else {}
    shuffle = (bundle.get("config") or {}).get("shuffle") or {}
    failed = unanswered_count(failures)
    # nel counts a request that never got an answer (a timeout, a server error) as a wrong answer.
    # SMA never fills a missing score with zero, so a suite with unanswered requests has no score.
    # A refusal or another scored failure is an answer, and stays in the score.
    incomplete = failed > 0
    payload = {
        "scorer": f"nemo-evaluator {bundle.get('sdk_version') or 'unknown'}",
        "suite": suite_name,
        "bench": bench,
        "primary_metric": (metric or ("pass@1" if pass_at_1 else "mean_reward")) if headline else None,
        "primary_score": None if incomplete else headline.get("value"),
        "ci_lower": None if incomplete else lower,
        "ci_upper": None if incomplete else upper,
        "pass_at_1_failures_as_wrong": headline.get("value") if incomplete else None,
        "error": (
            f"{failed} of {benchmark.get('samples')} requests got no answer (nel's infra_error and graceful "
            "failures), and nel scores those as wrong. The score is null. See the trajectories for each failure."
            if incomplete
            else None
        ),
        "n": benchmark.get("samples"),
        "repeats": benchmark.get("repeats"),
        "failures": failures.get("total_failures"),
        "limit": limit,
        "shuffle_seed": shuffle.get("seed"),
        "dataset_size": shuffle.get("ds_full_size"),
        "categories": benchmark.get("categories"),
        "bundle": str(bundle_path.relative_to(suite_dir)),
    }
    rows: list[str] = []
    details: list[dict[str, Any]] = []
    results = bundle_path.parent / "results.jsonl"
    if results.is_file():
        for line in results.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                # The full conversation stays in nel's trajectories.jsonl.
                row.pop("trajectory", None)
                details.append(row.get("scoring_details") if isinstance(row.get("scoring_details"), dict) else {})
            rows.append(json.dumps(row, sort_keys=True))
    payload["scorer_shares"] = None if incomplete else boolean_shares(details)
    atomic_write_json(suite_dir / "scores.json", payload)
    atomic_write_text(suite_dir / "samples.jsonl", "".join(row + "\n" for row in rows))
    return payload


def boolean_shares(details: list[dict[str, Any]]) -> dict[str, float]:
    """For each true-or-false field every row's scorer reported, the share of rows where it is true.

    MuSiQue's exact match is one: its headline is answer F1, and nel reports exact match only as a breakdown.
    """
    if not details:
        return {}
    names = set.intersection(*(set(key for key, value in row.items() if isinstance(value, bool)) for row in details))
    return {name: sum(1 for row in details if row[name]) / len(details) for name in sorted(names)}


def wilson_interval(correct: int, total: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a proportion. None when there is nothing to score."""
    if total <= 0:
        return None, None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _harness_metrics(results_yml: Path) -> dict[str, float]:
    """Every numeric score in an eval-factory results.yml, as <group>/<score>."""
    raw = yaml.safe_load(results_yml.read_text(encoding="utf-8")) or {}
    found: dict[str, float] = {}
    for section in ("tasks", "groups"):
        for group, block in ((raw.get("results") or {}).get(section) or {}).items():
            for metric in ((block or {}).get("metrics") or {}).values():
                for name, score in ((metric or {}).get("scores") or {}).items():
                    value = (score or {}).get("value")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        found[f"{group}/{name}"] = float(value)
    return found


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _bfcl_items(attempt_dir: Path) -> list[dict[str, Any]]:
    """One row per BFCL question. Its score file lists only the failures after a summary line."""
    rows: list[dict[str, Any]] = []
    for result in sorted(attempt_dir.glob("result/*/BFCL_*_result.json")):
        category = result.name.removeprefix("BFCL_v3_").removesuffix("_result.json")
        score = attempt_dir / "score" / result.parent.name / result.name.replace("_result.json", "_score.json")
        if not score.is_file():
            raise SmaError(f"BFCL generated {category} but did not score it ({score.name} is missing)")
        failed = {str(row["id"]) for row in _jsonl(score)[1:] if "id" in row}
        for entry in _jsonl(result):
            item = str(entry["id"])
            rows.append({"category": category, "id": item, "correct": item not in failed})
    return rows


def normalize_eval_factory(suite_dir: Path, attempt_dir: Path, suite_name: str, *, image: str, task: str) -> dict[str, Any]:
    """Read an eval-factory container's output into scores.json and samples.jsonl.

    The score is accuracy over every question SMA can pair across runs, with a Wilson interval.
    The harness's own scores (BFCL's category groups, in percent) are kept as harness_metrics.
    """
    results_yml = attempt_dir / "results.yml"
    if not results_yml.is_file():
        raise SmaError(f"The {task} container did not write results.yml for {suite_name} under {attempt_dir}")
    rows = _bfcl_items(attempt_dir) if task.startswith("bfcl") else []
    correct = sum(1 for row in rows if row["correct"])
    lower, upper = wilson_interval(correct, len(rows))
    categories: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = categories.setdefault(row["category"], {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += int(row["correct"])
    payload = {
        "scorer": f"eval-factory {task}",
        "suite": suite_name,
        "image": image,
        "task": task,
        "primary_metric": "accuracy" if rows else None,
        "primary_score": correct / len(rows) if rows else None,
        "ci_lower": lower,
        "ci_upper": upper,
        "n": len(rows) or None,
        "limit": None,
        "categories": categories,
        "harness_metrics": _harness_metrics(results_yml),
    }
    atomic_write_json(suite_dir / "scores.json", payload)
    atomic_write_text(suite_dir / "samples.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return payload
