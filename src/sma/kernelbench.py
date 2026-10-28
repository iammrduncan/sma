"""KernelBench level 1, the offline parts: problems, prompt, kernel extraction, and scoring.

SMA ships KernelBench's 100 level-1 problems and its one-shot CUDA fp32 prompt verbatim, from the
repository at COMMIT, with a sha256 manifest (data/kernelbench). The prompt goes to the model as
chat messages, the form KernelBench uses for API providers (owner decision). A remote service
compiles, checks, and times the kernels. Scoring follows KernelBench's
scripts/benchmark_eval_analysis.py and src/kernelbench/score.py, except that SMA writes None where
KernelBench writes 0 for "nothing to measure".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sma.errors import SmaError

COMMIT = "423217d9fda91e0c2d67e4a43bf62f96f6d104f1"
SYSTEM_PROMPT = "You are a helpful assistant"
FAST_P = (0.0, 0.5, 0.8, 1.0, 1.5, 2.0)

_DATA = Path(__file__).resolve().parent / "data" / "kernelbench"
_PLACEHOLDER = "{ref_arch_src}"
_FENCE = re.compile(r"```(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Problem:
    level: int
    problem_id: int
    name: str  # The filename without .py, as in KernelBench's Hugging Face dataset.
    source: str


def _manifest() -> dict[str, Any]:
    return json.loads((_DATA / "manifest.json").read_text(encoding="utf-8"))


def _read_pinned(relative: str, expected_sha256: str) -> str:
    try:
        data = (_DATA / relative).read_bytes()
    except OSError as exc:
        raise SmaError(f"KernelBench file {relative} cannot be read ({exc.strerror}). Reinstall SMA.") from exc
    digest = hashlib.sha256(data).hexdigest()
    expected = expected_sha256.removeprefix("sha256:")
    if digest != expected:
        raise SmaError(
            f"KernelBench file {relative} has sha256 {digest}, but the manifest says {expected}. "
            "SMA refuses a problem or prompt whose bytes changed. Reinstall SMA."
        )
    return data.decode("utf-8")


def load_problems(level: int = 1) -> list[Problem]:
    """The pinned problems of one level, sorted by id. Each file is checked against the manifest."""
    files = _manifest().get(f"level{level}")
    if not isinstance(files, dict):
        raise SmaError(f"SMA does not ship KernelBench level {level}. It ships level 1 only.")
    problems = [
        Problem(
            level=level,
            # KernelBench's id is the number before the first underscore of the filename.
            problem_id=int(filename.split("_")[0]),
            name=filename.removesuffix(".py"),
            source=_read_pinned(f"level{level}/{filename}", digest),
        )
        for filename, digest in files.items()
    ]
    return sorted(problems, key=lambda problem: problem.problem_id)


def build_prompt(problem: Problem) -> str:
    """KernelBench's one-shot CUDA fp32 prompt for this problem.

    KernelBench fills the template with str.format, and the source is its only field. A plain
    replace gives the same bytes, and braces in the source stay as they are.
    """
    prompt = _manifest()["prompt"]
    template = _read_pinned(prompt["file"], prompt["sha256"])
    return template.replace(_PLACEHOLDER, problem.source)


def chat_messages(problem: Problem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(problem)},
    ]


def extract_kernel(content: str | None) -> str | None:
    """The kernel in a chat answer, or None when there is none to submit.

    None still counts as a failed problem. Reasoning is dropped first: text before the last
    </think> is not the answer, and an answer that ends inside <think> has no kernel. The rest is
    KernelBench's utils.extract_first_code(text, ["python", "cpp"]), unchanged, except that an
    empty block becomes None.
    """
    if content is None:
        return None
    last_close = content.rfind("</think>")
    if content.rfind("<think>") > last_close:
        return None
    if last_close != -1:
        content = content[last_close + len("</think>"):]
    match = _FENCE.search(content.strip())
    if match is None:
        return None
    code = match.group(1).strip()
    for tag in ("python", "cpp"):
        if code.startswith(tag):
            code = code[len(tag):].strip()
    return code or None


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _results_by_id(problems: list[Problem], results: list[dict]) -> dict[int, dict]:
    ids = {problem.problem_id for problem in problems}
    by_id: dict[int, dict] = {}
    for result in results:
        problem_id = result.get("problem_id") if isinstance(result, dict) else None
        if isinstance(problem_id, bool) or not isinstance(problem_id, int) or problem_id not in ids:
            raise SmaError(f"A KernelBench result names problem {problem_id!r}, which is not in this level.")
        if problem_id in by_id:
            raise SmaError(f"KernelBench returned two results for problem {problem_id}. SMA sends one kernel per problem.")
        by_id[problem_id] = result
    return by_id


def score_level(problems: list[Problem], results: list[dict], baseline_ids: set[int]) -> dict[str, Any]:
    """KernelBench's level metrics from the service's per-kernel results.

    A problem with no result counts as not compiled and not correct. Speedup is the stored
    baseline over the kernel's time (baseline_ref_runtime_ms / runtime_ms), as in KernelBench's
    analysis, not the reference timed next to the kernel. Like KernelBench, speedups and fast_p
    use only problems with a stored baseline (baseline_ids), and excessive speedups are counted,
    not excluded.
    """
    by_id = _results_by_id(problems, results)
    total = len(problems)
    compiled = sum(1 for result in by_id.values() if result.get("compiled") is True)
    correct_ids = [problem_id for problem_id, result in by_id.items() if result.get("correct") is True]
    with_baseline = len({problem.problem_id for problem in problems} & baseline_ids)

    speedups: list[float] = []
    for problem_id in correct_ids:
        if problem_id not in baseline_ids:
            continue
        baseline_ms = _positive(by_id[problem_id].get("baseline_ref_runtime_ms"))
        runtime_ms = _positive(by_id[problem_id].get("runtime_ms"))
        if baseline_ms is not None and runtime_ms is not None:
            speedups.append(baseline_ms / runtime_ms)

    fast_p = {
        str(p): sum(1 for speedup in speedups if speedup > p) / with_baseline if with_baseline else None
        for p in FAST_P
    }
    error_kinds = Counter(
        str(result["error_kind"]) for result in by_id.values() if result.get("error_kind") is not None
    )
    return {
        "total": total,
        "submitted": len(by_id),
        "compiled": compiled,
        "correct": len(correct_ids),
        "compilation_rate": compiled / total if total else None,
        "correctness_rate": len(correct_ids) / total if total else None,
        "with_baseline": with_baseline,
        "fast_p": fast_p,
        "geomean_speedup": statistics.geometric_mean(speedups) if speedups else None,
        "excessive_speedup": sum(1 for result in by_id.values() if result.get("excessive_speedup") is True),
        "error_kinds": dict(error_kinds),
    }
