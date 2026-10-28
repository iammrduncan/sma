import shutil
from pathlib import Path

import pytest

from sma import kernelbench
from sma.errors import SmaError
from sma.kernelbench import (
    COMMIT,
    SYSTEM_PROMPT,
    Problem,
    build_prompt,
    chat_messages,
    extract_kernel,
    load_problems,
    score_level,
)

FIXTURES = Path(__file__).resolve().parent / "data" / "kernelbench"


def _tampered_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str) -> None:
    copy = tmp_path / "kernelbench"
    shutil.copytree(kernelbench._DATA, copy)
    with (copy / relative).open("a", encoding="utf-8") as handle:
        handle.write("\n# changed\n")
    monkeypatch.setattr(kernelbench, "_DATA", copy)


def test_level1_ships_all_100_problems_sorted_by_id() -> None:
    problems = load_problems()
    assert [problem.problem_id for problem in problems] == list(range(1, 101))
    assert problems[0].name == "1_Square_matrix_multiplication_"
    assert problems[94].name == "95_CrossEntropyLoss"
    assert all(problem.level == 1 and "class Model" in problem.source for problem in problems)


def test_shipped_files_match_the_manifest_and_commit() -> None:
    manifest = kernelbench._manifest()
    shipped = sorted(path.name for path in (kernelbench._DATA / "level1").iterdir())
    assert manifest["commit"] == COMMIT
    assert shipped == sorted(manifest["level1"])
    assert (kernelbench._DATA / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")


def test_prompt_for_level1_problem1_matches_kernelbench_byte_for_byte() -> None:
    expected = (FIXTURES / "prompt_level1_problem1.txt").read_text(encoding="utf-8")
    assert build_prompt(load_problems()[0]) == expected


def test_chat_messages_use_kernelbench_api_provider_form() -> None:
    problem = load_problems()[18]
    messages = chat_messages(problem)
    assert messages == [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": build_prompt(problem)},
    ]
    assert SYSTEM_PROMPT == "You are a helpful assistant"
    assert problem.source in messages[1]["content"]
    assert "{ref_arch_src}" not in messages[1]["content"]


def test_tampered_problem_file_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tampered_copy(tmp_path, monkeypatch, "level1/19_ReLU.py")
    with pytest.raises(SmaError, match="19_ReLU.py"):
        load_problems()


def test_tampered_prompt_template_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    problem = load_problems()[0]
    _tampered_copy(tmp_path, monkeypatch, "prompt_cuda_one_shot_fp32.txt")
    with pytest.raises(SmaError, match="prompt_cuda_one_shot_fp32.txt"):
        build_prompt(problem)


def test_level_sma_does_not_ship_is_refused() -> None:
    with pytest.raises(SmaError, match="level 2"):
        load_problems(2)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, None),
        ("```python\nimport torch\nclass ModelNew: ...\n```", "import torch\nclass ModelNew: ..."),
        ("Here it is:\n```cpp\nint x;\n```\nDone.", "int x;"),
        ("```\nimport torch\n```", "import torch"),
        ("```python\nfirst\n```\n```python\nsecond\n```", "first"),
        ("```py\nimport torch\n```", "py\nimport torch"),
        ("```python3\nimport torch\n```", "3\nimport torch"),
        ("I would fuse the ops.", None),
        ("```python\nimport torch\nclass ModelNew:", None),
        ("```python\n```", None),
        ("<think>try ```python\nbad\n```</think>\n```python\ngood\n```", "good"),
        ("plan ```python\nbad\n```</think>\n```python\ngood\n```", "good"),
        ("<think>a</think><think>b</think>\n```python\ngood\n```", "good"),
        ("<think>still planning ```python\nx\n```", None),
        ("<think>a</think>\n```python\nx\n```\n<think>more", None),
        ("<think>a</think>\nNo code after all.", None),
    ],
)
def test_extract_kernel_follows_kernelbench_extract_first_code(content: str | None, expected: str | None) -> None:
    assert extract_kernel(content) == expected


def _problems(count: int) -> list[Problem]:
    return [Problem(level=1, problem_id=i, name=f"{i}_P", source="") for i in range(1, count + 1)]


def _result(problem_id: int, **fields: object) -> dict:
    base = {
        "problem_id": problem_id,
        "compiled": True,
        "correct": True,
        "runtime_ms": None,
        "baseline_ref_runtime_ms": None,
        "excessive_speedup": False,
        "error_kind": None,
    }
    return {**base, **fields}


def test_score_level_matches_hand_computed_kernelbench_metrics() -> None:
    # Problem 4 has no result. Problem 5 has no stored baseline, like level 1's #95.
    results = [
        _result(1, runtime_ms=2.0, baseline_ref_runtime_ms=4.0, excessive_speedup=True),  # 2.0x
        _result(2, runtime_ms=4.0, baseline_ref_runtime_ms=2.0),  # 0.5x
        _result(3, correct=False, baseline_ref_runtime_ms=3.0, error_kind="incorrect"),
        _result(5, runtime_ms=1.0),
        _result(6, compiled=False, correct=False, error_kind="compile"),
    ]
    score = score_level(_problems(6), results, baseline_ids={1, 2, 3, 4, 6})
    assert score == {
        "total": 6,
        "submitted": 5,
        "compiled": 4,
        "correct": 3,
        "compilation_rate": pytest.approx(4 / 6),
        "correctness_rate": 0.5,
        "with_baseline": 5,
        # Strict >: 0.5x is not fast_0.5, and 2.0x is not fast_2.0.
        "fast_p": {"0.0": 0.4, "0.5": 0.2, "0.8": 0.2, "1.0": 0.2, "1.5": 0.2, "2.0": 0.0},
        "geomean_speedup": pytest.approx(1.0),
        "excessive_speedup": 1,
        "error_kinds": {"incorrect": 1, "compile": 1},
    }


def test_score_level_with_nothing_measured_is_none_not_zero() -> None:
    results = [_result(1, compiled=False, correct=False, error_kind="compile")]
    no_correct = score_level(_problems(2), results, baseline_ids={1, 2})
    assert no_correct["geomean_speedup"] is None
    assert no_correct["fast_p"]["0.0"] == 0.0
    no_baseline = score_level(_problems(2), [], baseline_ids=set())
    assert no_baseline["fast_p"] == {key: None for key in ("0.0", "0.5", "0.8", "1.0", "1.5", "2.0")}
    assert no_baseline["correctness_rate"] == 0.0
    assert score_level([], [], baseline_ids=set())["correctness_rate"] is None


def test_score_level_ignores_unusable_times() -> None:
    results = [
        _result(1, runtime_ms=0, baseline_ref_runtime_ms=4.0),
        _result(2, runtime_ms="2", baseline_ref_runtime_ms=4.0),
        _result(3, runtime_ms=1.0, baseline_ref_runtime_ms=True),
    ]
    score = score_level(_problems(3), results, baseline_ids={1, 2, 3})
    assert score["correct"] == 3
    assert score["geomean_speedup"] is None
    assert score["fast_p"]["0.0"] == 0.0


@pytest.mark.parametrize(
    "results",
    [
        [_result(7)],
        [_result(True)],
        [{"compiled": True}],
        ["not a dict"],
        [_result(1), _result(1)],
    ],
)
def test_score_level_refuses_results_that_do_not_match_the_problems(results: list) -> None:
    with pytest.raises(SmaError):
        score_level(_problems(2), results, baseline_ids={1, 2})
