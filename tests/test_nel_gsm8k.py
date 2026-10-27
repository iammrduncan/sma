"""SMA's GSM8K grader for NeMo Evaluator. The responses are real Qwen3-0.6B answers from a Spark trial."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("nemo_evaluator", reason="the quality extra is not installed")

from nemo_evaluator import ScorerInput  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "nel_benchmarks" / "gsm8k.py"
_spec = importlib.util.spec_from_file_location("sma_nel_gsm8k", _PATH)
gsm8k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gsm8k)


@pytest.mark.parametrize(
    ("response", "number"),
    [
        # nel's built-in grader read this as ": The answer is 140." and scored it wrong.
        ("90 + 50 = 140\n$$\n\nThe answer is: The answer is 140.", 140.0),
        ("The answer is: **10 inches**.", 10.0),
        ("The answer is:  \n**The answer is $22,500**.", 22500.0),
        ("Total: $1,250.50\n\nThe answer is $1,250.50", 1250.5),
        # No marker: the last number. A boxed answer counts.
        ("0.05 \\times 50,000 = 2,500\n$$\n\n### Final Answer:\n\n$$\n\\boxed{2500}\n$$", 2500.0),
        # "answer this" is not a marker, so the final number wins.
        ("To answer this, add 3 and 4.\n3 + 4 = 7", 7.0),
        ("No number at all.", None),
    ],
)
def test_the_final_number_is_the_one_the_model_gave_as_its_answer(response: str, number: float | None) -> None:
    assert gsm8k.final_number(response) == number


def test_a_correct_answer_with_a_trailing_period_counts_and_a_wrong_one_does_not() -> None:
    # nel's numeric_match got both wrong: it strips trailing zeros, so 140 became 14 and 2500 became 25.
    right = gsm8k.sma_gsm8k_scorer(ScorerInput(response="The answer is: The answer is 140.", target="140"))
    wrong = gsm8k.sma_gsm8k_scorer(ScorerInput(response="### Final Answer:\n$$\n\\boxed{2500}\n$$", target="25"))
    missing = gsm8k.sma_gsm8k_scorer(ScorerInput(response="I cannot tell.", target="25"))

    assert right == {"correct": True, "extracted": 140.0}
    assert wrong == {"correct": False, "extracted": 2500.0}
    assert missing == {"correct": False, "extracted": None}
