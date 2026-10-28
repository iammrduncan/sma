"""GSM8K with a numeric answer check, run by NeMo Evaluator (nel loads this file by path).

The prompt and dataset are nel's own built-in GSM8K. Only the grader is replaced, because both
graders in nemo-evaluator 0.3.0 mark answers wrongly. On a Spark trial (Qwen3-0.6B, 50 items):
- The built-in grader keeps everything after the first "The answer is", so
  "The answer is: 140" is read as ": 140". It scored 0 of 50.
- numeric_match strips trailing zeros from both sides, so "140." is wrong against 140 and "2500"
  is right against 25. It disagreed with a numeric comparison on 3 of 50.

This file imports nemo_evaluator, so only nel imports it. SMA passes its path.
"""

from __future__ import annotations

import math
import re

from nemo_evaluator import ScorerInput, benchmark, scorer

_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer after 'The answer is'.\n\n"
    "Problem: {question}"
)
# "answer is" or "answer:", so "to answer this" earlier in the reasoning is not a marker.
_MARKER = re.compile(r"(?i)\banswer\s*(?:is\b\s*:?|:)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def final_number(text: str | None) -> float | None:
    """The first number after the last answer marker, or else the last number in the text."""
    text = text or ""
    markers = list(_MARKER.finditer(text))
    if markers:
        after = _NUMBER.search(text, markers[-1].end())
        if after:
            return _to_float(after.group(0))
    numbers = _NUMBER.findall(text)
    return _to_float(numbers[-1]) if numbers else None


def _prepare(row, idx, rng):
    m = re.search(r"####\s*(.+)", row.get("answer", ""))
    answer = m.group(1).strip().replace(",", "") if m else row.get("answer", "")
    return {**row, "answer": answer}


@benchmark(
    name="sma_gsm8k",
    dataset="hf://openai/gsm8k?config=main&split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def sma_gsm8k_scorer(sample: ScorerInput) -> dict:
    got = final_number(sample.response)
    want = _to_float(str(sample.target))
    correct = got is not None and want is not None and math.isclose(got, want, rel_tol=0.0, abs_tol=1e-6)
    return {"correct": correct, "extracted": got}
