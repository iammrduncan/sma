"""SMA's MuSiQue benchmark for NeMo Evaluator: LongBench's prompt over MuSiQue's paragraphs, MuSiQue's own metric."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("nemo_evaluator", reason="the quality extra is not installed")

from nemo_evaluator import ScorerInput  # noqa: E402

_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "nel_benchmarks" / "musique.py"
_spec = importlib.util.spec_from_file_location("sma_nel_musique", _PATH)
musique = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(musique)

# The first row of the pinned dev file, shortened to three of its paragraphs.
ROW = {
    "id": "2hop__460946_294723",
    "paragraphs": [
        {"idx": 1, "title": "Steve Hillage", "paragraph_text": "Hillage's partner is Miquette Giraudy.", "is_supporting": True},
        {"idx": 0, "title": "Grant's First Stand", "paragraph_text": "The debut album by Grant Green.", "is_supporting": False},
        {"idx": 2, "title": "Green", "paragraph_text": "Green is an album by Steve Hillage.", "is_supporting": True},
    ],
    "question": "Who is the spouse of the Green performer?",
    "answer": "Miquette Giraudy",
    "answer_aliases": ["Miquette"],
}


def test_the_prompt_is_longbench_word_for_word_with_the_paragraphs_in_order() -> None:
    row = musique._prepare(ROW, 0, random.Random(0))
    prompt = musique._PROMPT.format(**row)
    instruction = "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    assert prompt == (
        f"{instruction}\n\nThe following are given passages.\n"
        "Passage 1:\nGrant's First Stand\nThe debut album by Grant Green.\n"
        "Passage 2:\nSteve Hillage\nHillage's partner is Miquette Giraudy.\n"
        "Passage 3:\nGreen\nGreen is an album by Steve Hillage.\n\n"
        f"{instruction}\n\nQuestion: Who is the spouse of the Green performer?\nAnswer:"
    )
    assert json.loads(row["golds"]) == ["Miquette Giraudy", "Miquette"]


@pytest.mark.parametrize(
    ("reply", "f1", "em"),
    [
        ("Miquette Giraudy", 1.0, True),
        ("The Miquette Giraudy.", 1.0, True),  # articles and punctuation are normalized away
        ("Miquette", 1.0, True),  # an alias
        ("Giraudy, Steve's partner", 0.4, False),  # best over golds: 1 shared token of 3 and 2
        ("Steve Hillage", 0.0, False),
        ("", 0.0, False),
    ],
)
def test_answers_score_with_musiques_f1_and_exact_match_over_aliases(reply: str, f1: float, em: bool) -> None:
    target = musique._prepare(ROW, 0, random.Random(0))["golds"]
    scored = musique.sma_musique_scorer(ScorerInput(response=reply, target=target))
    assert scored["answer_f1"] == pytest.approx(f1) and scored["correct"] == pytest.approx(f1)
    assert scored["exact_match"] is em
