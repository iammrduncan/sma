"""MuSiQue-Ans: multi-hop questions over the paragraphs given with them, run by NeMo Evaluator (nel loads this file by path).

Owner decision 2026-10-28: a rule-scored companion to BrowseComp-Plus. Each question comes with its own
paragraphs (2 to 4 hold the answer's steps, the rest are distractors; 17 to 20 in all), and the model
answers from them in one reply. No judge.

- Data: MuSiQue v1.0 dev (2,417 answerable questions; the test answers are hidden), from a Hugging Face
  mirror pinned by revision and checked byte for byte against the official release's SHA-256.
- Prompt: LongBench's musique prompt, word for word (THUDM/LongBench config/dataset2prompt.json; also
  lm-eval's longbench task). Its passages are MuSiQue's own paragraphs in LongBench's format.
- Score: MuSiQue's metrics/answer.py (the SQuAD v2 code): answer F1, the headline, and exact match, each
  the best over the answer and its aliases.

This file imports nemo_evaluator, so only nel imports it. SMA passes its path.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import string

from nemo_evaluator import ScorerInput, benchmark, scorer

REPO, REVISION = "dgslibisey/MuSiQue", "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
FILE = "musique_ans_v1.0_dev.jsonl"
# The official musique_v1.0.zip's data/musique_ans_v1.0_dev.jsonl (StonyBrookNLP/musique download_data.sh).
SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"

_INSTRUCTION = "Answer the question based on the given passages. Only give me the answer and do not output any other words."
_PROMPT = (
    _INSTRUCTION + "\n\nThe following are given passages.\n{context}\n\n" + _INSTRUCTION + "\n\nQuestion: {question}\nAnswer:"
)


def _load() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    data = open(path, "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256:
        raise RuntimeError(f"{REPO}@{REVISION[:8]} {FILE} hashes to {got}, not the official release's {SHA256}.")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _prepare(row, idx, rng):
    paragraphs = sorted(row["paragraphs"], key=lambda p: p["idx"])
    context = "\n".join(f"Passage {i}:\n{p['title']}\n{p['paragraph_text']}" for i, p in enumerate(paragraphs, 1))
    golds = [row["answer"], *row.get("answer_aliases", [])]
    return {"id": row["id"], "question": row["question"], "context": context, "golds": json.dumps(golds, ensure_ascii=False)}


# metrics/answer.py
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def exact_match(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def f1(gold: str, pred: str) -> float:
    gold_toks, pred_toks = _tokens(gold), _tokens(pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if same == 0:
        return 0.0
    precision, recall = same / len(pred_toks), same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def best(metric, pred: str, golds: list[str]) -> float:
    return max(metric(gold, pred) for gold in golds)


@benchmark(name="sma_musique", dataset=_load, prompt=_PROMPT, target_field="golds", prepare_row=_prepare)
@scorer
def sma_musique_scorer(sample: ScorerInput) -> dict:
    pred = (sample.response or "").strip()
    golds = json.loads(sample.target)
    answer_f1 = best(f1, pred, golds)
    # nel's reward is "correct": here answer F1, so the suite's headline is mean answer F1 (nel's mean_reward).
    return {"correct": answer_f1, "answer_f1": answer_f1, "exact_match": bool(best(exact_match, pred, golds)), "extracted": pred[:200]}
