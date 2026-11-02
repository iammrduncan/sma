# The BrowseComp-Plus judge

SMA's BrowseComp-Plus step grades each finished answer with an LLM judge (owner decision 2026-10-27: AI
judges are allowed). The judge runs on separate hardware behind an OpenAI-compatible endpoint, so it never
shares the Spark with the model under test. Scores are compared only within one judge.

| Since | Judge (`browsecomp.judge_model`) |
| --- | --- |
| 2026-11-01 | Qwen 3.8 Flash (FP4), `qwen3.8-flash-fp4` |
| 2026-10-28 to 2026-10-31 | Ornith 1.5 122B (Q4_K_M GGUF on llama.cpp), `ornith-1.5-122b` |

## How SMA calls it

The settings are `browsecomp.judge_*` in `sma.yaml`. The endpoint and key are `SMA_JUDGE_URL` and
`SMA_JUDGE_API_KEY` in the git-ignored root `.env`. A run record names the URL only as `${SMA_JUDGE_URL}` and
never holds the key.

One request per answer, one at a time, after the speed step:

```json
{
  "model": "qwen3.8-flash-fp4",
  "messages": [{"role": "user", "content": "<GRADER_TEMPLATE filled with the question, the model's answer and the correct answer>"}],
  "max_tokens": 4096,
  "temperature": 0.0,
  "chat_template_kwargs": {"enable_thinking": false},
  "cache_prompt": false
}
```

- **Prompt:** `GRADER_TEMPLATE` from BrowseComp-Plus `scripts_evaluation/evaluate_run.py` at commit
  `046949032b0328319cc9a02663a759ec601d9402`, word for word, with no system prompt. It is `GRADER_TEMPLATE` in
  `src/sma/browsecomp.py`, and `tests/test_browsecomp.py` checks its SHA-256.
- **Thinking off,** as BrowseComp-Plus runs its own judge (Qwen3-32B with `enable_thinking: false`). Its
  judge samples at temperature 0.7, top-p 0.8, top-k 20; SMA decodes greedily for repeatable verdicts.
- **Not sent:** top-p, top-k, min-p, a seed, `stream`.
- **`max_tokens` 4096** is BrowseComp-Plus's own grader limit.
- **Reading the verdict:** BrowseComp-Plus's `parse_judge_response` (`parse_judge` in `src/sma/browsecomp.py`):
  the `correct: yes` or `correct: no` line, plain or bold. A reply without that line counts the answer as
  wrong, as BrowseComp-Plus counts it, and the run warns with the count.
- **What a run keeps:** each reply's text, reasoning, finish reason and usage, in the run's git-ignored bundle
  `quality/browsecomp_plus/trajectories.jsonl.gz`, whose SHA-256 the manifest records. `scores.json` names the
  judge and these settings. The questions and answers are never published as plain text.

## Checks

Each judge graded 99 answers from BrowseComp-Plus's published GPT-5 and o3 BM25 runs
(`Tevatron/browsecomp-plus-runs` at `3c2b136c`), for the first 99 of SMA's 100 questions, 4 at a time. A
"yes" whose extracted answer contains the key, or the reverse, after normalizing case and punctuation, was
taken as right. Every other verdict was read by hand.

| Judge, thinking off (SMA's setting) | Verdicts it gave | Wrong verdicts | Median time per answer |
| --- | --- | ---: | ---: |
| Qwen 3.8 Flash (2026-11-01) | 95 of 95 (4 published answers had no final answer) | 1 of 95, too strict | 7 s |
| Ornith 1.5 122B (2026-10-28) | 98 of 99 (one reply left out its `correct:` line) | 0 of the 97 checked | 13 s |

- Qwen 3.8 Flash's wrong verdict refused a correct title over one preposition. It accepted a full legal name
  for a common name.
- Ornith with thinking on gave the same verdict as with thinking off on all 97 answers both judged.
- **Repeatability:** three identical requests in a row gave byte-identical replies (Qwen 3.8 Flash on 3
  answers, Ornith on 5).
- **Prompt cache:** Ornith's server honored `cache_prompt: false`. Qwen 3.8 Flash's does not (repeats reused
  cached prompt tokens), and its replies were identical anyway.
- These answers come from strong models that follow BrowseComp-Plus's answer format. Small models' answers are
  messier, and Qwen 3.8 Flash has not been checked on them.
