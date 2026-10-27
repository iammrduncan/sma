# AGENTS.md

SMA (Small Model Arena launcher) is a standalone Python CLI. It fetches one pinned Hugging Face
checkpoint, serves it on one DGX Spark in a pinned engine container, measures it, and writes a
resumable run folder. The format picks the engine: safetensors on vLLM (SGLang by flag or preflight),
GGUF on llama.cpp.

- [README.md](README.md) is the product contract: commands, run steps, run-record layout, caveats.
- [docs/arena-plan-alignment.md](docs/arena-plan-alignment.md) maps the arena plan onto SMA. It
  lists the owner-approved extensions and the deliberate deviations.
- [docs/first-spark-run.md](docs/first-spark-run.md) is the hardware checklist.
- [docs/serving-design.md](docs/serving-design.md) is the owner's engine, KV cache, context, and
  memory design: three engines by format, an 8-bit KV cache, the memory plan, deployment notes.
- [CONVENTIONS.md](CONVENTIONS.md) holds the detailed rules and their reasons. Read it before you
  change code.

## Boundaries

These are owner decisions. Do not change them without an explicit owner decision.

- SMA never quantizes or converts weights. It downloads GGUF files as published. A checkpoint that
  does not match the quant flag is refused. It is never converted or relabeled. The KV cache is
  always 8-bit: FP8 on vLLM and SGLang, Q8_0 on llama.cpp.
- Tensor parallelism is always 1. One Spark runs one model at a time, under `gpu.lock`.
- The engine is part of the subject. Never switch engines as a fallback. The engine is chosen before
  launch, by checkpoint format, by `--engine`, or because preflight shows that vLLM cannot serve the
  model. Never copy a measurement from one engine into another engine's record. The only fallback is
  the vLLM MoE backend chain.
- A run's planned memory stays under `memory.budget_gb`. Context per stream starts at 256K or the
  native maximum, never RoPE-stretched, and steps down to fit (docs/serving-design.md).
- `null` means not measured. Never invent a number, digest, backend, or percentile. Never fill a
  missing score with zero.
- No hostname and no Hugging Face token in a run record, log, or stored error. The token comes from
  the environment only.
- Do not claim hardware results. The test environment has no GPU. See the README section "What this
  environment did not verify".

## Commands

Run from the repository root.

```bash
uv sync --extra dev                                   # installs pytest; add --extra quality for NeMo Evaluator (nel)
uv run pytest                                         # whole suite (testpaths = tests, pythonpath = src)
uv run sma run Qwen/Qwen3-8B --bf16 --dry-run         # prints the plan; no GPU, download, or run folder
```

- No CI, linter, formatter, or type checker is configured. The test suite is the only automated
  check, and it runs only when someone runs it.
- Without uv, install the dependencies from `pyproject.toml` and run `python -m pytest`, or
  `PYTHONPATH=src python -m sma ...` for the CLI.
- `uv.lock` is not committed. `uv sync` creates one. Do not commit it unless the owner asks.
- `tests/test_pipeline.py::test_full_run_writes_a_valid_record_and_stops_the_server` fails on a
  machine whose hostname is also the user name: pytest's temp path (`pytest-of-<user>`) puts the
  hostname into the manifest. That failure comes from the machine, not from your change. Say so in
  your report. See the open decisions in CONVENTIONS.md.

## Layout

| Area | Modules in `src/sma/` |
| --- | --- |
| Entry points | `cli.py` (argparse; `SmaError` exits 2), `plan.py` (dry-run), `doctor.py` |
| Orchestration | `pipeline.py` (steps, markers, teardown), `queue.py` (filesystem queue, worker), `locks.py` |
| Commands | `commands.py`, `backends.py` (`EngineBackend` for vLLM and SGLang), `engines.py`, `runtime.py` (Docker) |
| Checkpoint | `fetch.py`, `quant.py`, `gguf.py` (GGUF header reader), `preflight.py`, `params.py`, `safetensors.py`, `hashing.py` |
| Memory plan | `memplan.py` (KV layout, context ladder, budget; pure), `runplan.py` (SMA's benchmarks as needs, serve plan) |
| Measurement | `probe.py`, `kl.py`, `corpus.py`, `speedmath.py`, `prom.py`, `telemetry.py`, `qualityout.py`, `compare.py`, `cleanup.py` |
| Records | `records.py`, `catalog.py`, `atomic.py`, `schemautil.py`, and `schemas/*.json` at the root |
| Shared helpers | `config.py`, `errors.py`, `proc.py`, `httpapi.py`, `retry.py`, `scrub.py`, `identity.py`, `logsetup.py` |

- `src/sma/data/` holds the pinned corpus and the SGLang bench dataset. `load_corpus` checks the
  corpus hash. Do not edit these files.
- `src/sma/nel_benchmarks/` holds graders that NeMo Evaluator loads by path (`sma:<name>` in a
  quality suite). They import `nemo_evaluator`, so SMA's own code never imports them.
- `tests/` uses in-process fakes: `fakes.py` (Docker, Hugging Face, NeMo Evaluator, llama-perplexity,
  bench, a GGUF repo), `fake_vllm.py` (an HTTP server), `gguf_writer.py` (small GGUF files), and
  `harness.py` (`make_config`, `make_services`). `tests/data/configs` holds real config.json files and
  `tests/data/llamacpp` real llama-perplexity output.
- `runs/`, `queue/`, and `.sma/` are output. They are git-ignored. Do not commit them.

## How to work

1. Understand the problem before you change anything. Read the README section and the module you
   will touch, and trace the flow end to end. Much of this code records an owner decision or a
   hardware finding. Check the README, the docs, and `git log` before you remove something.
2. Climb the ladder: does it need to exist, does it already exist here, does the standard library or
   an installed dependency do it. Then write the minimum code (CONVENTIONS.md `DES-1`).
3. Reuse the shared helpers. Do not call `subprocess`, `httpx`, or a plain file write for run records
   directly (`EXT-1`, `REC-1`).
4. Fix a bug at its root cause, in the shared function, and add a regression test that fails without
   the fix (`DES-12`, `TEST-1`).
5. When behavior changes, update the README in the same change. Update the docs too when they
   describe that behavior (`DOC-1`).
6. Before you report, run the suite. Say what ran, what passed, and what was not verified,
   especially anything that needs a GPU or a real engine image (`DOC-2`).

Write in plain, short sentences, like the README. An error message says what happened and what to do
next. Do not give time estimates.

Commit messages: one summary sentence in the imperative that ends with a period, then a body that
says what behavior changed.

The rules in CONVENTIONS.md adapt the engineering principles in
[GOOD_ENGINEERING_H.md](.agents/skills/engineering/reference/GOOD_ENGINEERING_H.md), an installed
skill tracked in `skills-lock.json`. CONVENTIONS.md does not depend on that file being present.
