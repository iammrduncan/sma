# SMA

SMA (Small Model Arena launcher) is a standalone CLI. It takes one Hugging Face checkpoint and a quant flag, checks that the checkpoint already is that quant, serves it with vLLM, and writes a resumable run folder.

SMA does not quantize. It does not use tensor parallelism, including across machines. One Spark runs one model at a time.

```bash
sma run RecursiveSI/flint --bf16
sma run HawkBearPig/flint-FP8-NVFP4 --nvfp4
```

Both quants of one model should run on the same Spark. Queue them as one job, or use `run-pair`:

```bash
sma run-pair --bf16 RecursiveSI/flint --nvfp4 HawkBearPig/flint-FP8-NVFP4
sma queue add-pair --bf16 RecursiveSI/flint --nvfp4 HawkBearPig/flint-FP8-NVFP4
sma worker
```

`--mxfp4` is for checkpoints that already ship as MXFP4, such as gpt-oss. A native MXFP4 model is its own reference. There is no second quant and no KL unless you compare two runs yourself.

## Quickstart

Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) is the intended installer.

```bash
uv sync --extra quality
# quality is lm-eval. The launcher itself installs without it.
export HF_TOKEN=hf_...          # never put this in sma.yaml or a run folder
```

Edit `sma.yaml` if you need to. The host alias defaults to `gpu-host-1`. It must not be a real hostname.

```bash
sma doctor
sma run tiny-model --bf16 --dry-run
sma run org/model --bf16
sma status
sma list
sma resume <run-id>
sma compare <bf16-run-id> <nvfp4-run-id>
```

`sma doctor` checks Docker, the NGC image, `nvidia-smi`, that the GPU has no compute process, free disk, and that `HF_TOKEN` is set. It does not pull the image and it does not print the token.

A dry-run prints the plan and the exact commands. It does not download, start a container, or write a run folder.

## What a run does

Each step is idempotent. A `steps/<name>.done` marker is written only after the step finishes. `sma resume <run-id>` skips steps that have a marker. A crash mid-step redoes that step and does not redo finished ones.

1. **Fetch.** Resolve the repo to a commit SHA and download that snapshot with `huggingface_hub` (resumable). Hash the weight files. Count total and active parameters from safetensors shapes. Read `config.json` and `hf_quant_config.json` and refuse if they do not match `--bf16`, `--nvfp4`, or `--mxfp4`.
2. **Serve for quality.** NVIDIA NGC vLLM image, digest recorded from `docker image inspect`. `VLLM_BATCH_INVARIANT=1` when the architecture allows it, `--no-enable-flashinfer-autotune`, `--no-async-scheduling`, fixed seed, `--tensor-parallel-size 1`. Health check with a deadline.
3. **Determinism.** The same prompts at batch sizes 1, 8, and 64. A mismatch is recorded. The run continues.
4. **Logprobs.** A versioned corpus is cut into 2048-token windows and scored as raw token ids, with no chat template. Each position stores top-k (default 64), a tail bucket, and the true next-token logprob. Perplexity is `exp(-mean logprob)`.
5. **Quality.** lm-eval against the local OpenAI-compatible server, at the concurrency in `sma.yaml`. Raw samples and scores are saved. The scorer label is `lm-eval-harness`.
6. **Speed.** The server is restarted with batch invariance off. After a warm-up soak, `vllm bench serve` runs prefill-heavy and decode-heavy shapes at concurrency 1, then a concurrency sweep (1, 4, 8). Percentiles are ttft, tpot, itl, and e2el at 50, 90, and 99. `/metrics` is scraped before and after each cell.
7. **Telemetry.** A 1 Hz `nvidia-smi` sample of SM clock, temperature, power, and throttle reasons for the whole run. A thermal or power slowdown marks that speed cell invalid and retries it once after a cool-down. Invalid cells stay on disk and are left out of the headline tok/s.
8. **Teardown.** The container is removed on success, failure, Ctrl-C, and SIGTERM. The page cache is dropped when the process is allowed to write `/proc/sys/vm/drop_caches`. The run then waits until the GPU is cool, or until the idle timeout.

Derived headlines, labeled as derived:

- prefill tok/s = input length / TTFT at concurrency 1
- decode tok/s = 1 / TPOT at concurrency 1

`sma compare <reference> <candidate>` computes per-token KL with the reference top-k plus a tail bucket, top-1 agreement, and delta perplexity. It records k. It refuses when the tokenizer fingerprints differ, or when the token-id windows differ. Top-k KL is an approximation.

For MoE NVFP4 and MXFP4, SMA always passes `--moe-backend`. The default chain is `flashinfer_b12x`, then `marlin` if the server never becomes healthy. The backend that actually started is stored on the run. Auto-selection is not used. Dense NVFP4 also passes `--linear-backend` (default `flashinfer_b12x`). BF16 does not get the FP4 MoE backend.

## Queue

Up to three Sparks can run `sma worker` against one queue directory. The directory has to be on a single filesystem because a claim is an atomic `link`. Each worker holds the GPU lock and runs one whole job at a time. A pair job runs the reference quant, then NVFP4, on that same Spark, and passes the reference run into the KL step.

Spark 1 is the one you keep a worker on. The other two are opportunistic: start a worker when the machine is free. Nothing in the queue requires them. Jobs are never split across Sparks.

```bash
sma queue add org/model --bf16
sma queue list
sma queue reclaim          # return a claim whose heartbeat is stale
sma worker --once
```

`gpu.lock` stops two SMA processes on the same Spark from starting two models.

## Run record contract

```
runs/<model-slug>/<quant>/<run-id>/
  manifest.json                 schema version 1
  summary.json                  flat headline numbers; null means not measured
  steps/<step>.done             the resume marker
  fetch/result.json             SHA, hashes, params, detected quant
  determinism.json
  logprobs/dump.json            per-token top-k, tail, true-token logprob
  logprobs/corpus.json
  logprobs/perplexity.json
  logprobs/kl_vs_reference.json only when a reference run was supplied
  quality/<suite>/scores.json
  quality/<suite>/samples.jsonl
  speed/summary.json
  speed/cells/<cell>/attempt-<n>/bench.json
  speed/cells/<cell>/attempt-<n>/metrics.json
  telemetry.csv
  logs/
  cleanup.json
runs/catalog.sqlite              fast index
runs/index.json                  atomic snapshot of the catalog
runs/compares/<ref>__<cand>.json output of sma compare
```

Schemas: [`schemas/manifest.schema.json`](schemas/manifest.schema.json) and [`schemas/summary.schema.json`](schemas/summary.schema.json).

JSON is written to a temp file, fsynced, and renamed. A crash does not leave a half-written manifest or summary.

The manifest is the subject record: model repo, revision SHA, weight hashes, quant, parameter counts, engine image and digest, every vLLM flag that was passed, the env vars SMA set, the SMA git commit, the host alias, timestamps, and step status. It does not contain the machine hostname or the HF token. `summary.json` uses null for anything that was not measured. It does not fill missing scores with zero.

A future app should list runs from `runs/index.json` or `catalog.sqlite` and read `summary.json` for headlines. Per-sample quality output and per-token logprobs stay in the run folder.

## Config

`sma.yaml` in the working directory, or `--config`. Unknown keys are an error. `--quantization` and `--tensor-parallel-size` other than 1 are refused, including inside `engine.extra_args`.

The image tag is pinned in config (`nvcr.io/nvidia/vllm:26.05.post1-py3`, the current DGX Spark playbook tag). Set `engine.image_digest` when you have a digest. SMA records the digest `docker image inspect` returns. It does not invent one.

## Caveats

- Batch invariance is beta. vLLM documents it for newer datacenter GPUs. It is not verified here on sm_121 or with NVFP4. Gated delta-net models (Qwen3.5 / Qwen3.6) are rejected by that mode, so SMA does not set the flag for those architectures, records a warning, and continues at the fixed concurrency.
- Turning batch invariance on is not enough if async scheduling is still composing the next batch. Quality mode also passes `--no-async-scheduling`. That flag was not executed against a live vLLM in this repo.
- MoE backend auto-selection on SM121 can pick Marlin, and Marlin NVFP4 on SM121 has crash reports. SMA sets `--moe-backend` explicitly. If every backend in the chain fails, the step is failed, not skipped.
- Some Spark units throttle or power off under a sustained vLLM load. Throttled speed cells are marked invalid. They are not deleted.
- Top-k KL is an approximation. The tail is one bucket. If a reference top-k id is missing from the candidate's stored top-k, that mass is folded into the tail and the position is labeled approximate. Full-vocabulary KL is not computed.
- Tokenizers must match. `sma compare` refuses otherwise. Tokens per answer are not a shared unit across models.
- lm-eval scores are not the arena's pinned scorers. The default suite list is small and limited (ifeval, gsm8k, mmlu_pro, gpqa_diamond). Clear `limit` to run a whole task. The arena's full suite set is not the v0 default.
- v0 speed repeats default to 1, plus a warm-up. The arena plan's "at least 3 runs" is `speed.repeats`.
- Energy is the trapezoid of 1 Hz GPU power samples over valid speed windows (`scope: gpu`). It is not joules per correct answer and not a wall-meter reading. Null means it was not measured.
- The bandwidth ceiling in the summary uses the configured 273 GB/s spec constant and approximate bytes per parameter. It is not a measurement. NVFP4 scale overhead is not included.
- The determinism probe sends one completions request with N copies of each prompt. The client cannot force the scheduler to fuse those copies into one batch. `max_num_seqs` is at least 64 so a batch of 64 is allowed.
- Logprobs go through the running server's `/v1/completions` `prompt_logprobs` on raw token ids. A second offline `LLM()` process is not started, because the quality server already holds the GPU.
- SMA only produces vLLM rows. A result from another engine has to be labeled as that engine and is not comparable with these rows.
- SMA does not apply arena fidelity verdicts (broken / degraded / faithful / lossless). It records KL, top-1, and perplexity. Thresholds belong to the arena.

## What this environment did not verify

This workspace has no NVIDIA GPU and no DGX Spark. The tests use a fake HTTP server and a fake container runtime. They do not pull the NGC image, start vLLM, run lm-eval, or read a real GPU sensor.

Not verified on hardware:

- the NGC image tag actually starting on GB10 / sm_121 / ARM64 / CUDA 13
- `VLLM_BATCH_INVARIANT=1` and `--no-async-scheduling` on sm_121 or NVFP4
- `--moe-backend flashinfer_b12x` and the Marlin fallback on a real MoE checkpoint
- `--no-enable-flashinfer-autotune` and `--max-logprobs` against the pinned image
- `vllm bench serve` writing the JSON shape this parser accepts (several historical shapes are accepted; a new shape needs a parser update)
- lm-eval task names (`mmlu_pro`, `gpqa_diamond`) on the installed harness
- thermal throttle flags from `nvidia-smi` on a Spark, including the older `clocks_throttle_reasons` names
- dropping the page cache (the unit test only checks that a permission error is recorded)
- any benchmark number, quality score, tok/s, joules, or KL on a real model

No benchmark numbers are shipped with this repo.

## Development

```bash
uv sync --extra dev
uv run pytest
```

`sma run tiny-model --bf16 --dry-run` is covered by the tests and does not need a GPU.
