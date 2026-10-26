# SMA

SMA (Small Model Arena launcher) is a standalone CLI. It takes one Hugging Face checkpoint that is already in the quant you name, serves it, and writes a resumable run folder.

SMA does not quantize. It does not use tensor parallelism, including across machines. One Spark runs one model at a time. The host alias in the run record is not a hostname, and the HF token is read from the environment only.

vLLM is the default engine. SGLang is opt-in (`--engine sglang`, `engine.name` in `sma.yaml`, or `engine` on a queue job). The engine is part of the subject. A vLLM row and an SGLang row are not the same run.

```bash
sma run Qwen/Qwen3-8B --bf16
sma run RedHatAI/Qwen3-8B-NVFP4 --nvfp4
```

Both legs of a pair use the same engine and the same Spark. `run-pair` holds one GPU lock across both legs. A queue pair is one job, pinned to the alias that created the first run, so a different Spark cannot pick up leg 1.

```bash
sma run-pair --bf16 Qwen/Qwen3-8B --nvfp4 RedHatAI/Qwen3-8B-NVFP4
sma queue add-pair --bf16 Qwen/Qwen3-8B --nvfp4 RedHatAI/Qwen3-8B-NVFP4
sma worker
```

`--mxfp4` is for checkpoints that already ship as MXFP4, such as gpt-oss. A native MXFP4 model is its own reference. There is no second quant and no KL unless you compare two runs yourself.

`--mixed` is only for a checkpoint whose counted per-layer or per-group formats include both FP8 and NVFP4. The name `dgpp_mixed` is not enough. `HawkBearPig/flint-FP8-BF16` (dgpp_mixed, recipe fp8) and a dgpp_mixed config with no layer map are not `mixed`, and `--mixed` refuses them. A real FP8+NVFP4 map is still refused on stock vLLM and on stock SGLang: vLLM does not implement `dgpp_mixed`, and SGLang refuses that method too. vLLM does accept compressed-tensors configs that count both FP8 and NVFP4. SGLang refuses every compressed-tensors checkpoint. `--nvfp4` refuses mixed configs and tells you to pass `--mixed`. SMA does not flatten a mixed checkpoint into NVFP4.

## What does not start

These commands fail in preflight, before the weight snapshot is downloaded. A dry-run only prints the plan. It does not mean the engine can load the architecture.

`RecursiveSI/flint` is `FlintForCausalLM` (`model_type: flint`). The repo has no `auto_map`. It ships `sglang/flint_sglang-0.1.4-py3-none-any.whl` and no vLLM modeling code. SMA does not install that wheel.

```bash
sma run RecursiveSI/flint --bf16
# vLLM preflight fails. The error suggests --engine sglang.

sma run RecursiveSI/flint --bf16 --engine sglang
# Stock SGLang preflight also fails. FlintForCausalLM is not in the pinned image's registry.
```

`HawkBearPig/flint-FP8-NVFP4` is the same architecture with `quant_method: dgpp_mixed` (FP8 and NVFP4 expert projections, BF16 activations).

```bash
sma run HawkBearPig/flint-FP8-NVFP4 --nvfp4
# Refused. The error says to pass --mixed.

sma run HawkBearPig/flint-FP8-NVFP4 --mixed
# Refused on vLLM. Stock vLLM does not implement dgpp_mixed.

sma run HawkBearPig/flint-FP8-NVFP4 --mixed --engine sglang
# Refused. Stock SGLang does not implement dgpp_mixed. Flint is refused either way.
```

Preflight pulls the engine image, with its own timeout and progress, then queries that image's recorded digest rather than the moving tag. If the pull or the registry query fails, SMA warns and continues when the architecture is only absent from the built-in list. Flint is still refused. A registry query that succeeds and does not list the architecture fails the run. `trust_remote_code` plus a causal-LM `auto_map` is a bypass for other architectures, and it is off by default. SMA does not install a Flint wheel.

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
sma run Qwen/Qwen3-8B --bf16 --dry-run
sma run Qwen/Qwen3-8B --bf16
sma status
sma list
sma resume <run-id>
sma compare <bf16-run-id> <nvfp4-run-id>
```

`sma doctor` checks Docker, the configured image, `nvidia-smi`, that the GPU has no compute process, free disk, and that `HF_TOKEN` is set. It does not pull the image and it does not print the token.

A dry-run prints the plan and the exact commands. It does not download, start a container, or write a run folder. It also does not prove the checkpoint's architecture is registered.

## What a run does

Each step is idempotent. A `steps/<name>.done` marker is written only after the step finishes. `sma resume <run-id>` skips steps that have a marker, and only on the host alias that started the run. A crash mid-step redoes that step and does not redo finished ones.

1. **Fetch.** Resolve the repo to a commit SHA. Read `config.json` and, when present, `hf_quant_config.json`. Refuse a quant mismatch or an unsupported quant/engine pair before the weight snapshot. Pull the engine image and query its digest for the architecture registry. Flint is refused. If the image cannot be queried, an unknown architecture is a warning and the download continues. Then download that snapshot with `huggingface_hub` (resumable). Hash the weight files. Count total and active parameters from safetensors shapes. Flags are `--bf16`, `--nvfp4`, `--mxfp4`, and `--mixed`.
2. **Serve for quality.** The pinned image, then the digest recorded from `docker image inspect` on later starts. vLLM: `VLLM_BATCH_INVARIANT=1` when the architecture allows it, `--no-enable-flashinfer-autotune`, `--no-enable-prefix-caching`, `--no-async-scheduling`, fixed seed, `--tensor-parallel-size 1`. SGLang: `--attention-backend flashinfer`, `--tp-size 1`, `--enable-deterministic-inference`, and `--disable-radix-cache` because FlashInfer determinism does not keep the radix cache. Health check with a deadline.
3. **Determinism.** The same prompts at batch sizes 1, 8, and 64, on the quality server with prefix caching off. A mismatch is recorded. The run continues.
4. **Logprobs.** The pinned corpus is cut into 2048-token windows and scored as raw token ids, with no chat template. vLLM uses `/v1/completions` `prompt_logprobs`. SGLang uses `/generate` input-token top-k. Each position stores top-k (default 64), a tail bucket, and the true next-token logprob. Perplexity is `exp(-mean logprob)`. A field the server does not return stays null.
5. **Quality.** lm-eval against the local OpenAI-compatible server, at the concurrency in `sma.yaml`. Raw samples and scores are saved. The scorer label is `lm-eval-harness`.
6. **Speed.** The server is restarted with batch invariance off. A fresh speed server, including on resume, runs the warm-up soak again. Each attempt's bench command writes `speed/cells/<cell>/attempt-<n>/bench.json`, which is the file SMA reads. Both benches run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and both pass `--tokenizer /model` so the tokenizer is the pinned snapshot. vLLM keeps `--model` as the served name and uses `vllm bench serve` with ttft, tpot, itl, and e2el at 50, 90, and 99. SGLang passes `--model /model` as well, and uses `sglang.bench_serving` with `--dataset-name random`, a pinned local dataset at `/sma-run/speed/sglang-random-dataset.json` (so the container does not download ShareGPT), `--random-range-ratio 1.0`, `--seed`, and `--output-details`. `speed.num_prompts` above that file's row count is refused. A bench that finishes fewer prompts than `num_prompts`, or reports request errors, marks the cell invalid and retries it once, the same as a throttle. That tool's summary, in tag `v0.5.10.post1` (`python/sglang/bench_serving.py`), has median and p99 TTFT/TPOT, median/p95/p99 ITL, and median/p90/p99 end-to-end latency (`median_e2e_latency_ms`, `p90_e2e_latency_ms`). It does not emit p90 TTFT, TPOT, or ITL. SMA computes those from the per-request arrays when they are present, and otherwise stores null with a reason. The file is JSONL and is parsed from the last JSON object, so a retry cannot glue two objects into one invalid document. SGLang serve passes `--enable-metrics`, `--max-running-requests` (the same cap as vLLM `max_num_seqs`), and `--random-seed`. A failed `/metrics` scrape stores null counters and a reason. It does not fail the cell. These SGLang commands are what the code builds. They have not been executed against the pinned image.
7. **Telemetry.** A 1 Hz `nvidia-smi` sample of SM clock, temperature, power, and throttle reasons for the whole run. A real thermal or power slowdown marks that speed cell invalid and retries it once after a cool-down. `sw_power_cap` is always stored. It invalidates a cell only when `telemetry.sw_power_cap_invalidates` is true, because a GB10 near 140 W can show that flag in steady state. A cell is `unverified` unless it has about `duration / interval` samples (`telemetry.min_sample_ratio`, default 0.8) and its largest gap is no bigger than `telemetry.max_sample_gap_s`. `[N/A]` flags are unverified too. The sampler's failure counter is not used, because a later success resets it. Unverified cells are not retried. Invalid and unverified cells stay on disk and are left out of the headline tok/s.
8. **Teardown.** The container is removed on success, failure, Ctrl-C, and SIGTERM. The page cache is dropped when the process is allowed to write `/proc/sys/vm/drop_caches`. The run then waits until the GPU is cool, or until the idle timeout.

Headline tok/s uses `speed.derive_stat` (`p50` by default, or `p90` / `p99`):

- prefill tok/s = input length / TTFT at concurrency 1
- decode tok/s = 1 / TPOT at concurrency 1

If that percentile is absent, the derived number is null and the cell records why. SMA does not invent it.

`sma compare <reference> <candidate>` computes per-token KL with an epsilon floor (`1e-12`) instead of dropping positions. A probability that was actually zero is counted in `kl_infinite_positions` before the floor. `kl_floored_positions` counts every position that hit the floor, including those zeros and merely tiny positive masses. It also records top-1 agreement and delta perplexity, and it records k. It accepts a BF16 reference against a `mixed` candidate when the tokenizer, corpus, and engine match. It refuses a different engine unless you pass `--allow-cross-engine`, and that result is labeled `CROSS-ENGINE`. Different image digests are labeled `IMAGE DIGEST` and are not refused. It refuses when the tokenizer fingerprints differ, or when the token-id windows differ.

For MoE NVFP4, MXFP4, and mixed on vLLM, SMA always passes `--moe-backend`. The chain is `flashinfer_b12x`, then `marlin` if that server never becomes healthy. Quality and speed each record the backend that started. Auto-selection is not used. Dense NVFP4 also passes `--linear-backend` (default `flashinfer_b12x`). BF16 does not get the FP4 MoE backend. SGLang does not take that chain. Its MoE fields stay unset unless a log line names a backend. SMA does not invent one.

NVFP4 on vLLM includes ModelOpt W4A4 and compressed-tensors `nvfp4-pack-quantized` when the activation scheme is also 4-bit float (for example a group whose `input_activations.num_bits` is 4). A compressed-tensors group with fp4 weights and `input_activations: null` is `nvfp4_w4a16`. `--nvfp4` refuses it and will not label it `nvfp4`. SMA does not serve W4A16. SGLang NVFP4 is ModelOpt checkpoints. SGLang refuses compressed-tensors NVFP4, FP8, and mixed FP8/NVFP4, and the error says to use vLLM.

## Queue

Up to three Sparks can run `sma worker` against one queue directory. The directory has to be on a single filesystem because a claim is an atomic `link`. The worker takes the GPU lock before it claims. If the lock is busy it does not take a job. Each claimed job is heartbeated until it finishes. One failed model is marked failed (or returned to pending while `attempts` is under `queue.max_attempts`) and the worker continues.

A pair is one claim. The alias that created the first run is stored on the job and is not cleared when the claim is reclaimed. Leg 1 runs on that same Spark. `sma resume` refuses a run whose `host_alias` is not this process. If that Spark is gone, `sma queue unpin JOB` or `sma queue reassign JOB --host ALIAS` clears the run ids and starts the job fresh on the next host. It does not resume the old run on a different machine. A failed attempt waits (`queue.retry_backoff_s`) and then sorts behind jobs that have not failed. A pending file that is not valid JSON is logged and moved to `corrupt/`.

```bash
sma queue add Qwen/Qwen3-8B --bf16
sma queue add org/model --nvfp4 --engine sglang
sma queue list
sma queue reclaim          # return a claim whose heartbeat is stale
sma queue unpin <job-id>
sma queue reassign <job-id> --host gpu-host-2
sma worker --once
```

`gpu.lock` stops two SMA processes on the same Spark from starting two models. `run-pair` holds that lock for both legs.

## Run record contract

```
runs/<model-slug>/<engine>/<quant>/<run-id>/
  manifest.json                 schema version 2
  summary.json                  flat headline numbers; null means not measured
  steps/<step>.done             the resume marker
  fetch/result.json             SHA, hashes, params, detected quant, composition
  determinism.json
  logprobs/dump.json            per-token top-k, tail, true-token logprob
  logprobs/corpus.json          dataset, revision, sha256, license
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
runs/catalog.sqlite              fast index, including engine
runs/index.json                  atomic snapshot of the catalog
runs/compares/<ref>__<cand>.json output of sma compare
```

Older runs that live at `runs/<model-slug>/<quant>/<run-id>/` are still found by `sma list`, `sma status`, `sma resume`, and `sma compare`. New runs are engine-separated. A resumed run keeps the engine it was started with.

Schemas: [`schemas/manifest.schema.json`](schemas/manifest.schema.json) and [`schemas/summary.schema.json`](schemas/summary.schema.json).

JSON is written to a temp file, fsynced, and renamed. A crash does not leave a half-written manifest or summary.

The manifest is the subject record: model repo, revision SHA, weight hashes, quant, mixed composition when the config exposes layer or group counts, architecture check, parameter counts, engine name, image and digest, every server flag that was passed, the env vars SMA set, the MoE backend for quality and for speed, the SMA git commit, the host alias, timestamps, and step status. It does not contain the machine hostname or the HF token. `summary.json` uses null for anything that was not measured. It does not fill missing scores with zero. `quant` in the index is `mixed` for a mixed checkpoint.

Energy in `summary.json` is the sum of per-window GPU power integrals over valid speed cells. A gap longer than `telemetry.max_sample_gap_s` (default 2.5 s) is not trapezoided, and the gap between two cells is not one interval. Invalid and unverified windows are excluded. Scope is `gpu`. Null means it was not measured.

## Corpus

`sma-corpus-v3` is a vendored public-domain slice of Pride and Prejudice, Project Gutenberg ebook 1342, file `1342-0.txt`. It starts at "It is a truth universally acknowledged". The illustrated edition's list of illustrations and page numbers that opened the previous slice was removed, and the hash changed. Commercial use is allowed. The bytes are pinned:

| Field | Value |
| --- | --- |
| dataset | `project-gutenberg-1342` |
| revision | `1342-0.txt` |
| license | `public-domain` |
| sha256 | `a09bbe27eade79c728f69f22f8e0bee90ef75c3e28ee99d396533de4733fb2ed` |
| size | 441,736 characters, 77,566 words |

`load_corpus` checks the hash and refuses a mismatch. The default asks for 48 windows of 2048 tokens (98,304 scored tokens). If a tokenizer yields fewer tokens than that, the logprob step fails instead of scoring a shorter corpus. The windows that were actually scored are what `sma compare` checks.

## Config

`sma.yaml` in the working directory, or `--config`. Unknown keys are an error. `--quantization` and `--tensor-parallel-size` other than 1 are refused, including inside `engine.extra_args`. SGLang attention must stay `flashinfer` (required on SM12x). `--tp-size` is forced to 1.

The vLLM tag is pinned (`nvcr.io/nvidia/vllm:26.05.post1-py3`). The SGLang tag is pinned (`lmsysorg/sglang:v0.5.10.post1-cu130`, the versioned arm64 / CUDA 13 tag, not `latest-cu130`). Set `engine.image_digest` or `engine.sglang_image_digest` when you already know a digest. Otherwise SMA records the digest `docker image inspect` returns and uses `image@sha256:…` for later starts in that run, including the other leg and a restart. It does not invent a digest.

## Caveats

- Batch invariance is beta. vLLM documents it for newer datacenter GPUs. It is not verified here on sm_121 or with NVFP4. Gated delta-net models (Qwen3.5 / Qwen3.6) are rejected by that mode, so SMA does not set the flag for those architectures, records a warning, and continues at the fixed concurrency.
- Turning batch invariance on is not enough if async scheduling is still composing the next batch. vLLM quality mode also passes `--no-async-scheduling`. That flag was not executed against a live vLLM in this repo.
- MoE backend auto-selection on SM121 can pick Marlin, and Marlin NVFP4 on SM121 has crash reports. vLLM MoE NVFP4 and MXFP4 set `--moe-backend` explicitly, then try `marlin` if `flashinfer_b12x` never becomes healthy. That is step 1 of the arena plan's NVFP4 chain. If every backend in the chain fails, the step is failed, not skipped. SMA does not quantize and does not switch to SGLang as a fallback.
- Some Spark units throttle or power off under a sustained load. Throttled speed cells are marked invalid. They are not deleted. `sw_power_cap` is recorded and, by default, does not invalidate the cell.
- Top-k KL is an approximation. The tail is one bucket. If a reference top-k id is missing from the candidate's stored top-k, that mass is folded into the tail and the position is labeled approximate. Positions that would have been infinite KL stay in the mean with probability floored at `1e-12`, and the count is `kl_floored_positions`. Full-vocabulary KL is not computed.
- Tokenizers must match. `sma compare` refuses otherwise. Tokens per answer are not a shared unit across models.
- lm-eval scores are not the arena's pinned scorers. The default suite list is small and limited (ifeval, gsm8k, mmlu_pro, gpqa_diamond). Clear `limit` to run a whole task. The arena's full suite set is not the default.
- Speed repeats default to 1, plus a warm-up. The arena plan's "at least 3 runs" is `speed.repeats`.
- Energy is the sum of 1 Hz GPU power integrals over valid speed windows (`scope: gpu`). It is not joules per correct answer and not a wall-meter reading.
- The bandwidth ceiling in the summary uses the configured 273 GB/s spec constant and approximate bytes per parameter. It is not a measurement. NVFP4 scale overhead is not included. Mixed checkpoints have no single byte width, so that ceiling is null.
- The determinism probe sends one completions request with N copies of each prompt, with prefix caching disabled on the quality server. The client cannot force the scheduler to fuse those copies into one batch. `max_num_seqs` is at least 64 so a batch of 64 is allowed.
- Logprobs go through the server that is already up. A second offline `LLM()` process is not started, because the quality server already holds the GPU.
- SGLang measurements that the bench JSON or `/metrics` scrape does not contain are null, with a reason on the cell or in `metrics.json`. They are not copied from vLLM's metric names.
- SMA does not apply arena fidelity verdicts (broken / degraded / faithful / lossless). It records KL, top-1, and perplexity. Thresholds belong to the arena.

## What this environment did not verify

This workspace has no NVIDIA GPU and no DGX Spark. The tests use a fake HTTP server and a fake container runtime. They do not pull the NGC or SGLang image, start vLLM or SGLang, run lm-eval, or read a real GPU sensor.

Not verified on hardware:

- either pinned image actually starting on GB10 / sm_121 / ARM64 / CUDA 13
- `VLLM_BATCH_INVARIANT=1` and `--no-async-scheduling` on sm_121 or NVFP4
- `--moe-backend flashinfer_b12x` and the Marlin fallback on a real MoE checkpoint
- `--no-enable-flashinfer-autotune`, `--no-enable-prefix-caching`, and `--max-logprobs` against the pinned vLLM image
- SGLang `--attention-backend flashinfer`, `--enable-deterministic-inference`, `--disable-radix-cache`, `--enable-metrics`, `--max-running-requests`, and `--random-seed` on SM12x
- SGLang `/generate` input-token top-k, `sglang.bench_serving` JSONL from the pinned tag, and SGLang `/metrics` after `--enable-metrics`
- that the vendored novel slice tokenizes to at least 98,304 tokens on a real checkpoint tokenizer
- `vllm bench serve` writing the JSON shape this parser accepts (several historical shapes are accepted; a new shape needs a parser update)
- lm-eval task names (`mmlu_pro`, `gpqa_diamond`) on the installed harness
- thermal throttle flags from `nvidia-smi` on a Spark, including the older `clocks_throttle_reasons` names
- dropping the page cache (the unit test only checks that a permission error is recorded)
- loading `FlintForCausalLM` or `dgpp_mixed` in any image
- any benchmark number, quality score, tok/s, joules, or KL on a real model

No benchmark numbers are shipped with this repo.

## Development

```bash
uv sync --extra dev
uv run pytest
```

`sma run Qwen/Qwen3-8B --bf16 --dry-run` does not need a GPU. It does not check that vLLM's registry contains `Qwen3ForCausalLM`; a real run does.
