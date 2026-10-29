# First run on a DGX Spark

This is a hardware checklist. The laptop tests do not pull an image, start vLLM or SGLang, or read a GPU. Work through vLLM first. Start the SGLang trial only after that run's checks below are true.

Use a host alias in `sma.yaml` (`gpu-host-1` is the default). It must not be a real hostname. Put the Hugging Face token in the environment, not in the yaml or a run folder.

```bash
export HF_TOKEN=hf_...
```

The smoke model is `Qwen/Qwen3-0.6B`. It is small, ungated, and already BF16. It is a wiring check, not an arena score. A larger ungated BF16 such as `Qwen/Qwen3-8B` is the same command with a longer load.

## Host memory

A GB10's GPU and CPU share one pool. On our Spark `/proc/meminfo` MemTotal is 130.7 GB, and SMA plans each run under `memory.budget_gb` (100 GB) of it. Check two things on the host before a run:

- **earlyoom.** `systemctl status earlyoom` shows its arguments. Ours stops processes at 8% free (about 10.5 GB) and prefers `python3`, `vllm`, and `llama-*`. A server stopped by it exits with no error of its own; `journalctl -u earlyoom` names what it stopped.
- **Display reserved memory (headless Sparks).** In UEFI, Display Reserved Memory set to Auto keeps about 0.5 GB for the display instead of 2 GB (a community tip, not checked here). The 1.5 GB it frees raises MemTotal; with the budget fixed at 100 GB it becomes headroom above earlyoom's floor, not a larger KV pool. Raise `memory.budget_gb` yourself if you want the pool to grow.

## vLLM

### 1. Doctor

```bash
sma doctor
```

Doctor does not pull an image and does not start a model. Expect `host_alias`, `docker`, `gpu`, `gpu_idle`, and `hf_cache` to be ok before you continue. `hf_cache` fails when the cache path contains the hostname, for example when the user name is also the hostname. Fetch would refuse that snapshot after the download. Set `hf.cache_dir` to a path outside your home directory. `ngc_image` fails until the image is local. That is expected on a first boot. `hf_token` must be ok. `quality_evaluator` may warn. Quality will fail later if `nel` is not installed (`uv sync --extra quality`).

### 2. Pull, digest, and the registry query

```bash
sma run Qwen/Qwen3-0.6B --bf16
```

Fetch pulls `nvcr.io/nvidia/vllm:26.05.post1-py3` before the weight snapshot. Progress is one line per pull update on the terminal. `runs/<slug>/vllm/bf16/<run-id>/logs/sma.log` has the same updates as JSON, event `image_pull_progress`. While the server starts, `logs/server-quality.log` fills live with the engine's output.

After fetch, open `manifest.json`:

- `engine.image_digest` is `sha256:…` from `docker image inspect` `RepoDigests`. It is not a digest typed into `sma.yaml`.
- `architecture_check.source` is `container`.
- `architecture_check.message` names `Qwen3ForCausalLM` and does not contain `WARNING`.
- `architecture_check.ok` is true.

`source: bundled` plus a `WARNING` means the image was not queried. Stop and fix Docker before reading any speed number. SMA records the inspect digest on `engine.image_digest`. The quality server, the speed server, and any resume of this run start that same image as `image@sha256:…` taken from `engine.image_digest`, not from the floating tag.

### 3. Bench tokenizer, offline Hub, and a finished cell

The speed command is `speed/cells/<cell>/attempt-1/command.json`.

For every cell, the argv must contain all of:

- `--tokenizer` then `/model`
- `--model` then `Qwen/Qwen3-0.6B` (the served name, not a Hub revision)
- `-e` `HF_HUB_OFFLINE=1`
- `-e` `TRANSFORMERS_OFFLINE=1`
- `--result-dir` ending in `attempt-1` (or `attempt-2` if a retry ran)
- `--result-filename` `bench.json`

`speed/cells/<cell>/attempt-1/bench.json` must exist. It must not be only `speed/cells/<cell>/bench.json`.

In that JSON, `completed` equals `speed.num_prompts` from `sma.yaml` (default 32). `errors` is absent or empty. The cell in `speed/summary.json` has `validity` `valid`, and `ttft_ms_p50`, `ttft_ms_p90`, and `ttft_ms_p99` are numbers for the prefill cell. The same three percentiles exist for tpot, itl, and e2el. `summary.json` `speed.prefill_tok_s` and `speed.decode_tok_s` are numbers, not null.

If a gated repo is the thing you care about, fetch still uses `HF_TOKEN` on the host. The bench container does not. A speed step that finishes with `--tokenizer /model` and the offline env vars did not refresh the tokenizer from `main`. A Hub timeout or 401 in `logs/speed-*.log` means the offline pin failed. Do not treat that cell as a result. `invalidation_reason` is `bench_failed`, and SMA retries it once after a backoff. That retry does not cool the GPU. Only `throttled` waits for a cool-down.

### 4. Telemetry

`telemetry.csv` is about one row per second (`telemetry.interval_s` is 1). A valid speed cell's samples inside its window are roughly `duration / 1`, and the largest gap is at most `telemetry.max_sample_gap_s` (2.5 s).

On a GB10 near 140 W, `sw_power` is often `Active` for the whole cell. With `telemetry.sw_power_cap_invalidates` false (the default), that cell stays `valid`. The flag is still in the csv. Set the config flag to true only when you want that column to invalidate the cell.

`[N/A]` in the throttle columns makes the cell `unverified`. Unverified cells stay on disk, are not retried, and `speed.prefill_tok_s` ignores them. A real `hw_thermal`, `sw_thermal`, or `hw_power` value of Active marks the cell `invalid` and retries it once after the cool-down.

### 5. Logprobs

`logprobs/corpus.json` has `num_windows` 48 and `corpus_version` `sma-corpus-v3`. `logprobs/dump.json` has 48 windows. The step fails, instead of scoring fewer windows, if this tokenizer emits fewer than 48 times 2048 tokens. The error is on `manifest.steps.logprobs`. The request timeout is `timeouts.logprob_request_s` (600 s each). Windows are one request at a time. `engine.max_model_len` is 8192, which is enough for a 2048-token window.

### 6. Determinism and batch invariance

`determinism.json` records batches 1, 8, and 64. `passed` may be true or false. A failure is a warning, and the run continues.

On the quality server, `engine.batch_invariant_quality` is true for this architecture, `engine.env_quality` contains `VLLM_BATCH_INVARIANT=1`, and `engine.flags_quality` contains `--no-async-scheduling` and `--no-enable-prefix-caching`. This model is not a gated-delta-net, so the flag should be on. If `batch_invariant_quality` is false, read `batch_invariant_warning` before comparing quality numbers.

### 7. MoE backend

`Qwen3-0.6B` is dense. `engine.flags_quality` must not contain `--moe-backend`.

For a later MoE NVFP4 or MXFP4 run, `engine.moe_backend_quality` and `engine.moe_backend_speed` are each either `flashinfer_b12x` or `marlin`. `moe_backend_*_fallback` is true only when the server that became healthy was not the first backend. A failed start is a failed step, not a skipped cell.

### 8. Memory plan, 8-bit KV cache, and deployment notes

`manifest.json` `memory_plan` has the context per stream and why (`context_reason`), the weights, the engine overhead (`memory.overhead_gb`), the KV pool, and each benchmark's context. For `Qwen/Qwen3-0.6B` the context is its native 40,960 tokens.

- `engine.kv_cache_type` is `fp8`, and `engine.flags_quality` and `engine.flags_speed` contain `--kv-cache-dtype fp8`, `--kv-cache-memory-bytes` equal to `memory_plan.pool_bytes`, and `--max-model-len` equal to `memory_plan.context`.
- `engine.kv_cache_reported.quality.kv_type_line` is vLLM's own line, "Using fp8 data type to store kv cache", and `pool_tokens` is within a few tokens of `memory_plan.pool_tokens`.
- `engine.kv_cache_scales` is `engine default (scale 1.0, not calibrated)` for a BF16 checkpoint. For a checkpoint that declares FP8 KV scales it starts with `checkpoint (FP8)`, and adds "the engine log says it used a KV scale of 1.0" when vLLM logs `Using KV cache scaling factor 1.0`. `nvidia/Qwen3-8B-NVFP4` declares them, and all 72 are 1.0.
- An NVFP4 run's quality server has batch invariance off (`engine.batch_invariant_quality` false, with `engine.batch_invariant_warning` saying why), and its log does not contain `Using EmulationNvFp4LinearKernel`. With batch invariance on, vLLM forced that emulation kernel and generated about 2 tokens per second per request for `nvidia/Qwen3-8B-NVFP4` (2026-10-27).
- `summary.json` `deployment.measured.peak_memory_gb` is under `memory.budget_gb` (100). If it is over, see where in `memory.csv` (the `phase` column) and raise `memory.overhead_gb` for that engine to the measured overhead: peak, minus weights, minus the pool.

Measured on a Spark on 2026-10-27, as memory above weights plus pool:

| Model on vLLM | Server up | Quality (nel beside it) | Speed (bench client in the container) |
| --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B` BF16 | 5 GB | 8.9 GB | 6.9 GB |
| `nvidia/Qwen3-8B-NVFP4` | 8.1 GB | 11.4 GB (NVFP4 emulation under batch invariance) | 8.3 GB |
| `Qwen/Qwen3-0.6B` BF16 on SGLang | | 12.3 GB at the run's peak | |

`memory.overhead_gb.vllm` and `memory.overhead_gb.sglang` are 13 for that reason; SGLang's margin is under 1 GB. The overhead grows with the model, so measure again with a larger one. The logprob step peaked 13.4 GB above weights plus pool until SMA streamed its dump to disk one window at a time. A run whose measured peak passes the budget gets a warning that names the setting to raise.

## llama.cpp and GGUF

```bash
sma run unsloth/Qwen3-0.6B-GGUF --gguf BF16
sma run unsloth/Qwen3-0.6B-GGUF --gguf UD-Q4_K_XL --reference <the BF16 run id>
```

The image is `ghcr.io/ggml-org/llama.cpp:full-cuda13-b11913` (arm64, CUDA 13). Its log line `system_info: ... CUDA : ARCHS = ...,1200,1210 | ... | BLACKWELL_NATIVE_FP4 = 1` shows kernels built for the GB10.

- `engine.selected_by` is `format`, `engine.build_info` is llama.cpp's own build (`b11913-7aca0d2ea`), and the run folder is `runs/<slug>/llamacpp/gguf-<QUANT>/<run-id>`.
- `engine.kv_cache_reported.quality.kv_type_line` is llama.cpp's `llama_kv_cache: size = ... K (q8_0): ... V (q8_0): ...`, and its cell count equals `memory_plan.pool_tokens`.
- `manifest.json` `gguf` names the file llama-server loaded, `general.file_type` from its metadata (`MOSTLY_Q4_K_M` for Unsloth's `UD-Q4_K_XL`), and the tensor types.
- `logprobs/command.json` is the `llama-perplexity` command. The BF16 run writes `logprobs/kld-base.bin` (about 2.5 GB for 8 chunks of Qwen3's 152K vocabulary). The quant run's `logprobs/kl_vs_reference.json` has `method` `llama-perplexity-kld`, `kl_mean`, and `top1_agreement`.
- `speed/cells/<cell>/attempt-1/command.json` is a `docker run ... --network host` of the vLLM image with `vllm bench serve` and `--tokenizer /model/<file>.gguf`, or `--tokenizer /bench-tokenizer` with the source model's tokenizer mounted when the image's transformers cannot read the GGUF's architecture (`engine.bench_tokenizer` says which). `engine.bench_image` is that image at its digest.
- Memory: `logs/server-quality.log` has llama.cpp's own breakdown (`common_memory_breakdown_print`): model, context (the KV pool), and compute. On 2026-10-27, with a 90.8 GB pool, compute was 7,357 MiB, sized by the whole `-c`; at idle the server sat 11.5 GB above weights plus pool, and 3.6 GB with a quarter of the pool, so about 0.9 GB plus 11.7% of the pool. During speed, its host prompt cache (`--cache-ram`, 8 GiB by default) grew it by another 6.8 GB, and that run peaked at 110.6 GB before the plan counted them. The plan now counts the compute buffers per token of the cache and caps the pool at the slots at the context per stream (docs/serving-design.md), and counts `engine.llamacpp_cache_ram_mib`. On 2026-10-28, LFM3-1.7B with the uncapped pool (`-c 11,618,048`) showed model 3,298 + context 72,347 + compute 34,063 MiB on the GPU and 11,364 MiB of compute on the host; capped at 8 slots (`-c 2,099,641`), the server sat at 27.4 GB with an empty prompt cache, against 28.8 GB planned before the cache.

## SGLang, after the vLLM checks pass

This is a separate subject. Do not compare it with the vLLM run unless you pass `--allow-cross-engine`, and then the compare file is labeled `CROSS-ENGINE`.

```bash
sma run Qwen/Qwen3-0.6B --bf16 --engine sglang
```

The image is `lmsysorg/sglang:v0.5.10.post1-cu130`. The same pull, digest, and `architecture_check.source: container` rules apply. The registry query runs with `--gpus`: without a GPU, SGLang's registry drops every model whose import needs `libcuda.so.1`, and on 2026-10-27 it listed 59 architectures without Qwen3, so preflight refused Qwen3-0.6B. With the GPU it lists 193. `FlintForCausalLM` is still refused. This command does not install a wheel.

Quality flags include `--attention-backend flashinfer`, `--enable-deterministic-inference`, `--disable-radix-cache`, `--enable-metrics`, `--max-running-requests 64`, `--random-seed`, `--kv-cache-dtype fp8_e4m3`, `--max-total-tokens`, and `--context-length`. Both servers run with `SGLANG_ENABLE_JIT_DEEPGEMM=0`: on 2026-10-27 the pinned image's deterministic mode crashed at CUDA-graph capture on the GB10 ("DeepGEMM failed ... Unsupported architecture") with DeepGEMM on, with a 16-bit or an FP8 KV cache. `engine.kv_cache_reported.quality.kv_type_line` is SGLang's "Using KV cache dtype: torch.float8_e4m3fn". `engine.env_quality` is `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1`.

### Metrics

`speed/cells/<cell>/attempt-1/metrics.json` has `diff.watched` keys that start with `sglang:`. A number there is a counter delta. Null plus `scrape_error` or `missing_reason` means that series was not on `/metrics`. Do not copy a vLLM name into the gap. The server must have been started with `--enable-metrics`. That flag is in `engine.flags_speed`.

### Bench

`command.json` contains `--model /model`, `--tokenizer /model`, both offline `-e` variables, `--dataset-path /sma-run/speed/sglang-random-dataset.json`, `--random-range-ratio 1.0`, and `--seed`. `speed/sglang-random-dataset.json` is in the run directory. Its sha256 matches `speed/sglang-random-dataset.meta.json`. There is no ShareGPT download in `logs/speed-*.log`.

`speed.num_prompts` cannot be greater than 256. SMA refuses that configuration. It does not shorten the run.

`bench.json` is JSONL. `completed` equals `num_prompts`. In `speed/summary.json`, the prefill cell has numeric `ttft_ms_p50`, `ttft_ms_p90`, and `ttft_ms_p99`. p90 TTFT, TPOT, and ITL are not in the SGLang summary on this tag. SMA fills them from the per-request arrays, after dropping failed requests (`ttft` 0, an empty `itl`, or a non-empty `errors` entry). If those arrays are missing, the p90 stays null and `measurement_note` says why.

### Logprobs

`engine.logprobs_mode` is `sglang_input_top_logprobs`. `logprobs/dump.json` windows have `top` entries from `/generate`. A missing field is null. It is not filled from a vLLM response.

## What still is not a pass

A finished run with null `prefill_tok_s`, `validity` `unverified`, `architecture_check.source` `bundled`, or a bench log that mentions the Hugging Face Hub is not a successful first run. Fix that check and run again. Do not publish the numbers.
