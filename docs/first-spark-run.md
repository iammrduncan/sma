# First run on a DGX Spark

This is a hardware checklist. The laptop tests do not pull an image, start vLLM or SGLang, or read a GPU. Work through vLLM first. Start the SGLang trial only after that run's checks below are true.

Use a host alias in `sma.yaml` (`gpu-host-1` is the default). It must not be a real hostname. Put the Hugging Face token in the environment, not in the yaml or a run folder.

```bash
export HF_TOKEN=hf_...
```

The smoke model is `Qwen/Qwen3-0.6B`. It is small, ungated, and already BF16. It is a wiring check, not an arena score. A larger ungated BF16 such as `Qwen/Qwen3-8B` is the same command with a longer load.

## vLLM

### 1. Doctor

```bash
sma doctor
```

Doctor does not pull an image and does not start a model. Expect `host_alias`, `docker`, `gpu`, and `gpu_idle` to be ok before you continue. `ngc_image` fails until the image is local. That is expected on a first boot. `hf_token` must be ok. `lm_eval` may warn. Quality will fail later if `lm_eval` is not installed (`uv sync --extra quality`).

### 2. Pull, digest, and the registry query

```bash
sma run Qwen/Qwen3-0.6B --bf16
```

Fetch pulls `nvcr.io/nvidia/vllm:26.05.post1-py3` before the weight snapshot. Progress is one JSON log line per pull update, event `image_pull_progress`, on stderr and in `runs/<slug>/vllm/bf16/<run-id>/logs/sma.log`.

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

## SGLang, after the vLLM checks pass

This is a separate subject. Do not compare it with the vLLM run unless you pass `--allow-cross-engine`, and then the compare file is labeled `CROSS-ENGINE`.

```bash
sma run Qwen/Qwen3-0.6B --bf16 --engine sglang
```

The image is `lmsysorg/sglang:v0.5.10.post1-cu130`. The same pull, digest, and `architecture_check.source: container` rules apply. `FlintForCausalLM` is still refused. This command does not install a wheel.

Quality flags include `--attention-backend flashinfer`, `--enable-deterministic-inference`, `--disable-radix-cache`, `--enable-metrics`, `--max-running-requests 64`, and `--random-seed`. `engine.env_quality` is `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1`.

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
