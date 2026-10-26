# Arena plan alignment

This maps the arena plan page ("Multi-quant benchmark and small model arena plan for DGX Spark") onto SMA. The launcher brief wins where the two disagree, except where the owner later required a change. Those are listed under owner-approved extensions. SMA does not import or copy inference-benchmarks, Netsphere-Eval, or any other benchmark repo.

| Plan section | Status | Where |
| --- | --- | --- |
| Subject = model × checkpoint × quant × engine | Recorded. The run path is `runs/<slug>/<engine>/<quant>/<run-id>`. Older `runs/<slug>/<quant>/<run-id>` folders still resolve | `manifest.json` `subject`, `runs/index.json` |
| No judge on the ranked board | Quality scores are lm-eval only, labeled `lm-eval-harness` | `quality/<suite>/scores.json`, `summary.json` |
| Deterministic seed, temperature 0 | Server seed, bench `--seed` on vLLM, determinism and logprob requests at temperature 0. vLLM quality disables prefix caching so the probe is not a cache hit. SGLang quality sets deterministic inference and disables the radix cache because attention is FlashInfer | `sma/backends.py`, `sma/probe.py` |
| Publish failures, do not hide them | Failed steps store the error. A failed server start stays failed. Invalid and unverified speed cells are kept and left out of headlines | `manifest.steps`, `speed/summary.json` |
| Run folders the arena can read later | SMA layout below, not the branch's `run.json` / `raw.jsonl` tree | README "Run record contract" |
| Reference is BF16, or the native checkpoint for MXFP4 | `--bf16` or `--mxfp4`. KL only when a reference run is linked. `--mixed` is an extra quant the owner added | `sma compare`, `--reference`, `run-pair` |
| One guaranteed Spark, up to two more, each pulls a whole job | `sma worker` takes the GPU lock, then claims one job. A failed job is recorded and the worker continues | `sma/queue.py` |
| Never tensor-parallel across Sparks | vLLM `--tensor-parallel-size 1` and SGLang `--tp-size 1` are forced. Other values are refused | `sma/backends.py`, `sma/config.py` |
| Both quants of a model on the same Spark | `sma run-pair` holds one GPU lock from the start of leg 0 through the end of leg 1. `sma queue add-pair` is one claim. The job is pinned to the alias that created the first run, and a different alias cannot claim or resume it | `sma/cli.py`, `sma/queue.py` |
| vLLM from the NGC image, pin the tag, record the digest | Default engine. `engine.image` is `nvcr.io/nvidia/vllm:26.05.post1-py3`. Digest comes from `docker image inspect` and later starts in that run use `image@sha256:…` | `sma.yaml`, `sma/runtime.py` |
| Explicit `--moe-backend`, default `flashinfer_b12x` | Always passed for vLLM MoE NVFP4, MXFP4, and mixed. Not passed for BF16. Quality and speed each store the backend that started | `sma/backends.py`, `manifest.engine` |
| NVFP4 fallback chain, step 1 | If `flashinfer_b12x` never becomes healthy, SMA tries `marlin` and records that backend. This is the plan's first fallback step, not a deviation. SMA does not quantize, does not switch engines on failure, and does not relabel another quant as NVFP4 | `sma/pipeline.py` `_start_server` |
| Dense NVFP4 linear backend recorded | `--linear-backend` default `flashinfer_b12x` for dense NVFP4 on vLLM | `sma/backends.py` |
| FlashInfer autotune off for quality | `--no-enable-flashinfer-autotune` on the vLLM quality server | `sma/backends.py` |
| `VLLM_BATCH_INVARIANT=1` for quality and logprobs | Set when the architecture is not a known gated delta-net. Also `--no-async-scheduling` | `sma/backends.py`, `sma/quant.py` |
| Determinism probe at batch 1, 8, 64. Record pass or fail. Do not abort | `determinism.json`. A failure is a warning. Prefix caching is off for this server | `sma/probe.py` |
| Quality packed, concurrency recorded | `quality.concurrency`, copied to `dispatch_concurrency` | `sma.yaml`, manifest |
| Speed and energy solo, batch invariance off | Speed server is a second container with the flag unset. One server at a time. Warm-up runs again whenever that speed container is a fresh start, including resume | `sma/pipeline.py` |
| 1 Hz telemetry: clock, temperature, power, throttle reasons | `telemetry.csv`, including `[N/A]` as unverified rather than a fake 0 | `sma/telemetry.py` |
| Reject throttled speed or energy cells and re-run after cool-down | One retry when a real hw/sw thermal or hw power flag is active, then the cell stays `invalid`. `sw_power_cap` is stored and does not invalidate unless `telemetry.sw_power_cap_invalidates` is true. Missing, failed, empty, or `[N/A]` telemetry is `unverified`, not retried, and excluded from headlines | `sma/pipeline.py` `_cell_record` |
| Warm-up soak, idle cool-down between models, drop page cache | Warm-up before speed, again after a new speed server. Cool-down and page-cache drop in cleanup. Page cache is skipped when not permitted | `sma/cleanup.py` |
| Bench percentiles ttft, tpot, itl, e2el at 50, 90, 99 | vLLM: `vllm bench serve --save-result` into `attempt-<n>/bench.json`. SGLang: `sglang.bench_serving` with a pinned local dataset, `--random-range-ratio 1.0`, and `--seed`. Tag `v0.5.10.post1` (`python/sglang/bench_serving.py`) emits median/p99 TTFT and TPOT, median/p95/p99 ITL, and median/p90/p99 `e2e_latency`. Missing p90 values are computed from per-request arrays or stored null. Not executed on the image | `sma/backends.py`, `sma/speedmath.py` |
| Prefill tok/s = input length / TTFT at c=1. Decode tok/s = 1 / TPOT at c=1 | `summary.json` `speed.prefill_tok_s` and `speed.decode_tok_s`, stat name recorded. Only `valid` cells contribute | `sma/speedmath.py` |
| `/metrics` diff per cell | `speed/cells/<cell>/attempt-<n>/metrics.json`. Names match with labels ignored. Absent watched series are null with a reason | `sma/prom.py` |
| KL / PPL from prompt logprobs, top-k plus tail, k recorded | `logprobs/dump.json`, `sma compare`. An epsilon floor keeps positions in the mean. `kl_infinite_positions` counts zeros before the floor. `kl_floored_positions` counts every floored position. vLLM uses `/v1/completions`. SGLang uses `/generate` input-token top-k. Compare labels a digest mismatch | `sma/kl.py`, `sma/compare.py` |
| Same token ids for both quants, no chat template | Corpus `sma-corpus-v3` (Project Gutenberg 1342, revision `1342-0.txt`, sha256 `a09bbe27eade79c728f69f22f8e0bee90ef75c3e28ee99d396533de4733fb2ed`, public domain, novel text only). Default is 48 windows of 2048 tokens. Compare refuses a tokenizer, corpus, or token-id mismatch | `sma/corpus.py`, `sma/compare.py` |
| Params from safetensors, total and active, size class by total | `manifest.params`. Class bounds match the plan, ASCII hyphen in the label | `sma/params.py` |
| Host alias, no real hostname, no secrets | `host_alias` is checked. Stored errors are scrubbed. HF token is only read from the environment | `sma/identity.py`, `sma/scrub.py` |
| Other-engine results must be labeled | `engine` is on the manifest, the summary, and the index. `sma compare` refuses a cross-engine pair unless `--allow-cross-engine`, and that file is labeled `CROSS-ENGINE` | `sma/compare.py` |
| Quant must match `quantization_config` / `hf_quant_config.json` | Mismatch raises before the server starts. `--nvfp4` accepts ModelOpt W4A4 and compressed-tensors NVFP4 only when activations are also 4-bit float. `input_activations: null` is `nvfp4_w4a16` and `--nvfp4` refuses it. `--mixed` requires both FP8 and NVFP4 in the counted formats | `sma/quant.py` |
| Clock cap recorded if used | `engine.clock_cap_mhz`. Applied only when `apply_clock_cap` is true | `sma.yaml` |
| Energy block | Joules are the sum of 1 Hz GPU power integrals, one per valid speed window. Gaps longer than `telemetry.max_sample_gap_s` are not bridged. Scope `gpu`. Not per correct answer | `summary.json` `energy` |
| Per-model run order | One checkpoint is fetch, quality-mode serve, probe, logprobs, quality, speed-mode serve, speed, teardown. A pair runs that twice under one GPU lock, reference first, same engine | `sma/pipeline.py`, `sma/cli.py` |
| L · SMA launcher | This repository | README |

## Owner-approved extensions

The page's default is one engine (vLLM) and two quants (BF16 or native, and NVFP4). The owner added these. They are not silent deviations.

1. **`--mixed`.** A third quant label, used only when the counted per-layer or per-group formats include both FP8 and NVFP4. `quant_method: dgpp_mixed` with a fp8 recipe, or with no layer map, is not `mixed`. `--nvfp4` does not accept a real mix. `sma compare` accepts BF16 against `mixed` on the same engine. Stock vLLM and stock SGLang both refuse `dgpp_mixed` before the weight download. vLLM does accept compressed-tensors FP8+NVFP4. SGLang refuses every compressed-tensors checkpoint, including that mix. SMA does not create the mix and does not install a plugin.
2. **`--engine sglang`.** Opt-in. The pinned image is `lmsysorg/sglang:v0.5.10.post1-cu130` (versioned arm64 / CUDA 13, not `latest-cu130`). The digest is recorded at inspect, not hardcoded. On SM12x the attention backend is FlashInfer. Quality mode enables deterministic inference and disables the radix cache. NVFP4 on SGLang is ModelOpt, not compressed-tensors. SGLang is not an automatic step in the NVFP4 fallback chain. A cross-engine comparison is refused unless `--allow-cross-engine`, and the output is labeled. The page's SGLang row ("labeled exception only", and fallback-chain step 4) is this opt-in, not a silent engine switch.
3. **`sw_power_cap` does not invalidate by default.** The flag is always stored (`telemetry.sw_power_cap_invalidates`, also on `summary.thermal`). A GB10 near 140 W can show it in steady state. Set the config flag to true to treat it like the other power flags.

## Not in this version

These plan sections are intentionally not implemented. They belong to the arena app, a later phase, or a tool SMA is not supposed to copy.

- Quant plates, win rate, Bradley-Terry, Pareto boards, and the static site
- Arena submission, signed PRs, replication badges, fidelity verdict thresholds
- The full arena suite list and pinned bench scorers (IFEval 541, GPQA Diamond 198, GSM-Plus, LiveCodeBench, AIME, BFCL, Global MMLU, Terminal-Bench, SWE-bench)
- Easy-tier suites, ledger / payload generators, contamination policy, roster versions
- TensorRT-LLM, llama.cpp, the GGUF ladder, the imatrix, and the V100 / P100 track
- Installing a checkpoint's custom wheel (including `sglang/flint_sglang-0.1.4-py3-none-any.whl`). `FlintForCausalLM` fails preflight on stock vLLM and on the pinned stock SGLang image
- Joules per correct answer, dollars from `rates.yaml`, and tokens per correct answer
- Workload-track benchmarks (KernelBench, BigCodeBench, MuSiQue, JSONSchemaBench, and the custom sets)
- Automatic fallback from a failed NVFP4 server into self-quantization, W4A16, MXFP4, or SGLang

## Deviations from the arena plan

The brief wins on these. Marlin-as-the-next-MoE-backend is step 1 of the page's NVFP4 chain, so it is not in this list. SGLang and `--mixed` are owner-approved extensions above, not deviations.

1. **`sma run` is one checkpoint.** The plan's six-step order runs both quants in one sitting. `sma run-pair` and `sma queue add-pair` do that on one Spark, under one GPU lock, on one engine. A lone `sma run` does not fetch the other quant.
2. **Logprobs use the server that is already up.** The plan describes an offline `LLM()` process. SMA calls that server's HTTP API (vLLM `/v1/completions` with raw token ids and `prompt_logprobs`, SGLang `/generate` input-token top-k), because a second process would need the same GPU.
3. **Quality defaults are the small lm-eval list with limits,** not the arena's full item counts, and the scorer is labeled `lm-eval-harness`.
4. **`speed.repeats` defaults to 1,** not 3. The warm-up still runs.
5. **`--moe-backend flashinfer_b12x` is not passed for BF16 MoE.** That backend is an FP4 kernel. It is passed for vLLM MoE NVFP4, MXFP4, and mixed.
6. **The fallback chain stops after the MoE backend list.** SMA does not quantize, does not start the other engine, and does not relabel W4A16 or MXFP4 as W4A4 NVFP4. A checkpoint that is already W4A16 NVFP4, including compressed-tensors with `input_activations: null`, is detected as `nvfp4_w4a16` and `--nvfp4` refuses it. SMA does not serve that format.
7. **`--no-async-scheduling` is on in vLLM quality mode** when batch invariance is on. The plan does not name that flag. Without it, batch invariance does not mean what the probe is checking.
8. **Autotune is forced off for vLLM quality only.** Speed mode leaves the image default and records that.
9. **KL is an approximation of the page formula.** When a reference top-k id is absent from the candidate top-k, its reference mass is folded into the tail (`kl_approximate_positions`). A q mass that is actually zero is counted in `kl_infinite_positions` and then floored at `1e-12` so the position stays in the mean. `kl_floored_positions` also counts tiny positive masses under that floor. The exact page formula is used when every reference id is present and every q mass is above the floor.
10. **No fidelity verdict.** SMA does not invent broken / degraded / faithful / lossless thresholds.
11. **Run folder names follow the brief** (`manifest.json`, `summary.json`, `logprobs/`, `quality/`, `speed/`, `telemetry.csv`), with an engine directory added so the subject identity includes the engine. They are not `results/<subject>/<suite>/...`.
12. **The image digest is not hardcoded.** The tag is pinned. The digest is written when Docker can inspect the image, then reused for the rest of that run.
13. **The determinism probe cannot force batch fusion.** It is one completions request with N copies of each prompt, with prefix caching disabled. `max_num_seqs` allows a batch of 64. The scheduler may still not fuse the copies.
14. **Energy is GPU-only.** It is the sum of 1 Hz integrals over valid speed windows, not joules per correct answer and not a wall meter.
15. **The bandwidth ceiling is the configured 273 GB/s spec constant,** not a measurement. Mixed checkpoints leave it null.
16. **The corpus is vendored, and the scored length is the tokenizer's.** The page asks for about 100k tokens of natural text, pinned by revision and hashed. SMA ships the public-domain Gutenberg 1342 novel text (sha256 `a09bbe27eade79c728f69f22f8e0bee90ef75c3e28ee99d396533de4733fb2ed`, 77,566 words) after removing the illustrated edition's illustration list and page numbers. The default asks for 48 windows of 2048 tokens (98,304). A tokenizer that yields fewer tokens fails the logprob step instead of scoring a shorter slice. SMA does not re-download the book on each run.
17. **An architecture is refused from the container registry, not from the built-in list alone.** Preflight pulls the image and queries `image@digest`. If that query cannot be run, SMA warns and continues unless the architecture is a known-unsupported case. Flint is always refused. SMA does not install a Flint wheel or plugin.
