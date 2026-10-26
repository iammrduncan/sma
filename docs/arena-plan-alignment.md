# Arena plan alignment

This maps the arena plan page ("Multi-quant benchmark and small model arena plan for DGX Spark") onto SMA v0. The launcher brief wins where the two disagree. SMA does not import or copy inference-benchmarks, Netsphere-Eval, or any other benchmark repo.

| Plan section | v0 status | Where |
| --- | --- | --- |
| Subject = model × checkpoint × quant × engine | Recorded | `manifest.json` `subject`, plus revision SHA, weight hashes, image, digest, host alias |
| No judge on the ranked board | Quality scores are lm-eval only, labeled `lm-eval-harness` | `quality/<suite>/scores.json`, `summary.json` |
| Deterministic seed, temperature 0 | Server `--seed`, bench `--seed`, determinism and logprob requests at temperature 0 | `sma/commands.py`, `sma/probe.py` |
| Publish failures, do not hide them | Failed steps store the error. A failed FP4 start stays failed. Invalid speed cells are kept | `manifest.steps`, `speed/summary.json` |
| Run folders the arena can read later | SMA layout below, not the branch's `run.json` / `raw.jsonl` tree | README "Run record contract" |
| Reference is BF16, or the native checkpoint for MXFP4 | `--bf16` or `--mxfp4`. KL only when a reference run is linked | `sma compare`, `--reference`, `run-pair` |
| One guaranteed Spark, up to two more, each pulls a whole job | `sma worker` claims one job. Pair jobs are one claim | `sma/queue.py` |
| Never tensor-parallel across Sparks | `--tensor-parallel-size 1` is forced. Other values are refused | `sma/commands.py`, `sma/config.py` |
| Both quants of a model on the same Spark | `sma run-pair` and `sma queue add-pair` run reference then NVFP4 under one GPU lock | `sma/cli.py`, `sma/queue.py` |
| vLLM only, NGC image, pin the tag, record the digest | `engine.image` default `nvcr.io/nvidia/vllm:26.05.post1-py3`. Digest comes from `docker image inspect` | `sma.yaml`, `sma/runtime.py` |
| Explicit `--moe-backend`, default `flashinfer_b12x` | Always passed for MoE NVFP4 and MXFP4. Not passed for BF16 | `sma/commands.py` |
| NVFP4 fallback chain | v0 tries the configured backend chain and records which one started. It does not quantize, switch engines, or relabel another quant as NVFP4 | `sma/pipeline.py` `_start_server` |
| Dense NVFP4 linear backend recorded | `--linear-backend` default `flashinfer_b12x` for dense NVFP4 | `sma/commands.py` |
| FlashInfer autotune off for quality | `--no-enable-flashinfer-autotune` on the quality server | `sma/commands.py` |
| `VLLM_BATCH_INVARIANT=1` for quality and logprobs | Set when the architecture is not a known gated delta-net. Also `--no-async-scheduling` | `sma/commands.py`, `sma/quant.py` |
| Determinism probe at batch 1, 8, 64. Record pass or fail. Do not abort | `determinism.json`. A failure is a warning | `sma/probe.py` |
| Quality packed, concurrency recorded | `quality.concurrency`, copied to `dispatch_concurrency` | `sma.yaml`, manifest |
| Speed and energy solo, batch invariance off | Speed server is a second container with the flag unset. One server at a time | `sma/pipeline.py` |
| 1 Hz telemetry: clock, temperature, power, throttle reasons | `telemetry.csv` | `sma/telemetry.py` |
| Reject throttled speed or energy cells and re-run after cool-down | One retry, then the cell stays `valid: false` and is dropped from headline tok/s | `sma/pipeline.py` `_run_cell` |
| Warm-up soak, idle cool-down between models, drop page cache | Warm-up before speed. Cool-down and page-cache drop in cleanup. Page cache is skipped when not permitted | `sma/cleanup.py` |
| `vllm bench serve` percentiles ttft, tpot, itl, e2el at 50, 90, 99, `--save-result` | The command is built in one place and printed by `--dry-run` | `sma/commands.py` |
| Prefill tok/s = input length / TTFT at c=1. Decode tok/s = 1 / TPOT at c=1 | `summary.json` `speed.prefill_tok_s` and `speed.decode_tok_s`, stat name recorded | `sma/speedmath.py` |
| `/metrics` diff per cell | `speed/cells/<cell>/attempt-<n>/metrics.json` | `sma/prom.py` |
| KL / PPL from `prompt_logprobs`, top-k plus tail, k recorded | `logprobs/dump.json`, `sma compare` | `sma/kl.py`, `sma/probe.py` |
| Same token ids for both quants, no chat template | Corpus `sma-corpus-v1`. Compare refuses a tokenizer or token-id mismatch | `sma/corpus.py`, `sma/compare.py` |
| Params from safetensors, total and active, size class by total | `manifest.params`. Class bounds match the plan, ASCII hyphen in the label | `sma/params.py` |
| Host alias, no real hostname, no secrets | `host_alias` is checked. Stored errors are scrubbed. HF token is only read from the environment | `sma/identity.py`, `sma/scrub.py` |
| Other-engine results must be labeled | Every SMA run is `engine: vllm`. Compare warns if a manifest is not | `sma/compare.py` |
| Quant must match `quantization_config` / `hf_quant_config.json` | Mismatch raises before the server starts | `sma/quant.py` |
| Clock cap recorded if used | `engine.clock_cap_mhz`. Applied only when `apply_clock_cap` is true | `sma.yaml` |
| Energy block | Joules from the 1 Hz GPU power integral over valid speed windows. Scope `gpu`. Not per correct answer | `summary.json` `energy` |
| Per-model run order | One checkpoint is fetch, quality-mode serve, probe, logprobs, quality, speed-mode serve, speed, teardown. A pair runs that twice, reference first | `sma/pipeline.py` |
| L · SMA launcher | This repository | README |

## Not in v0

These plan sections are intentionally not implemented. They belong to the arena app, a later phase, or a tool SMA is not supposed to copy.

- Quant plates, win rate, Bradley-Terry, Pareto boards, and the static site
- Arena submission, signed PRs, replication badges, fidelity verdict thresholds
- The full arena suite list and pinned bench scorers (IFEval 541, GPQA Diamond 198, GSM-Plus, LiveCodeBench, AIME, BFCL, Global MMLU, Terminal-Bench, SWE-bench)
- Easy-tier suites, ledger / payload generators, contamination policy, roster versions
- SGLang, TensorRT-LLM, llama.cpp, the GGUF ladder, the imatrix, and the V100 / P100 track
- Joules per correct answer, dollars from `rates.yaml`, and tokens per correct answer
- Workload-track benchmarks (KernelBench, BigCodeBench, MuSiQue, JSONSchemaBench, and the custom sets)

## Deviations from the arena plan

The brief wins on these. They are also listed in the pull request.

1. **`sma run` is one checkpoint.** The plan's six-step order runs both quants in one sitting. `sma run-pair` and `sma queue add-pair` do that on one Spark. A lone `sma run` does not fetch the other quant.
2. **Logprobs use the server that is already up.** The plan describes an offline `LLM()` process. v0 calls `/v1/completions` with raw token ids and `prompt_logprobs`, because a second process would need the same GPU.
3. **Quality defaults are the small lm-eval list with limits,** not the arena's full item counts, and the scorer is labeled `lm-eval-harness`.
4. **`speed.repeats` defaults to 1,** not 3. The warm-up still runs.
5. **`--moe-backend flashinfer_b12x` is not passed for BF16 MoE.** That backend is an FP4 kernel. It is always passed for MoE NVFP4 and MXFP4.
6. **The fallback chain stops at the MoE backend list.** v0 does not quantize, does not start SGLang, and does not relabel W4A16 or MXFP4 as W4A4 NVFP4. A W4A16 NVFP4 checkpoint is accepted by `--nvfp4` and stored as `nvfp4_w4a16`.
7. **`--no-async-scheduling` is on in quality mode** when batch invariance is on. The plan does not name that flag. Without it, batch invariance does not mean what the probe is checking.
8. **Autotune is forced off for quality only.** Speed mode leaves the image default and records that.
9. **KL missing-token positions are approximate.** When a reference top-k id is absent from the candidate top-k, its reference mass is folded into the tail. The position is counted in `kl_approximate_positions`. The exact page formula is used when every reference id is present.
10. **No fidelity verdict.** SMA does not invent broken / degraded / faithful / lossless thresholds.
11. **Run folder names follow the brief** (`manifest.json`, `summary.json`, `logprobs/`, `quality/`, `speed/`, `telemetry.csv`), not `results/<subject>/<suite>/...`.
12. **The image digest is not hardcoded.** The tag is pinned. The digest is written when Docker can inspect the image.
