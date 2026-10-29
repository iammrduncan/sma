# Serving design: engines, KV cache, context, and memory

Status: decided by the owner on 2026-10-27, and built (the build goal was
[`serving-design.goal`](../serving-design.goal)). The README describes how SMA behaves; this file keeps
the decisions and their reasons. Not built: the GGUF quant grid (future, below).

## Why

SMA's results should show what someone actually gets on a DGX Spark. Most Spark users run NVFP4.
Production deployments run an 8-bit KV cache. Local setups commonly run 256K context or more. So SMA
serves each model the way it would really be deployed, and records every setting.

If a model's size and context capacity give it an edge within the Spark's memory, that edge counts.
It shows up through measurements: the capacity numbers in the deployment notes, and scores on tasks
that need the context. It is never assumed.

## Engines

The checkpoint format picks the engine before launch.

| Checkpoint | Engine | KV cache |
| --- | --- | --- |
| Safetensors: BF16, NVFP4, mixed, MXFP4 | vLLM (default) | FP8 |
| Safetensors that the pinned vLLM image cannot serve | SGLang | FP8 |
| GGUF, for example Unsloth `UD-Q4_K_XL` | llama.cpp | Q8_0 |

- The engine is chosen before launch: by format, by `--engine sglang`, or because preflight shows
  that the pinned vLLM image cannot serve the architecture. The run records it.
- A failed server start is never retried on another engine. A measurement is never copied from one
  engine's record into another's. These existing rules stay.
- SMA downloads GGUF files as published. It never makes or converts a GGUF.
- Every engine image is pinned by tag and its digest is recorded. llama.cpp needs an arm64 CUDA build
  that runs on the GB10 (sm_121). The run records its llama.cpp build number.

## KV cache: always 8-bit

- vLLM and SGLang pass FP8 explicitly, never `auto`. With `auto`, vLLM 0.21 follows the checkpoint:
  NVIDIA's NVFP4 checkpoints ask for FP8, but a BF16 checkpoint gets a 16-bit cache.
- llama.cpp uses `q8_0` for both K and V, which needs flash attention on. llama.cpp has no FP8 KV
  type. Q8_0 is its 8-bit equivalent: 8.5 bits per value, about 6% more than FP8.
- SMA never runs a 16-bit KV cache. The BF16 leg's label describes its weights. Its KL reference is
  BF16 weights with an FP8 KV cache.
- The run records the KV type and where its scales came from: calibrated in the checkpoint, or the
  engine's default.

## Context per stream

- Start at 256K, or at the model's native maximum if that is smaller. Native means without RoPE
  stretching such as YaRN. SMA never turns stretching on. If a config ships with stretching on, the
  native maximum is its original length.
- If that does not fit the memory budget, step down by halves: 256K, 128K, 64K, 32K, and so on.
- Stop at the largest step that fits. If the smallest context a benchmark needs does not fit, refuse
  the run before launch and say why.
- This is computed before launch and recorded as the model's context.
- Each benchmark gets the largest step that fits the number of streams it runs at once. A SWE-bench
  run with 4 agents needs 4 full-length streams. Each suite records the context it ran with.

## Memory budget: 100 GB

- A run's planned footprint stays under 100 GB of the Spark's 130.7 GB. The footprint is the weights,
  the KV pool, and the engine's own overhead.
- The plan is computed before launch. Weight bytes come from the checkpoint files. KV bytes per token
  come from the config: 2 (K and V) × full-attention layers × KV heads × head size × bytes per
  value. Linear-attention, convolution, and Mamba layers keep a small fixed state per stream instead.
  The engine overhead is measured per engine on the Spark and kept in config.
- SMA sizes the KV pool explicitly instead of using `gpu_memory_utilization: 0.90`. vLLM takes
  `--kv-cache-memory-bytes`, which `engine.kv_cache_gib` already sets. On the Spark the GPU shares
  system RAM, and 0.90 reserves 117.6 GB whatever the model.
- llama.cpp's pool holds its slots (`-np`) at the context per stream and no more. llama-server sizes
  its compute buffers by its whole cache, about 4 to 6 KB per token, so a pool no request can use
  still costs memory. The plan counts those buffers per token (`runplan.llamacpp_buffer_bytes_per_token`).
- The run records its measured peak memory next to the plan.
- The budget is a plan, not a limit. Nothing stops an engine that allocates more than planned. On the
  GB10 the GPU's memory is not charged to the process (earlyoom saw llama-server at 20 MiB while the
  machine had none free), and earlyoom set to `-m 8 -s 100` only sends SIGTERM: its SIGKILL also
  waits for free swap under 50%. On 2026-10-28 LFM3-1.7B got the rest of the budget as its llama.cpp
  pool, 11.6 million tokens for 8 slots of 262,144; its buffers took 47 GB the plan did not count,
  the run peaked at 127 GB, and the next run's server took the Spark down.
- Why the margin: our Spark runs `earlyoom`, which stops `vllm` and `python` processes when free RAM
  falls under 8% (about 10.5 GB). That stop looks like an engine crash. The 100 GB budget leaves
  about 20 GB free.

Worked example: a 40B MoE at each leg's context, with 5 GB of engine overhead. "Streams" means
full-length streams that fit at once. Weight sizes scale real files: NVIDIA's Qwen3.6-35B-A3B NVFP4
averages 5.34 bits per weight, Unsloth's Qwen3-30B-A3B `UD-Q4_K_XL` 4.64. The planner computes these
numbers from the real configs (tests/test_memplan.py); the hybrid column includes Qwen3.6's 64 MB
of linear-attention state per stream.

| Leg | Weights | Context | Full attention, 48 layers × 4 KV heads (Qwen3-30B-A3B layout) | Hybrid, 10 of 40 layers attend (Qwen3.6 layout) |
| --- | --- | --- | --- | --- |
| BF16 + FP8 KV | 80 GB | 256K | 12.9 GB per stream, 1 stream, 97.9 GB | 2.75 GB per stream, 5 streams |
| NVFP4 + FP8 KV | 26.7 GB | 256K | 12.9 GB per stream, 5 streams | 2.75 GB per stream, 24 streams |
| `UD-Q4_K_XL` + Q8_0 KV | 23.2 GB | 256K | 13.7 GB per stream, 5 streams | 2.92 GB per stream, 24 streams |

## Deployment notes

Every run gets a deployment block. The summary states it as a sentence. For NVIDIA's
Qwen3.6-35B-A3B NVFP4 the plan computes: "NVFP4 weights with an FP8 KV cache: 23.4 GB of weights,
256K per stream, 26 full-length streams within 100 GB." A real run replaces the estimates with
measurements.

| Field | Source |
| --- | --- |
| Engine, weight format, weight size | Checkpoint files |
| KV type, KV bytes per token | Config, computed |
| Context per stream, and why (native maximum or the step that fit) | Memory plan |
| KV pool size in tokens | The engine's own report (vLLM logs `GPU KV cache size: N tokens`) |
| Streams that fit at full context and at 32K | Pool size divided by context |
| Speed at 1, 4, and 8 streams | Existing speed cells, measured |
| Peak memory | Measured during the run |

Computed values are labeled computed. `null` means not measured.

## GGUF quant grid for one model (future)

Not part of this build, but the build must not block it. This is the arena plan's "GGUF ladder".

- Pick one model and one GGUF repo, for example an Unsloth repo. Run every published quant file, from
  the smallest (Q1) through Q8 and the repo's BF16 or F16 file, on llama.cpp with a Q8_0 KV cache.
- Each file is one run, labeled with its GGUF quant type and the file's sha256.
- The view is a heatmap: one row per quant file ordered by size, one column per benchmark. Size, KL,
  speed, and streams that fit sit beside the scores, so it shows where the model works best.
- The KL reference is the same repo's BF16 or F16 GGUF on llama.cpp, so KL stays on one engine.
- A grid multiplies runs by the number of files, so it uses a smaller, faster benchmark set than a
  full arena run.

## Boundary changes

These shipped with the build. AGENTS.md has them:

- "SMA never quantizes" becomes: SMA never quantizes or converts weights. It downloads GGUF files as
  published. The KV cache is always 8-bit: FP8 on vLLM and SGLang, Q8_0 on llama.cpp.
- "Never switch engines as a fallback" stays. Add: the engine is chosen before launch, by checkpoint
  format, by `--engine`, or because preflight shows that vLLM cannot serve the model.
- The summary line at the top names all three engines.

## Settled in the build

These were build details, not owner decisions.

- **llama.cpp and each SMA step.** Quality and tool calls go through llama-server's OpenAI endpoint
  (`--jinja`). Speed uses vLLM's bench client from the pinned vLLM image, in its own container, with
  the GGUF's own tokenizer. llama-server returns no prompt logprobs, so the quality server stops and
  `llama-perplexity` scores the corpus with the same Q8_0 cache. A full-precision GGUF writes its
  logits, and a quant run with `--reference` reports llama.cpp's full-vocabulary KL against them. The
  determinism probe runs and records what it sees; the record says llama.cpp has no batch-invariant
  mode.
- **A benchmark's context.** The quality suites need at most `memory.quality_min_context` (8,192)
  per request, so the server's context is the only limit and each suite records its step. KernelBench
  (8 streams) and SWE-bench (4 agents, at least 32,768 each) record theirs the same way. mini-swe-agent
  has no client-side cap, so an agent that outgrows its step still runs; the server may then queue
  others behind it.
- **vLLM batch invariance with an FP8 KV cache.** It starts on the GB10 (checked 2026-10-27,
  Qwen3-0.6B, TRITON_ATTN backend).
- **SGLang's pool.** `--max-total-tokens` sets it in tokens. Deterministic SGLang also needs
  `SGLANG_ENABLE_JIT_DEEPGEMM=0` on the GB10, with or without FP8.
- **Real kernels for quantized checkpoints** (owner decision during the build). vLLM's batch-invariant
  quality mode served NVFP4 by emulation on the Spark ("VLLM_BATCH_INVARIANT forces NVFP4 linear to use
  the emulation backend"), at about 2 tokens per second per request for an 8B model. Quantized
  checkpoints now run their real kernels with batch invariance off; BF16 keeps it.
- **Engine overhead.** `memory.overhead_gb` per engine, measured on the Spark
  (docs/first-spark-run.md), covers the engine and SMA's tools beside it (nel, the bench client):
  10 GB for vLLM. llama.cpp also needs compute buffers for each token of its cache (the 16-bit
  attention mask for one 512-token micro-batch, on the GPU and again on the host, and one attention
  layer's K and V converted to 16 bits for flash attention; measured 6,164 bytes per token for
  Qwen3-0.6B and 4,100 for LFM3-1.7B against 6,144 and 4,096 computed) and its host prompt cache
  (`--cache-ram`, 8 GiB, llama.cpp's default, counted in the plan).
- **SMA's own memory.** The logprob step streams its dump to disk one window at a time; building it
  whole took 7 GB beside the pool and pushed a run to 107 GB.

## Open for the owner

- A config that ships RoPE stretching on gets its original length as the native context, as decided.
  gpt-oss (YaRN, original 4,096) and DeepSeek-style configs then plan at 4,096 tokens, though they
  were released for long context. Treating a release config's own stretching as native would plan
  them at their advertised length; SMA would still never add stretching.

## KernelBench and SWE-bench (decided 2026-10-27)

- Both run in every `sma run`, after the quality suites, on the quality server. `kernelbench.enabled`
  and `swebench.enabled` turn them off.
- Both use the model's own thinking mode (the chat template's default) and sampling (the
  checkpoint's `generation_config.json`), not greedy decoding.
- KernelBench: Level 1 only, one answer per problem, KernelBench's own prompt and scoring at the
  service's commit. The headline is fast_1.
- SWE-bench Verified-mini: mini-swe-agent's `swebench.yaml`, 4 agents, 250 steps. The runner's
  gold-patch baseline is in
  [`runs/cpu/baseline/sweverified/`](../runs/cpu/baseline/sweverified/README.md).

## BrowseComp-Plus and MuSiQue (decided 2026-10-28)

- BrowseComp-Plus runs in every `sma run` after SWE-bench, on the quality server; its judge step runs
  after speed. `browsecomp.enabled` turns it off. 100 of 830 questions, a seeded subset.
- The BM25 index (2.17 GB) and its search server run on the x86_64 runner, not the Spark, so the
  document store takes nothing from the model's memory or CPU. The Spark reaches it over SSH.
- It uses the model's own thinking mode and sampling, as KernelBench and SWE-bench do. Each question
  gets a stream with at least 32,768 tokens of context (`browsecomp.min_context`), the largest ladder
  step that fits 4 of them.
- The judge is the lab's Ornith 1.5 122B, with thinking off at temperature 0, one request at a time
  ([judge-info.md](../judge-info.md)).
- MuSiQue-Ans is a quality suite (100 seeded questions, thinking off, greedy), graded by rule.
