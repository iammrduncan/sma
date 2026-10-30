# SMA

SMA (Small Model Arena launcher) is a standalone CLI. It takes one Hugging Face checkpoint that is already in the quant you name, serves it, and writes a resumable run folder.

SMA does not quantize or convert weights. It does not use tensor parallelism, including across machines. One Spark runs one model at a time. The host alias in the run record is not a hostname, and the HF token is read from the environment or the git-ignored `.env`, never from `sma.yaml`.

SMA serves each model the way people really run it on a Spark ([docs/serving-design.md](docs/serving-design.md)). The checkpoint format picks the engine before launch:

| Checkpoint | Engine | KV cache |
| --- | --- | --- |
| Safetensors: BF16, NVFP4, mixed, MXFP4 | vLLM (default) | FP8 |
| Safetensors the pinned vLLM image cannot serve | SGLang (`--engine sglang`, `engine.name`, or chosen by preflight) | FP8 |
| GGUF (`--gguf QUANT`, for example `UD-Q4_K_XL`) | llama.cpp | Q8_0 |

The engine is part of the subject. A failed start is never retried on another engine. The KV cache is always 8-bit, never 16-bit. Context per stream starts at 256K, or the model's native maximum if smaller, and steps down until it fits a 100 GB memory budget; see [Memory on a Spark](#memory-on-a-spark).

```bash
sma run Qwen/Qwen3-8B --bf16
sma run nvidia/Qwen3-8B-NVFP4 --nvfp4
sma run unsloth/Qwen3-8B-GGUF --gguf BF16
sma run unsloth/Qwen3-8B-GGUF --gguf UD-Q4_K_XL --reference <bf16-gguf-run-id>
```

Both legs of a pair use the same engine and the same Spark. A pair is safetensors (`--bf16`/`--mxfp4` with `--nvfp4`/`--mixed`). For GGUF, run the repo's full-precision file first, then each quant with `--reference <that run id>`. `run-pair` holds one GPU lock across both legs. A queue pair is one job, pinned to the alias that created the first run, so a different Spark cannot pick up leg 1.

```bash
sma run-pair --bf16 Qwen/Qwen3-8B --nvfp4 RedHatAI/Qwen3-8B-NVFP4
sma queue add-pair --bf16 Qwen/Qwen3-8B --nvfp4 RedHatAI/Qwen3-8B-NVFP4
sma worker
```

`--mxfp4` is for checkpoints that already ship as MXFP4, such as gpt-oss. A native MXFP4 model is its own reference. There is no second quant and no KL unless you compare two runs yourself.

`--mixed` is only for a checkpoint whose counted per-layer or per-group formats include both FP8 and NVFP4. The name `dgpp_mixed` is not enough. `HawkBearPig/flint-FP8-BF16` (dgpp_mixed, recipe fp8) and a dgpp_mixed config with no layer map are not `mixed`, and `--mixed` refuses them. A real FP8+NVFP4 map is still refused on stock vLLM and on stock SGLang: vLLM does not implement `dgpp_mixed`, and SGLang refuses that method too. vLLM does accept compressed-tensors configs that count both FP8 and NVFP4. SGLang refuses every compressed-tensors checkpoint. `--nvfp4` refuses mixed configs and tells you to pass `--mixed`. SMA does not flatten a mixed checkpoint into NVFP4.

## How it works

One `sma run` takes one checkpoint through these stages. Each is recorded, and a crash resumes at the stage that did not finish.

1. **Pick the engine, before the run exists.** The format decides: safetensors go to vLLM, GGUF (`--gguf QUANT`) to llama.cpp. `--engine sglang` picks SGLang for safetensors. With no flag, preflight asks the pinned vLLM image's registry whether it can load the architecture; when it cannot and SGLang's image can, SGLang serves the run. `engine.selected_by` says which (`format`, `flag`, `default`, or `preflight`) and `engine.selection_reason` why. The engine is part of the run's path, so it is settled before the folder is made. A server that later fails to start is not retried on another engine.
2. **Plan memory, before the weight download.** From `config.json` and the repo's file sizes (or the GGUF header, read through ranged requests), SMA works out the bytes of KV each token adds, then the context per stream: 256K, or the model's native maximum if smaller, halved until one stream fits next to the weights and the engine's overhead in the 100 GB budget. What is left becomes the KV pool. Each benchmark gets the largest context that fits the streams it runs at once. A model that cannot fit is refused here, with the numbers. The plan is `manifest.json` `memory_plan`; `sma run ... --dry-run` prints it.
3. **Download and describe the checkpoint.** A pinned commit, only the files the engine loads, hashed. The quant must match the flag; for GGUF, the file's own metadata names it. Parameters are counted from the tensor shapes.
4. **Serve.** The pinned image starts with the plan's context and pool and an 8-bit KV cache: `--kv-cache-dtype fp8` on vLLM and SGLang, `-ctk q8_0 -ctv q8_0` on llama.cpp. SMA then reads the engine's own startup log for the KV type it chose and the pool it allocated, and stops the run if that log reports a 16-bit cache.
5. **Measure.** Determinism (the same prompts at batch 1, 8, and 64), logprobs on a pinned corpus (perplexity, and KL against a reference run), quality suites through the chat API (MuSiQue among them), then KernelBench Level 1, SWE-bench Verified-mini, and BrowseComp-Plus deep research with the model's own thinking mode and sampling (their kernels, tasks, and document search run on lab machines, and a lab judge grades BrowseComp-Plus), then speed on a restarted server, while a 1 Hz sampler records GPU clocks, temperature, power, throttle flags, and available memory.
6. **Record.** `summary.json` holds the headline numbers and the deployment notes: what someone gets running this model this way on a Spark. For example, from a run of Qwen3-0.6B on vLLM on a Spark: "bf16 weights with an FP8 KV cache on vllm: 1.5 GB of weights, 40,960 tokens per stream, 36 full-length streams by the engine's own KV report within 100.0 GB. Peak memory 95.24 GB." Containers are removed on every exit.

How each engine covers each stage:

| | vLLM | SGLang | llama.cpp |
| --- | --- | --- | --- |
| Checkpoint | safetensors (BF16, NVFP4, MXFP4, mixed) | safetensors (BF16, ModelOpt NVFP4) | GGUF files, as published |
| KV cache | FP8, `--kv-cache-memory-bytes`, `--max-model-len` | FP8, `--max-total-tokens`, `--context-length` | Q8_0, one unified pool (`-c`) shared by `-np` slots, each capped at the context |
| Quality server | batch invariant for BF16 (quantized checkpoints run their real kernels), prefix caching off | deterministic inference, radix cache off, DeepGEMM off | no batch-invariant mode; prompt cache on |
| Tool calls | `--tool-parser NAME` | `--tool-parser NAME` | the model's chat template (`--jinja`) |
| Logprobs and KL | `/v1/completions` prompt logprobs, top-64; `sma compare` | `/generate` input top-k; `sma compare` | `llama-perplexity` on the same corpus, full-vocabulary KL against a full-precision GGUF run (`--reference`) |
| Speed | `vllm bench serve` in the server container | `sglang.bench_serving` in the server container | `vllm bench serve` from the vLLM image, in its own container |

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

Python 3.11 or newer; the quality extra needs 3.12. [uv](https://docs.astral.sh/uv/) is the intended installer.

```bash
uv sync --extra quality
# quality is NVIDIA NeMo Evaluator (nel). The launcher itself installs without it.
cp .env.example .env            # then fill it in: HF_TOKEN, SMA_HF_CACHE_DIR, and the lab machines
```

`sma` reads `.env` from the folder it runs in each time it starts, so nothing has to be exported. A variable already set in the shell wins, and `SMA_ENV_FILE` names another file. `.env` holds what belongs to this machine and never changes: the Hugging Face token, the weight cache (`SMA_HF_CACHE_DIR`, which overrides `hf.cache_dir`), the names of the KernelBench service and the x86_64 runner (SWE-bench and BrowseComp-Plus's search), and the judge's endpoint and key (`SMA_JUDGE_URL`, `SMA_JUDGE_API_KEY`). Run records name those machines `${NAME}`, never by value, and never hold a key. `.env.example` lists every variable.

Edit `sma.yaml` if you need to. The host alias defaults to `gpu-host-1`. It must not be a real hostname.

```bash
sma doctor
sma run Qwen/Qwen3-8B --bf16 --dry-run
sma run Qwen/Qwen3-8B --bf16 --tool-parser hermes   # the tool-call parser for this model
sma status
sma watch                       # live view; Ctrl-C to quit
sma list
sma resume <run-id>
sma compare <bf16-run-id> <nvfp4-run-id>
```

`sma watch` redraws every 5 seconds (`--interval`; `--once` prints one screen), one line per item so it fits one terminal page. MACHINES lists each machine SMA uses, by kind and chip and never by name: this Spark (product, GPU, CPU cores, memory and what is free), the KernelBench service (its GPU, CUDA, and torch, from `/v1/environment`; it reports no machine type), the x86_64 runner (a read-only script over SSH), and the judge (the models it serves; its API reports no hardware). Each is asked at start and every minute, and shows as up, down, or not set in `.env`. QUEUE counts the queue's jobs and lists the pending and claimed ones. CURRENT RUN has one line per benchmark, in the order the run does them: determinism, perplexity (and KL for a quant with a reference), each quality suite, KernelBench, SWE-bench, BrowseComp-Plus, and speed. Each shows its score (as a percentage, colored from red at 0 through orange and yellow to green at 100), its progress (a nel suite's answered count from `inference_log.jsonl`; BFCL's from the eval-factory container's per-category result files, without a total since its dataset sets it; KernelBench's kernels written; BrowseComp-Plus's questions answered; SWE-bench's finished tasks of 50 by how each ended, submitted or at the step limit, read from the runner over SSH at most once a minute; and for the agent steps the time since this attempt began, from its `settings.json`, with the attempt number when a step is redone), its error, or queued. Setup and bookkeeping steps (fetch, servers, results, teardown) get a line only while running or failed. SERVING shows prefill and decode tok/s per stream (what one request sees while sharing the server), decode tok/s over all streams (the server's total: generated tokens over wall-clock time), and inter-token latency. Prefill has no all-streams row: prompts take a sliver of the server's time, so prompt tokens over wall-clock time measure how fast prompts arrive, not a speed. Each is averaged from the engines' own counters (`engine_metrics.jsonl` in each run of the last 24 hours) over the last 5 minutes, hour, and 24 hours, after a "since start" column read live from the current server, whose counters run from its start (the all-streams rate there divides by the container's uptime from `docker inspect`). For a run that records no `engine_metrics.jsonl` (one started before SMA recorded them), `sma watch` keeps its own reads of the server, so its windows cover only the time `sma watch` has been open. Each engine's formulas live in its backend (`serving_rates`, which also derives TTFT and all-stream rates where the engine reports them). llama-server adds a request's tokens to its counters when the request ends, so with long requests a 5-minute window is uneven; the hour and the day even it out. FINISHED lists the latest 3 runs that are not running, with their scores. A run is labeled by its steps, not the catalog's status: done (every step succeeded), failed (every step ran, some failed), or stopped (steps left; `sma resume` goes on from them). The catalog calls a run failed as soon as one step is and stamps it finished whenever its process ends, so a run stopped on purpose would otherwise look lost. The current run is the one with a container up, so a run left "running" by a machine that went down is not mistaken for it. `--plan FILE` adds runs a script will start: one `REPO QUANT` per line, in order, and an optional `since TIME` line (ISO time or run id) so older runs of the same checkpoint do not count. Each line shows its newest run since then, or queued. It reads files, `docker ps`, and the services' status endpoints only, and scrubs the screen like a run record (`<host>` for this machine, `${NAME}` for lab machines, no keys), so a screenshot is safe to share.

`sma doctor` checks Docker, the configured image, `nvidia-smi`, that the GPU has no compute process, free disk, that the Hugging Face cache path does not contain the hostname, and that `HF_TOKEN` is set. It does not pull the image and it does not print the token.

Fetch refuses a snapshot path that contains the hostname, because that path goes into the run record. The default cache is under your home directory. On a machine whose user name is also its hostname, that path fails. Set `SMA_HF_CACHE_DIR` in `.env` (or `hf.cache_dir` in `sma.yaml`) to a path outside your home directory.

A dry-run prints the plan and the exact commands. It downloads no weights, starts no container, and writes no run folder. It does read the repo's `config.json` and file sizes from the Hub (for GGUF, the file's header through ranged requests) to print the memory plan: the engine and why, the KV type, the context per stream and why, the memory split, the streams that fit, and each benchmark's context. Offline, or for a repo it cannot read, it says so and prints the commands with placeholders. It also does not prove the checkpoint's architecture is registered.

## What a run does

Each step is idempotent. A `steps/<name>.done` marker is written only after the step finishes. `sma resume <run-id>` skips steps that have a marker, and only on the host alias that started the run. A crash mid-step redoes that step and does not redo finished ones.

A step that fails does not stop the run, except fetch and the quality server, which everything after them needs (owner decision 2026-10-29). The step is marked failed with its error in the manifest, the run goes on to the next step, and at the end `sma run` exits 2 naming the failed steps (`RunIncomplete`); `sma resume` retries just those. Inside quality, a suite that fails gets a `scores.json` with its error and no `.done`, and the other suites still run. So a speed bench that fails still lets KernelBench and BrowseComp-Plus be scored, and a judge outage loses only BrowseComp-Plus's score until a resume.

1. **Fetch.** Resolve the repo to a commit SHA. Read `config.json` and, when present, `hf_quant_config.json`. Refuse a quant mismatch or an unsupported quant/engine pair before the weight snapshot. Pull the engine image and query its digest for the architecture registry. Flint is refused. If the image cannot be queried, an unknown architecture is a warning and the download continues. Then download that commit with `huggingface_hub` (resumable) into a plain folder, `<hf.cache_dir>/<model>/<sha>`. The engine container mounts only that folder at `/model`, so it holds real files, not the Hugging Face cache's links. Hash the weight files. Count total and active parameters from safetensors shapes. Flags are `--bf16`, `--nvfp4`, `--mxfp4`, `--mixed`, and `--gguf QUANT`. Before the weight download, SMA plans memory from `config.json` and the file sizes and refuses a run that cannot fit (see [Memory on a Spark](#memory-on-a-spark)). For `--gguf QUANT`, SMA picks the one file, or the one complete split set, whose path holds that quant as a whole token (`Q4_K` does not match `Q4_K_M`). It reads the GGUF header through ranged requests, plans, then downloads only those files. The file's own metadata gives the quant type (`general.file_type`), the tensor types, the parameter count, the KV layout, the native context, and a tokenizer fingerprint. SMA never makes or converts a GGUF. An unmatched quant names the ones the repo has.
2. **Serve for quality.** The pinned image, then the digest recorded from `docker image inspect` on later starts. vLLM: `VLLM_BATCH_INVARIANT=1` when the architecture allows it and the checkpoint is not quantized (NVFP4, MXFP4, mixed, and FP8 run their real kernels; see Caveats), `--no-enable-flashinfer-autotune`, `--no-enable-prefix-caching`, `--no-async-scheduling`, fixed seed, `--tensor-parallel-size 1`, and `--default-chat-template-kwargs` from `quality.chat_template_kwargs` (thinking off for every quality client). With `--tool-parser NAME`, `--enable-auto-tool-choice --tool-call-parser NAME`. SGLang: `--attention-backend flashinfer`, `--tp-size 1`, `--enable-deterministic-inference`, and `--disable-radix-cache` because FlashInfer determinism does not keep the radix cache, and `--tool-call-parser NAME` with `--tool-parser`. SGLang also runs with `SGLANG_ENABLE_JIT_DEEPGEMM=0`, because DeepGEMM in the pinned image does not support the GB10 and deterministic serving crashed at CUDA-graph capture without it. llama.cpp: `llama-server` from the pinned image with `-ngl all`, `-fa on`, `-ctk q8_0 -ctv q8_0`, one unified KV pool (`-c`, `--kv-unified`) shared by `-np` slots, each capped at the context per stream (`--kv-unified-per-slot`), `--cache-ram` (its host prompt cache, counted in the plan), `--metrics`, `--jinja` for tool calls (no parser flag), and `--chat-template-kwargs` from `quality.chat_template_kwargs`. Every engine gets its context and KV pool from the memory plan and an 8-bit KV cache: vLLM `--kv-cache-dtype fp8 --kv-cache-memory-bytes --max-model-len`, SGLang `--kv-cache-dtype fp8_e4m3 --max-total-tokens --context-length`. After a start, SMA reads the engine's own log for the KV type and the pool it allocated (`engine.kv_cache_reported`); a 16-bit KV cache there stops the run. Health check with a deadline.
3. **Determinism.** The same prompts at batch sizes 1, 8, and 64, on the quality server with prefix caching off. A mismatch is recorded. The run continues. llama.cpp has no batch-invariant mode and caches prompts; the probe still runs there and records what it sees.
4. **Logprobs.** The pinned corpus is cut into 2048-token windows and scored as raw token ids, with no chat template. vLLM uses `/v1/completions` `prompt_logprobs`. SGLang uses `/generate` input-token top-k. Each position stores top-k (default 64), a tail bucket, and the true next-token logprob. Perplexity is `exp(-mean logprob)`. A field the server does not return stays null. llama-server returns no prompt logprobs, so a llama.cpp run stops its quality server and runs llama.cpp's own `llama-perplexity` on the same corpus text instead, with the same Q8_0 cache: `logprobs.llamacpp_chunks` chunks of `logprobs.window_tokens`, scoring the second half of each (1,023 tokens per 2,048-token chunk). A full-precision GGUF (`BF16`, `F16`, `F32`) writes its full-vocabulary logits to `logprobs/kld-base.bin`. A quant started with `--reference <that run>` reads them and writes `kl_vs_reference.json` from llama.cpp's own KL statistics (mean, median and p99 KLD, same-top-token share, RMS Δp, PPL of both). That reference must be a llama.cpp run of the same tokenizer with the same chunk count.
5. **Quality.** One suite at a time against the local server's chat API, at the concurrency in `sma.yaml`: NVIDIA NeMo Evaluator (`nel`) for most suites, and an NVIDIA eval-factory container run directly for BFCL. Per-question rows, scores with an interval, and the harness's own output are saved. Each suite's `scores.json` records the context per stream the memory plan gave quality's streams (`context_per_stream`). It also records `measures`: every measure an lm-eval task scored per question, as a share. For IFEval those are strict and loose accuracy, by prompt and by instruction. The headline is strict by prompt (`prompt_level_strict_acc`), the strictest of the four; model cards often quote a loose or by-instruction one, which can run 10 points higher. By-instruction accuracy counts every instruction across all prompts. The summary repeats `measures` for each suite. MuSiQue (owner decision 2026-10-28) is one of them: 100 seeded questions from MuSiQue-Ans v1.0 dev, each with its own 17 to 20 Wikipedia paragraphs, asked with LongBench's musique prompt word for word and scored by rule with MuSiQue's own `metrics/answer.py` (`src/sma/nel_benchmarks/musique.py`). The headline is answer F1, the best over the answer and its aliases; exact match is `scorer_shares.exact_match`. The dev file comes from the `dgslibisey/MuSiQue` mirror at a pinned revision and must match the official release's SHA-256.
6. **KernelBench.** On the quality server, the model writes a kernel for each of KernelBench's 100 Level 1 problems (shipped in `src/sma/data/kernelbench/` at commit `423217d9`, the one the service runs). The prompt is KernelBench's one-shot CUDA fp32 prompt, sent as a chat with its system prompt, `kernelbench.concurrency` at a time. Requests set no sampling field, so the engine applies the checkpoint's `generation_config.json`, and they pass `enable_thinking` as the chat template's own default (`sma/thinking.py` renders the template to find it). The kernel is KernelBench's `extract_first_code` after the reasoning block. All kernels go to the service at `KERNELBENCH_URL` as one job, and the step ends: the service compiles, checks, and times them on its own GB10 while the run goes on.
7. **SWE-bench.** mini-swe-agent's own `swebench.yaml` runs the 50 Verified-mini tasks on the x86_64 runner, `swebench.workers` agents at once, `swebench.step_limit` steps each, against this run's server. SMA logs in with `SWE_RUNNER_HOST`, `SWE_RUNNER_USER`, and `SWE_RUNNER_SSH_KEY`, writes its settings next to the run's folder there (the model, the port, the step limit, the template's thinking default, and `cost_tracking: ignore_errors`, because LiteLLM has no price for a checkpoint served here and mini-swe-agent otherwise stops every task on its first call; no sampling field), forwards the server's port to the runner's loopback over SSH, and starts the agent detached, so a dropped connection does not stop it. A task that ends in anything but the model's own result (`Submitted`, `LimitsExceeded`, a format error, a full context) lost the model; SMA drops those and launches once more. Then the SWE-bench harness grades the patches there. SMA brings back the patches, the harness report, each task's exit status, the logs, and the trajectories, scrubbed. The score is resolved over 50, null if any task still has no answer. The agent acts through a bash tool call, so the run needs `--tool-parser`.
8. **BrowseComp-Plus.** Deep research over a fixed set of 100,195 web documents (owner decision 2026-10-28). On the quality server, the model answers 100 of BrowseComp-Plus's 830 questions, the same seeded 100 for every model, `browsecomp.workers` at a time. Each question runs as the repo's chat client with native tool calls (`search_agent/glm_zai_client.py` at `04694903`) runs it with its defaults: its prompt without a system message, one `search` tool returning the top 5 BM25 hits, each cut to its first 512 tokens, one 20,000-token output budget for the whole conversation, and up to 100 replies. A question ends when a reply calls no tool; it counts as answered only if that reply stopped on its own. Requests set no sampling field and pass the chat template's own `enable_thinking`, as for KernelBench. The index and its search server run on the x86_64 runner (`services/browsecomp-plus`, the `SWE_RUNNER_*` machine), so the document store takes no memory or CPU from the Spark. SMA starts the server there, reaches it through an SSH forward, and stops it after the last question. SMA departs from the client in three ways, recorded in `settings.json`: reasoning is read from vLLM's `reasoning` field as well as `reasoning_content` and kept out of the history, as the client intends; tool arguments that are not JSON go back to the model as a tool error instead of ending the whole run; and a request the server refuses, such as one past the context, ends that question as wrong. The questions and answers are read from the runner and kept only in memory and in git-ignored files. The run needs `--tool-parser`.
9. **Speed.** The server is restarted with batch invariance off. A fresh speed server, including on resume, runs the warm-up soak again. Each attempt's bench command writes `speed/cells/<cell>/attempt-<n>/bench.json`, which is the file SMA reads. Both benches run with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and both pass `--tokenizer /model` so the tokenizer is the pinned snapshot. vLLM keeps `--model` as the served name and uses `vllm bench serve` with ttft, tpot, itl, and e2el at 50, 90, and 99. SGLang passes `--model /model` as well, and uses `sglang.bench_serving` with `--dataset-name random`, a pinned local dataset at `/sma-run/speed/sglang-random-dataset.json` (so the container does not download ShareGPT), `--random-range-ratio 1.0`, `--seed`, and `--output-details`. `speed.num_prompts` above that file's row count is refused. A bench that finishes fewer prompts than `num_prompts`, or reports request errors, marks the cell `bench_failed` and retries it once after a backoff, with no cool-down. Only a thermal or power throttle is `throttled` and waits for the GPU to cool. A telemetry gap is `telemetry_gap` and is not retried. That tool's summary, in tag `v0.5.10.post1` (`python/sglang/bench_serving.py`), has median and p99 TTFT/TPOT, median/p95/p99 ITL, and median/p90/p99 end-to-end latency (`median_e2e_latency_ms`, `p90_e2e_latency_ms`). It does not emit p90 TTFT, TPOT, or ITL. SMA computes those from the per-request arrays when they are present, and otherwise stores null with a reason. The file is JSONL and is parsed from the last JSON object, so a retry cannot glue two objects into one invalid document. SGLang serve passes `--enable-metrics`, `--max-running-requests` (the same cap as vLLM `max_num_seqs`), and `--random-seed`. A failed `/metrics` scrape stores null counters and a reason. It does not fail the cell. llama.cpp's image has no bench client, so a llama.cpp run starts `vllm bench serve` from the pinned vLLM image in its own container (`<run>-bench`, host network, the host user, offline) against llama-server, with the same arguments as for vLLM and `--tokenizer /model/<file>.gguf` (the GGUF's own tokenizer). When the vLLM image's transformers cannot read the GGUF's architecture (`lfm3` on its transformers 5.6), the bench loads the tokenizer of the source model the GGUF names (`general.base_model.0.repo_url`) instead, downloaded at a pinned commit and mounted at `/bench-tokenizer`, and only if every token id and BPE merge matches the GGUF's and the GGUF's other ids are unused padding; otherwise speed is refused. The bench only builds prompts of set lengths and counts tokens with it. `engine.bench_tokenizer` records which tokenizer it used and why. The manifest records that image as `engine.bench_image`.
10. **KernelBench results.** SMA waits for the service's job (`kernelbench.wait_s`), stores its results, and scores them with KernelBench's own `score.py` formulas: correctness over 100, and fast_p, the share correct and faster than the stored eager-PyTorch time by more than p, over the problems with a stored time (99: problem 95 fails in KernelBench itself). The headline is fast_1. Compare scores only within one service fingerprint.
11. **BrowseComp-Plus results.** After speed, the judge grades each answer that finished: `SMA_JUDGE_URL` and `SMA_JUDGE_API_KEY` from `.env`, model `browsecomp.judge_model`, Ornith 1.5 122B on the lab's V100s ([`judge-info.md`](judge-info.md)), so the judge runs on lab hardware and a judge outage cannot hold up the measurements. The prompt is BrowseComp-Plus's leaderboard grader (`scripts_evaluation/evaluate_run.py`), word for word, sent one at a time with thinking off, temperature 0, 4,096 tokens, and no prompt cache; its `correct: yes|no` line is read with the repo's own parser. The headline is accuracy over the 100: an unfinished question, or a judge reply without a verdict, counts as wrong, as in BrowseComp-Plus, and the run warns with the count. Evidence recall (the share of each question's evidence documents its searches returned), gold recall, and search calls need no judge. Calibration error is null: the repo's formula needs at least 100 judged confidences and drops its last bin. `per_query.jsonl` keeps numbers and verdicts only; the conversations, the judge's replies, the questions, and the answers go into the git-ignored `trajectories.jsonl.gz`, whose hash the manifest keeps. Compare scores only within one judge.
12. **Telemetry.** A 1 Hz `nvidia-smi` sample of SM clock, temperature, power, and throttle reasons for the whole run. A real thermal or power slowdown marks that speed cell invalid and retries it once after a cool-down. `sw_power_cap` is always stored. It invalidates a cell only when `telemetry.sw_power_cap_invalidates` is true, because a GB10 near 140 W can show that flag in steady state. A cell is `unverified` unless it has about `duration / interval` samples (`telemetry.min_sample_ratio`, default 0.8) and its largest gap is no bigger than `telemetry.max_sample_gap_s`. `[N/A]` flags are unverified too. The sampler's failure counter is not used, because a later success resets it. Unverified cells are not retried. Invalid and unverified cells stay on disk and are left out of the headline tok/s.
13. **Teardown.** The container is removed on success, failure, Ctrl-C, and SIGTERM. The page cache is dropped when the process is allowed to write `/proc/sys/vm/drop_caches`. The run then waits until the GPU is cool, or until the idle timeout.

Headline tok/s uses `speed.derive_stat` (`p50` by default, or `p90` / `p99`):

- prefill tok/s = input length / TTFT at concurrency 1
- decode tok/s = 1 / TPOT at concurrency 1

If that percentile is absent, the derived number is null and the cell records why. SMA does not invent it.

`sma compare <reference> <candidate>` computes per-token KL with an epsilon floor (`1e-12`) instead of dropping positions. A probability that was actually zero is counted in `kl_infinite_positions` before the floor. `kl_floored_positions` counts every position that hit the floor, including those zeros and merely tiny positive masses. It also records top-1 agreement and delta perplexity, and it records k. It accepts a BF16 reference against a `mixed` candidate when the tokenizer, corpus, and engine match. It refuses a different engine unless you pass `--allow-cross-engine`, and that result is labeled `CROSS-ENGINE`. Different image digests are labeled `IMAGE DIGEST` and are not refused. It refuses when the tokenizer fingerprints differ, or when the token-id windows differ.

`kl_vs_reference.json` also reports the statistics of llama.cpp's `--kl-divergence`: `kl_mean_se`, `kl_quantiles` (p50, p90, p95, p99, p99.9, max), `top1_agreement_se`, `ln_ppl_ratio` with `ln_ppl_ratio_se`, and `rms_delta_p` on the true token. Standard errors are cluster-robust with one cluster per window, because positions in a window share context. The KL is coarse-grained (reference top-k ids plus one tail bucket), so it is a lower bound on full-vocabulary KL. On a GB10 with batch invariance on, two BF16 runs of Qwen3-0.6B gave identical logprobs at all 98,256 positions, so the BF16-vs-BF16 noise floor there was exactly zero.

For MoE NVFP4, MXFP4, and mixed on vLLM, SMA always passes `--moe-backend`. The chain is `flashinfer_b12x`, then `marlin` if that server never becomes healthy. Quality and speed each record the backend that started. Auto-selection is not used. Dense NVFP4 sets `VLLM_NVFP4_GEMM_BACKEND` from `engine.linear_backend` (default `flashinfer_b12x`, passed as `flashinfer-b12x`): vLLM 0.21 has no `--linear-backend` flag and refused it on a Spark. BF16 does not get the FP4 MoE backend. SGLang does not take that chain. Its MoE fields stay unset unless a log line names a backend. SMA does not invent one.

NVFP4 on vLLM includes ModelOpt W4A4 and compressed-tensors `nvfp4-pack-quantized` when the activation scheme is also 4-bit float (for example a group whose `input_activations.num_bits` is 4). A compressed-tensors group with fp4 weights and `input_activations: null` is `nvfp4_w4a16`. `--nvfp4` refuses it and will not label it `nvfp4`. SMA does not serve W4A16. SGLang NVFP4 is ModelOpt checkpoints. SGLang refuses compressed-tensors NVFP4, FP8, and mixed FP8/NVFP4, and the error says to use vLLM.

## Watching a run

The terminal shows one readable line per event: each step as it starts and finishes, retries, warnings, and server startup. `logs/sma.log` in the run folder keeps the same events as JSON, one object per line.

The engine's own output goes to `logs/server-quality.log` and `logs/server-speed.log`. SMA copies it there live while the server runs, so `tail -f` shows what the engine is doing. Lines appear as they arrive, so stdout and stderr are interleaved. Two lines written at the same instant can swap. While a server starts, the terminal prints a progress line about every 15 seconds with the newest line of that log. If the container exits before it is healthy, SMA stops waiting at once and puts the last log lines in the error.

If a server stops while a step is sending it requests, the step error says so, with the exit code and the last server log lines. A clean shutdown with no error in the log means something on the host stopped the server. On a Spark that runs earlyoom, check `journalctl -u earlyoom`: it stops `vllm` and `python` processes when free memory falls under its floor (8% on our Spark). The memory plan keeps a run's footprint under `memory.budget_gb` for that reason.

## Queue

Up to three Sparks can run `sma worker` against one queue directory. The directory has to be on a single filesystem because a claim is an atomic `link`. The worker takes the GPU lock before it claims. If the lock is busy it does not take a job. Each claimed job is heartbeated until it finishes. One failed model is marked failed (or returned to pending while `attempts` is under `queue.max_attempts`) and the worker continues.

A pair is one claim. The alias that created the first run is stored on the job and is not cleared when the claim is reclaimed. Leg 1 runs on that same Spark. `sma resume` refuses a run whose `host_alias` is not this process. If that Spark is gone, `sma queue unpin JOB` or `sma queue reassign JOB --host ALIAS` clears the run ids and starts the job fresh on the next host. It does not resume the old run on a different machine. A failed attempt waits (`queue.retry_backoff_s`) and then sorts behind jobs that have not failed. A pending file that is not valid JSON is logged and moved to `corrupt/`.

```bash
sma queue add Qwen/Qwen3-8B --bf16
sma queue add org/model --nvfp4 --engine sglang
sma queue add Qwen/Qwen3-8B --bf16 --tool-parser hermes
sma queue add unsloth/Qwen3-8B-GGUF --gguf UD-Q4_K_XL
sma queue list
sma queue reclaim          # return a claim whose heartbeat is stale
sma queue unpin <job-id>
sma queue reassign <job-id> --host gpu-host-2
sma worker --once
```

`gpu.lock` stops two SMA processes on the same Spark from starting two models. `run-pair` holds that lock for both legs.

## Run record contract

```
runs/<model-slug>/<engine>/<quant>/<run-id>/      a GGUF run's <quant> is gguf-<QUANT>, for example gguf-UD-Q4_K_XL
  manifest.json                 schema version 2; memory_plan, gguf, engine.kv_cache_*
  summary.json                  flat headline numbers and deployment notes; null means not measured
  steps/<step>.done             the resume marker
  fetch/result.json             SHA, hashes, params, detected quant, composition
  determinism.json
  logprobs/dump.json            per-token top-k, tail, true-token logprob (not in git)
  logprobs/corpus.json          dataset, revision, sha256, license
  logprobs/perplexity.json
  logprobs/kl_vs_reference.json only when a reference run was supplied
  logprobs/corpus.txt, command.json   llama.cpp: the text and the llama-perplexity command
  logprobs/kld-base.bin         llama.cpp full-precision GGUF: logits a quant's KL reads (not in git)
  quality/<suite>/nel.yaml          the NeMo Evaluator config SMA wrote
  quality/<suite>/run_config.yaml   the eval-factory container's config (BFCL)
  quality/<suite>/<grader>.py       a grader SMA ships, when the suite uses one
  quality/<suite>/attempt-<n>/      nel's output; a retry writes a new attempt
  quality/<suite>/scores.json       the headline (pass@1, or a named mean such as MuSiQue's answer_f1), interval, question count, seed
  quality/<suite>/samples.jsonl     one row per question
  quality/kernelbench_l1/settings.json, generations/<id>.json   thinking, sampling, each answer and its kernel
  quality/kernelbench_l1/job.json, results.json   the service job, its environment, its per-kernel results
  quality/swebench_verified_mini/settings.json, tasks.jsonl, preds.json, report.json   agent settings, exit statuses, patches, the harness report
  quality/swebench_verified_mini/trajectories.jsonl.gz   every agent conversation (not in git)
  quality/browsecomp_plus/settings.json, per_query.jsonl, scores.json   the protocol and the picked ids; per question: status, searches, recall, verdict (numbers only)
  quality/browsecomp_plus/trajectories.jsonl.gz   every search conversation and judge reply, with the questions and answers (not in git)
  speed/summary.json
  speed/cells/<cell>/attempt-<n>/bench.json
  speed/cells/<cell>/attempt-<n>/metrics.json
  telemetry.csv
  memory.csv                    1 Hz MemAvailable and MemTotal, for the measured peak
  engine_metrics.jsonl          the engine's /metrics counters every 15 s while a server is up (telemetry.engine_metrics_interval_s), with the step
  logs/
  cleanup.json
runs/catalog.sqlite              fast index, including engine (not in git; rebuilt from the manifests)
runs/index.json                  atomic snapshot of the catalog (not in git)
runs/compares/<ref>__<cand>.json output of sma compare
```

Run folders are committed to the repository. They are the data SMA submits and keeps. Git ignores only what cannot or must not go in:

- `logprobs/dump.json` (about 300 MB), `logprobs/kld-base.bin` (about 2.5 GB for a 0.6B model), and SWE-bench's `trajectories.jsonl.gz` are too big for git. They stay in the run folder on the machine that ran it. The manifest's `large_files` keeps each one's sha256 and size. `sma compare` and a llama.cpp KL reference check a copy against that hash and refuse one that does not match.
- BrowseComp-Plus's `trajectories.jsonl.gz` holds its questions and answers, which BrowseComp-Plus asks never appear as plain text online, so it stays out of git the same way. While a run is between its two BrowseComp-Plus steps, the per-question files under `quality/browsecomp_plus/queries/` hold them too; git ignores that folder, and the second step bundles it and removes it.
- `runs/catalog.sqlite` and `runs/index.json` are derived from the manifests. On a fresh clone, `sma list` rebuilds them.
- The repository root's `.env` holds this machine's settings and the lab machines' names (below). Git ignores it.

Check a run folder for hostnames, lab names, and keys before you commit it. SMA scrubs what it writes, but an engine's own log can print anything.

`runs/gb10/` and `runs/cpu/` hold baselines that are not SMA runs. They were measured on the lab's benchmark services, and each README says how to call its service during an eval:

- [`runs/gb10/baseline/kernelbench/`](runs/gb10/baseline/kernelbench/README.md): KernelBench levels 1–3 timed in eager PyTorch on a separate Spark's GB10 (fp32, `cuda_event`, 100 trials, clock locked at 2200 MHz). Compare speedups only within one environment fingerprint.
- [`runs/cpu/baseline/sweverified/`](runs/cpu/baseline/sweverified/README.md): SWE-bench Verified-mini's 50 tasks graded with their gold patches on the x86_64 runner (50 of 50 resolved). This shows every task image and test works there.

Machine names in those files are `${NAME}` placeholders. Their values are in the root `.env`, which git ignores: `set -a; . ./.env; set +a`.

Older runs that live at `runs/<model-slug>/<quant>/<run-id>/` are still found by `sma list`, `sma status`, `sma resume`, and `sma compare`. New runs are engine-separated. A resumed run keeps the engine it was started with.

Schemas: [`schemas/manifest.schema.json`](schemas/manifest.schema.json) and [`schemas/summary.schema.json`](schemas/summary.schema.json).

JSON is written to a temp file, fsynced, and renamed. A crash does not leave a half-written manifest or summary.

The manifest is the subject record: model repo, revision SHA, weight hashes, quant, mixed composition when the config exposes layer or group counts, architecture check, parameter counts, engine name, image and digest, every server flag that was passed, the env vars SMA set, the MoE backend for quality and for speed, the SMA git commit, the host alias, timestamps, step status, and the sha256 and size of the files kept out of git. It does not contain the machine hostname or the HF token. `summary.json` uses null for anything that was not measured. It does not fill missing scores with zero. `quant` in the index is `mixed` for a mixed checkpoint.

Energy in `summary.json` is the sum over valid speed cells of the GPU's cumulative energy counter (NVML `nvmlDeviceGetTotalEnergyConsumption`), read just before and just after each bench command (`speed/windows.json` `energy_j`). That is how Zeus, AIPerf, and the ML.ENERGY benchmark measure energy. On a GB10 it updates more often than 4 times a second; under a steady 82 W load it read 3.4% below a 4 Hz power integral. When the counter cannot be read, that cell falls back to the 1 Hz `nvidia-smi` power integral, and `energy.method` says which was used (`mixed` when both). A gap longer than `telemetry.max_sample_gap_s` (default 2.5 s) is not bridged in the fallback. The 1 Hz sampler still records clocks, temperature, and throttle reasons for every cell.

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

`sma.yaml` in the working directory, or `--config`. Unknown keys are an error; `engine.max_model_len` and `engine.kv_cache_gib` say the memory plan replaced them. `--quantization` and `--tensor-parallel-size` other than 1 are refused, including inside `engine.extra_args`, and so are the KV type, context, and pool flags of all three engines (`--kv-cache-dtype`, `--max-model-len`, `-ctk`, `-c`, and the rest), which the memory plan sets. SGLang attention must stay `flashinfer` (required on SM12x). `--tp-size` is forced to 1.

The vLLM tag is pinned (`nvcr.io/nvidia/vllm:26.05.post1-py3`). The SGLang tag is pinned (`lmsysorg/sglang:v0.5.10.post1-cu130`, the versioned arm64 / CUDA 13 tag, not `latest-cu130`). The llama.cpp tag is pinned (`ghcr.io/ggml-org/llama.cpp:full-cuda13-b11913`, arm64 / CUDA 13 with SM 12.1 kernels; the "full" image has `llama-server` and `llama-perplexity`), and the run records llama.cpp's own build (`engine.build_info`). Set `engine.image_digest`, `engine.sglang_image_digest`, or `engine.llamacpp_image_digest` when you already know a digest. Otherwise SMA records the digest `docker image inspect` returns and uses `image@sha256:…` for later starts in that run, including the other leg and a restart. It does not invent a digest.

### Memory on a Spark

The CPU, the GPU, the OS, and every process share one pool of unified memory, 130.7 GB on a GB10. An out-of-memory guard such as earlyoom stops `vllm` and `python` processes when free memory falls under its floor, and that looks like an engine crash. So SMA plans each run's memory before it starts anything, and refuses a run that cannot fit ([docs/serving-design.md](docs/serving-design.md)).

- **Budget.** Weights, the KV pool, and the engine's own overhead stay under `memory.budget_gb` (100 GB). `memory.overhead_gb` holds the overhead per engine, measured on a Spark (see docs/first-spark-run.md): the engine's activations, CUDA graphs, and server processes, plus SMA's tools beside it (nel, the bench client). llama.cpp also sizes its compute buffers by its whole KV cache (the 16-bit attention mask for one 512-token micro-batch, on the GPU and again on the host, and one attention layer's K and V converted to 16 bits for flash attention: 6,144 bytes per token for Qwen3, 4,096 for LFM3), and keeps a prompt cache in host RAM (`engine.llamacpp_cache_ram_mib`, 8 GiB, llama.cpp's default, passed as `--cache-ram`). The plan counts both.
- **KV cache.** Always 8-bit: FP8 on vLLM and SGLang, Q8_0 on llama.cpp (34 bytes per 32 values). Never 16-bit. `engine.kv_cache_scales` records where the scales come from: the checkpoint, when it declares FP8 KV scales, or the engine's default of 1.0. It adds what the engine's log says it used: `nvidia/Qwen3-8B-NVFP4` declares FP8 KV scales, and all 72 are 1.0, so vLLM reports a scale of 1.0.
- **KV per token.** From `config.json` or the GGUF metadata: K and V values in the layers that attend to every token. Sliding-window layers count only their window. Linear-attention, convolution, and Mamba layers keep a small fixed state per stream instead (an estimate). Hybrid models such as Qwen3.6 or Nemotron-H need a fraction of a dense model's KV.
- **Context per stream.** 256K (`memory.start_context`), or the model's native maximum if smaller, then halves until one stream fits next to the weights. Native means without RoPE stretching. If a config ships with stretching on, the native maximum is its original length. SMA never turns stretching on.
- **KV pool.** Everything left in the budget. vLLM gets it as `--kv-cache-memory-bytes`, SGLang and llama.cpp as tokens. llama.cpp's pool holds its slots (`-np`) at the context per stream and no more, because no request can use more and its buffers grow with the whole cache. On 2026-10-28 a pool of the rest of the budget gave LFM3-1.7B 11.6 million tokens for 8 slots of 262,144, its buffers took 47 GB the plan did not count, and the Spark ran out of memory. `engine.gpu_memory_utilization` no longer sizes it; it is vLLM's startup free-memory check and SGLang's static share.
- **Per benchmark.** Each benchmark gets the largest step that fits the streams it runs at once: determinism 64, logprobs 1, quality `quality.concurrency`, speed the highest sweep concurrency. If even its least context (`memory.quality_min_context` for quality, the window for logprobs, the longest speed shape) does not fit, the run is refused before launch.

The plan is in `manifest.json` `memory_plan`. `summary.json` `deployment` says what someone gets running this model this way: the engine, the weight format, the KV type, the context per stream and why, and, computed before launch, the weight and pool sizes and the streams that fit at full context and at 32K. Measured during the run: the pool the engine reported, the streams it holds, the peak memory (`memory.csv`, the drop in available memory below the run's first sample), and output tok/s at 1, 4, and 8 streams. Its `sentence` states it in one line.

Quality runs NVIDIA NeMo Evaluator 0.3.0 (`nel eval run`) against the quality server's chat API (`/v1/chat/completions`). For each suite SMA writes `quality/<suite>/nel.yaml` and runs nel in that folder. Every request is greedy (temperature 0, `seed`), with `quality.max_tokens`, and carries `chat_template_kwargs: {"enable_thinking": false}` (`quality.chat_template_kwargs`). nel's `reasoning` interceptor moves a leading `<think>` block out of the answer before it is scored. SMA decodes greedily with `quality.max_tokens` 4096, and Qwen3's model card says not to decode thinking mode greedily and to allow 32,768 output tokens, so thinking-mode scores would be truncated. A chat template that does not read `enable_thinking` ignores it. Failed requests are retried `retries.attempts` times in all, each with `quality.request_timeout_s`. That limit reaches nel's model client too: nemo-evaluator 0.3.0 builds it with a fixed 120 s timeout that no config sets, so SMA runs nel's own console script under its own Python after raising that floor (`commands.NEL_CLIENT_TIMEOUT`); `command.json` records it as `client_timeout_s`. A `nel` that is not a Python script is refused. Before this, LFM3-1.7B's 4,096-token MuSiQue answers (about 123 s on a Spark) were dropped at 2 minutes and the suite's score was null.

A suite's `bench` is a nel benchmark: a built-in name (`mmlu_pro`, `gpqa`), `lm-eval://<task>` for a generative lm-eval task, or `sma:<name>` for a grader SMA ships in `src/sma/nel_benchmarks/`. SMA copies that file into the suite folder, so the grader that scored a run stays with it. `limit` picks that many questions by a seeded shuffle of the whole dataset (`shuffle_seed` = `seed`), so every model and quant answers the same questions. `scores.json` holds `pass@1`, its bootstrap interval, the question count, the failure count, and the dataset size. nel scores a request that never got an answer (a timeout or a server error: its `infra_error` and `graceful` failures) as wrong; SMA does not fill a missing score with zero, so a suite with any unanswered request has a null score and an `error` that says how many, and nel's number is kept as `pass_at_1_failures_as_wrong`. A reply nel categorizes as a refusal is an answer and stays in the score. The summary copies the score, the interval, and the count. `samples.jsonl` has one row per question (`problem_idx`, the response, the extracted answer, the reward); the full conversation stays in nel's `trajectories.jsonl`. nel writes its absolute output path into some files; SMA removes the hostname from every text file nel wrote, also when nel fails.

GSM8K uses `sma:gsm8k`: nel's own prompt and dataset with a numeric answer check. Both GSM8K graders in nemo-evaluator 0.3.0 mark answers wrongly. On a Spark run of Qwen3-0.6B (50 questions), the built-in grader read "The answer is: 140" as ": 140" and scored 0 of 50, and `numeric_match` strips trailing zeros (140 became 14, 2500 matched 25). SMA's check takes the first number after the last "answer is" or "answer:", or else the last number, and compares the values. It scored 27 of 50, the same as a hand count.

BFCL v3 (`bfcl_v3`, Berkeley Function Calling Leaderboard) runs NVIDIA's eval-factory container (`nvcr.io/nvidia/eval-factory/bfcl:26.03`, arm64 build) with task `bfclv3_ast`: single-turn, live, and multi-turn function calls, sent to the chat API with native tool calling, so the server's tool-call parser is part of what is scored. SMA runs the container itself, not through nel, because nel runs such containers in a temporary folder and deletes BFCL's per-question files. The image is pinned by the digest `docker image inspect` returns. The container runs with host networking (127.0.0.1 is the quality server), as the host user, and with `NEMO_EVALUATOR_TELEMETRY_LEVEL=0`; without that it reports the model name and task to NVIDIA. Its `limit` keeps the first questions in file order, so SMA runs all 4,441. The score is accuracy over every question, with a 95% Wilson interval; `samples.jsonl` has one row per question (`category`, `id`, `correct`). BFCL's own group scores (percent) are kept as `harness_metrics`; they are not the leaderboard's overall score. The quality server has prefix caching off, and BFCL repeats long tool lists every turn, so the multi-turn questions are slow.

A suite with `tools: true` needs the server's tool-call parser. It depends on the model (`hermes` for Qwen3, `llama3_json` for Llama 3, `qwen3_coder` for Qwen3-Coder on vLLM), so pass `--tool-parser NAME` to `sma run`, `run-pair`, `queue add`, or `queue add-pair`, or set `engine.tool_call_parser`. The manifest records it, and a resume keeps it. Without a parser, that suite records a null score with the reason, and the other suites run.

The `quality` extra installs `nemo-evaluator[lm-eval]` and `lm-eval[ifeval]` (`langdetect`, `immutabledict`, `nltk` for `lm-eval://ifeval`). lm-eval's log-likelihood tasks (multiple choice scored by probabilities, such as `global_mmlu`) cannot run over a chat API; nel refuses them.

## Caveats

- Batch invariance is beta, and SMA uses it only for unquantized checkpoints on vLLM. On a Spark (2026-10-27) it started with an FP8 KV cache on BF16 Qwen3-0.6B and the determinism probe passed. For quantized checkpoints vLLM's batch-invariant mode replaces the real kernels: it logged "VLLM_BATCH_INVARIANT forces NVFP4 linear to use the emulation backend" and served NVFP4 Qwen3-8B by dequantizing to BF16, at about 2 tokens per second per request, so its quality and KL numbers were not what an NVFP4 user gets. By owner decision (2026-10-27), NVFP4, MXFP4, mixed, and FP8 checkpoints run their real kernels with batch invariance off; `engine.batch_invariant_warning` says so, and the determinism probe records the batching noise that remains.
- Turning batch invariance on is not enough if async scheduling is still composing the next batch. vLLM quality mode also passes `--no-async-scheduling`. On a Spark (2026-10-27) vLLM 0.21 accepted it with batch invariance on.
- MoE backend auto-selection on SM121 can pick Marlin, and Marlin NVFP4 on SM121 has crash reports. vLLM MoE NVFP4 and MXFP4 set `--moe-backend` explicitly, then try `marlin` if `flashinfer_b12x` never becomes healthy. That is step 1 of the arena plan's NVFP4 chain. If every backend in the chain fails, the step is failed, not skipped. SMA does not quantize and does not switch to SGLang as a fallback.
- Some Spark units throttle or power off under a sustained load. Throttled speed cells are marked invalid. They are not deleted. `sw_power_cap` is recorded and, by default, does not invalidate the cell.
- Top-k KL on vLLM and SGLang is an approximation. The tail is one bucket. If a reference top-k id is missing from the candidate's stored top-k, that mass is folded into the tail and the position is labeled approximate. Positions that would have been infinite KL stay in the mean with probability floored at `1e-12`, and the count is `kl_floored_positions`. llama.cpp's KL is llama-perplexity's full-vocabulary KL over fewer tokens (8 chunks × 1,023 by default), and its perplexity scores the second half of each chunk. Neither is comparable with the vLLM numbers; `sma compare` refuses llama.cpp runs.
- The KV cache is 8-bit on every leg, so the BF16 leg is BF16 weights with an FP8 KV cache, not a 16-bit reference. KL between two legs measures the weights with the same KV type. A BF16 checkpoint has no calibrated KV scales; vLLM and SGLang use a scale of 1.0, which is what a production deploy with `--kv-cache-dtype fp8` gets too.
- The memory plan is computed. Engine overhead (`memory.overhead_gb`) was measured on a Spark with small models; a vision tower or a long-context activation peak can need more. The measured peak in `deployment.measured` is the check. vLLM pads hybrid models' KV pages, so its reported pool can differ from the plan.
- A config that ships RoPE stretching on gets its original length as the native context. gpt-oss ships YaRN with an original length of 4,096, so SMA plans it at 4,096 tokens. DeepSeek-style configs are the same.
- llama.cpp has no batch-invariant mode and caches prompts, so its quality and determinism numbers carry batching noise the vLLM quality server does not have. Its determinism probe failed on Qwen3-0.6B on a Spark, as expected.
- llama-server dropped one connection out of 32 at 8 concurrent streams in one speed cell on a Spark (`ServerDisconnectedError`). The cell was marked `bench_failed` and passed on its retry.
- A GGUF's sliding-window layers are counted as full attention when the file names no window pattern (gemma3, gpt-oss); the plan then overestimates their KV and fits fewer streams.
- Tokenizers must match. `sma compare` refuses otherwise. Tokens per answer are not a shared unit across models.
- NeMo Evaluator scores are not the arena's pinned scorers. The default suite list is small and limited (ifeval 50, gsm8k 50, mmlu_pro 700, gpqa_diamond 20 of 198). Twenty GPQA questions give an interval about 25 points wide, too wide to separate two quants. The `gpqa_diamond` suite runs nel's built-in `gpqa`, whose dataset is gated on Hugging Face; accept the terms with the account whose token is in the environment or the Hugging Face token file. Clear `limit` to run a whole dataset. The arena's full suite set is not the default.
- KernelBench and SWE-bench run with the model's own sampling and thinking mode (owner decision 2026-10-27), not KernelBench's temperature 0, so a rerun can score a little differently and the numbers are not KernelBench's greedy leaderboard numbers. A GGUF has no `generation_config.json`; llama-server's own sampling defaults apply. SMA renders the chat template as transformers does, with its `generation` tag (LFM3's template uses it), to read the default. When the chat template's thinking default cannot be read, no `enable_thinking` is sent and the quality server's setting (off) applies; `settings.json` says which.
- KernelBench's fast_1 counts a kernel faster than the baseline by any margin. The unchanged reference code scored fast_1 on the service (58.8 ms against 58.9 ms), so read fast_1 near 1.0× as noise.
- Each benchmark's recorded context is the plan's step for its streams. The server enforces only its own context, so an answer or an agent that runs longer than its step still runs, and the server may queue other requests behind it.
- Stopping a SWE-bench run ends its processes on the runner, but mini-swe-agent's task containers (`minisweagent-*`, `sleep 2h`, `--rm`) can stay up to two hours.
- BrowseComp-Plus scores are not comparable with its leaderboard. SMA asks 100 of the 830 questions, grades with the lab's Ornith 1.5 122B at temperature 0 instead of Qwen3-32B at 0.7, and runs every model through the repo's native-tool chat client, while the published small-model rows used other clients (Qwen-Agent's text tool calls for Qwen3-32B, the Responses API for gpt-oss). For orientation, the leaderboard's BM25 accuracy: Qwen3-32B 3.6%, gpt-oss-20b at high effort 21.5%, Tongyi-DeepResearch-30B-A3B 34.0%. General models score near zero, and over 100 questions a score's 95% interval is about ±9 points at 30%.
- The judge check behind the choice of Ornith (`judge-info.md` section 7) used answers from GPT-5 and o3, which follow BrowseComp-Plus's answer format. A judge reply without a verdict counts as wrong, as in BrowseComp-Plus; the run warns with the count.
- MuSiQue has no published score for small chat models in SMA's setting (all paragraphs in the prompt, zero-shot, greedy, thinking off). Published numbers use other prompts, subsets, or retrieval, and vary widely; compare MuSiQue scores only between SMA runs.
- Speed repeats default to 1, plus a warm-up. The arena plan's "at least 3 runs" is `speed.repeats`.
- Energy is GPU only (`scope: gpu`): the NVML counter per valid speed window, with the 1 Hz power integral as a fallback. On a DGX Spark the CPU shares the 140 W SoC budget and is not counted, and there is no on-box system-power counter. It is not joules per correct answer and not a wall-meter reading.
- The bandwidth ceiling in the summary uses the configured 273 GB/s spec constant and approximate bytes per parameter. It is not a measurement. NVFP4 scale overhead is not included. Mixed checkpoints have no single byte width, so that ceiling is null.
- The determinism probe sends one completions request with N copies of each prompt, with prefix caching disabled on the quality server. The client cannot force the scheduler to fuse those copies into one batch. `max_num_seqs` is at least 64 so a batch of 64 is allowed.
- Logprobs go through the server that is already up. A second offline `LLM()` process is not started, because the quality server already holds the GPU.
- SGLang measurements that the bench JSON or `/metrics` scrape does not contain are null, with a reason on the cell or in `metrics.json`. They are not copied from vLLM's metric names.
- SMA does not apply arena fidelity verdicts (broken / degraded / faithful / lossless). It records KL, top-1, and perplexity. Thresholds belong to the arena.

## Hardware checks

The test suite uses a fake HTTP server and a fake container runtime. It needs no GPU, and it does not pull an image, start an engine, run NeMo Evaluator, or read a GPU sensor.

These were run on a DGX Spark (GB10, arm64, CUDA 13) on 2026-10-27, with `sma.yaml` plus changes to keep runs short: quality limited to `ifeval` 10 and `gsm8k` 10 (no BFCL), a 60-second cool-down cap, and, for the vLLM and SGLang runs, `quality.max_tokens` 1024. Each run's engine log confirmed its 8-bit KV type, its context and deployment block were filled, its records validated against both schemas, and no hostname appeared in its folder. These five run folders are committed under `runs/`, without their logprob dumps and logits (see the run record contract). Earlier attempts from that day are not.

| Path | KV type, from the engine's own log | Context and pool | Peak memory | Results |
| --- | --- | --- | --- | --- |
| vllm `Qwen/Qwen3-0.6B` bf16 | `fp8`, 1,490,944 tokens (plan 1,490,944) | 40,960 per stream; 85.5 GB pool; 36 full-length streams | 95.24 GB | gsm8k 0.9 (n=10); ifeval 0.7 (n=10); PPL 29.42 over 98,256 tokens; determinism passed; decode 123 tok/s, 8 streams 890 tok/s |
| vllm `nvidia/Qwen3-8B-NVFP4` nvfp4 | `fp8`, 1,093,232 tokens (plan 1,093,247) | 40,960 per stream; 80.6 GB pool; 26 full-length streams | 97.75 GB | gsm8k 1.0 (n=10); ifeval 0.9 (n=10); PPL 12.24 over 98,256 tokens; determinism passed; decode 36 tok/s, 8 streams 276 tok/s |
| sglang `Qwen/Qwen3-0.6B` bf16 | `torch.float8_e4m3fn`, 1,490,944 tokens (plan 1,490,944) | 40,960 per stream; 85.5 GB pool; 36 full-length streams | 99.32 GB | gsm8k 0.8 (n=10); ifeval 0.6 (n=10); PPL 29.87 over 98,256 tokens; determinism passed; decode 122 tok/s, 8 streams 873 tok/s |
| llamacpp `unsloth/Qwen3-0.6B-GGUF` BF16 | `q8_0,q8_0`, 1,278,208 tokens (plan 1,278,029) | 40,960 per stream; 77.87 GB pool; 31 full-length streams | 95.91 GB | gsm8k 0.8 (n=10); ifeval 0.6 (n=10); PPL 25.86 over 8,184 tokens; determinism failed; decode 163 tok/s, 8 streams 292 tok/s |
| llamacpp `unsloth/Qwen3-0.6B-GGUF` UD-Q4_K_XL | `q8_0,q8_0`, 1,289,728 tokens (plan 1,289,647) | 40,960 per stream; 78.58 GB pool; 31 full-length streams | 94.17 GB | gsm8k 0.7 (n=10); ifeval 0.6 (n=10); PPL 26.60 over 8,184 tokens; KL 0.0627 vs the BF16 file, same top token 86.0%; determinism failed; decode 286 tok/s, 8 streams 375 tok/s |

The two llama.cpp rows are KL's reference and candidate: the quant's KL is against the BF16 file's logits. llama.cpp's perplexity scores 8 chunks (8,184 tokens) and is not comparable with the vLLM and SGLang perplexity over 98,256 tokens. The NVFP4 row ran the real FP4 kernel (`FlashInferB12xNvFp4LinearKernel`) with batch invariance off; its determinism probe still passed.

Found on the Spark and fixed in SMA:

- vLLM 0.21 rejects `--linear-backend` ("unrecognized arguments"), so every dense NVFP4 run failed to start. SMA now sets `VLLM_NVFP4_GEMM_BACKEND`.
- vLLM's batch-invariant quality mode served NVFP4 by emulation, at about 2 tokens per second per request for an 8B model, so no NVFP4 quality or KL number measured the real kernels. Quantized checkpoints now run with batch invariance off (owner decision).
- A suite whose requests timed out was scored as if they were wrong answers (GSM8K read 0.2 with 8 of 10 requests unanswered). A suite with unanswered requests now has a null score and says how many; a refusal still counts as an answer.
- NVIDIA's Qwen3-8B NVFP4 checkpoint declares FP8 KV scales that are all 1.0; the record said "checkpoint" scales. It now adds what the engine's log says it used.
- SGLang's deterministic mode crashed at CUDA-graph capture ("DeepGEMM ... Unsupported architecture"), with a 16-bit or an FP8 KV cache. SMA sets `SGLANG_ENABLE_JIT_DEEPGEMM=0`.
- SGLang's architecture registry query ran without a GPU, so SGLang's registry dropped Qwen3 and preflight refused it. The query now runs with `--gpus`.
- SMA's own logprob step built the whole per-token dump in memory, 7 GB beside the KV pool, and one run peaked at 107.4 GB. The dump is now streamed to disk one window at a time.
- llama.cpp's compute buffer is sized by its whole context (about 12% of the KV pool), and its prompt cache can take 8 GiB of host RAM. A run peaked at 110.6 GB until the plan counted both.
- The default Hugging Face cache path is under the home directory, which holds the hostname on a machine whose user name is its hostname; fetch refuses it, as designed. Set `hf.cache_dir` outside your home directory.

Not verified on hardware:

- `--moe-backend flashinfer_b12x` and the Marlin fallback on a real MoE checkpoint
- BFCL against a live server (the tool-call path through llama-server's `--jinja` was checked with one direct request)
- loading `FlintForCausalLM` or `dgpp_mixed` in any image
- dropping the page cache (`/proc/sys/vm/drop_caches` was not writable; the run records that and warns)
- the memory plan on a multimodal checkpoint or a model larger than 8B; the overhead constants come from small models
- a whole run through the KernelBench, SWE-bench, and BrowseComp-Plus steps, or the MuSiQue suite, with a real model. Checked live on 2026-10-27: one KernelBench job through SMA's client (submit, poll, results, baselines, scoring), and every runner operation the SWE-bench step uses, including the reverse forward (the runner read an HTTP server on the Spark through it). On 2026-10-28: SMA's judge client against Ornith (`judge-info.md` section 7); MuSiQue's pinned file, prompt, and scorer on the real dev file; and BrowseComp-Plus's runner set up from `services/browsecomp-plus/setup.sh`, with SMA starting the search server there, searching through its SSH forward, and stopping it
- that a benchmark's dataset is the same revision on two runs. nel 0.3.0 loads Hugging Face datasets without a pinned revision
- `--default-chat-template-kwargs` on SGLang (SMA passes it to vLLM only; nel suites still send it with every request, the BFCL container does not)

No benchmark numbers are shipped with this repo. The numbers above are wiring checks on small models, not arena results.

## Benchmark services

KernelBench, SWE-bench Verified-mini, and BrowseComp-Plus's search run on other machines: a second GPU machine compiles and times the kernels, an x86_64 Docker host runs the agent's tasks, and an x86_64 machine (the lab uses the same one) holds BrowseComp-Plus's index and answers its searches. [`services/`](services/README.md) has all three, with how to deploy your own copy and how to call it. The judge that grades BrowseComp-Plus is described in [`judge-info.md`](judge-info.md).

## Development

```bash
uv sync --extra dev
uv run pytest
```

`sma run Qwen/Qwen3-8B --bf16 --dry-run` does not need a GPU. It does not check that vLLM's registry contains `Qwen3ForCausalLM`; a real run does.
