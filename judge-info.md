# Ornith-1.5-122B deployment: reproduction spec

This is how the lab serves Ornith-1.5-122B (as of 2026-10-28), written so another system can rebuild it exactly. For example, a benchmark can use it as an LLM judge and get the same model, build and settings. The pod, image, arguments, served chat template and weight hash below were checked against the running deployment on 2026-10-28 between 05:06 and 05:20 UTC.

## 1. Weights

| | |
|---|---|
| Model | Ornith-1.5-122B (ornith-ai, MIT license), a fine-tune of Qwen3.5-122B-A10B |
| Repository | Hugging Face `ornith-ai/Ornith-1.5-122B-GGUF` |
| Revision | `5514dc9eb187cb71eb3b1089716fbe1bdd7e7a7d` (2026-10-20). Later commits up to `8d5893d` (2026-10-21) change only the README, so the file is identical on `main` |
| File | `Ornith-1.5-122B-Q4_K_M.gguf`, 75,843,911,648 bytes |
| SHA-256 | `729fe18c2491aaa446074a923e12f184f9f2239ffd5ce3dae47b0ce9cf8df40c`. This is Hugging Face's LFS hash, and the live file matched it on 2026-10-28 |
| Not used | the multimodal projector `mmproj-Ornith-1.5-122B-BF16.gguf`, so the deployment is text only. Also not used: the BF16, Q5_K_M, Q6_K and Q8_0 quants |

The file's SHA-256 covers the weights, the tokenizer, the embedded chat template and the embedded sampling defaults.

**Quant:** Ornith's own Q4_K_M (`general.file_type = 15`, `quantization_version = 2`). The lab did not re-quantize it.
- **Calibration:** Ornith's own imatrix, 612 entries and 4,321 chunks.
- **Q4_K:** 453 tensors, 53.92 GiB.
- **Q6_K:** 76 tensors, 16.55 GiB.
- **F32:** 370 tensors (norms, router, small state parameters), 0.15 GiB.
- **Routed experts:** `ffn_gate_exps` and `ffn_up_exps` are Q4_K, and `ffn_down_exps` is Q6_K (50.62 + 14.77 GiB). The output head is Q6_K.

**GGUF header:**
- **Identity:** GGUF v3, 899 tensors, 46 metadata keys. `general.architecture = qwen35moe`, `general.name = Qwen3.5-122B-A10B-ck2100-v1014-MTP1740`, `general.size_label = 256x6.7B`.
- **Layers:** `block_count = 49`: 48 model layers plus 1 MTP layer (`blk.48`, `nextn_predict_layers = 1`).
  - Every 4th layer is full attention (`full_attention_interval = 4`, so 12 layers): 32 query heads, 2 KV heads, head size 256.
  - The other 36 layers are linear attention (SSM: conv kernel 4, state 128, 16 groups, inner 8192).
- **Experts:** 256 experts with 8 used per token, expert FFN 1,024, plus a shared expert with FFN 1,024.
- **Size and context:** embedding 3,072, vocabulary 248,320, `context_length = 262,144`.
- **RoPE:** base 1e7, 64 dimensions, sections [11, 11, 10, 0].
- **Tokenizer:** `gpt2` / `qwen35`, EOS `<|im_end|>` (248046), no BOS added.
- **Embedded sampling:** temperature 0.6, top-p 0.95, top-k 20.

**Chat template:** the GGUF's own (`tokenizer.chat_template`), 7,594 bytes, SHA-256 `8b4d21a1e70ccbc8849e5e5e6dd32f3acfb2e2060739475909edc6e9b27c7165`.
- **What the server uses:** llama-server strips the trailing newline. The running server's `/props` `chat_template` is 7,593 characters, SHA-256 `9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235`. Compare against this hash to confirm a rebuilt server uses the same template.
- **Behavior:** Qwen3.5 format. Thinking is on by default, and tool calls use `<tool_call>` blocks.

## 2. Engine: llama.cpp, built for Volta

- **Source:** `ggml-org/llama.cpp` release **b11528**, commit `e60eff95fda7b6e46bf402a8e9f1e86d88ab29c8`. The build fails unless `HEAD` equals this commit.
- **One patch:** the one-line Volta FlashAttention config from llama.cpp PR #27997 (draft, head `327f0b14`). It goes immediately before the line containing `TODO tune specifically for Volta` in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`:
  ```
      GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 32, 128, 64, 128, 2, false);
  ```
  Nothing else is patched.
- **Base image:** `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66`. It uses CUDA 12.8, because CUDA 13 dropped Volta.
- **CMake:** `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined`. The image's binary reports GNU 13.3.0, Ubuntu 24.04's gcc.
- **What the running server reports:** `build_info = b11528-e60eff95f`, version "0.6.0-dev (build 11528, commit e60eff95f)".
- **The image here:** `llamacpp-sm70:e60eff95fda7-pr27997`, image ID `sha256:f810c1ee68a4700fa65434fb1b220549ca94bfe23d308f8d69de18dead7404a1`, 5,729,735,789 bytes. It was built 2026-10-17 00:17:11Z with buildah 1.39.3, and has no registry copy.

The Containerfile, unchanged since 2026-10-17. Build it with `--build-arg VOLTA_FA_PR27997=1`:

```dockerfile
FROM docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
ARG LLAMA_CPP_COMMIT=e60eff95fda7b6e46bf402a8e9f1e86d88ab29c8
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
 && git -C /opt/llama.cpp checkout --quiet "$LLAMA_CPP_COMMIT" \
 && test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "$LLAMA_CPP_COMMIT"
WORKDIR /opt/llama.cpp
ARG VOLTA_FA_PR27997=0
RUN if [ "$VOLTA_FA_PR27997" = 1 ]; then \
      f=ggml/src/ggml-cuda/fattn-mma-f16.cuh \
      && test "$(grep -c 'TODO tune specifically for Volta' $f)" = 1 \
      && sed -i '/TODO tune specifically for Volta/i\    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 32, 128, 64, 128, 2, false);\n' $f \
      && grep -n 'CASE(256, 256, 64, 128, 2, 32, 128, 64, 128, 2, false)' $f; \
    fi
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
 && cmake --build build -j 16 --target llama-server llama-perplexity llama-bench llama-cli \
 && install -m 0755 build/bin/llama-* /usr/local/bin/ \
 && find build/bin -maxdepth 1 -name '*.so*' -exec cp -a {} /usr/local/lib/ \; \
 && ldconfig && rm -rf build \
 && test -x /usr/local/bin/llama-server
RUN useradd --uid 10001 --create-home --home-dir /home/serve serve
USER 10001
ENTRYPOINT ["/bin/bash"]
```

**A rebuild is close but not byte-identical:**
- `apt-get` installs `build-essential`, `cmake` and `git` from the Ubuntu mirror on the build day, so their versions can differ from this build's.
- `GGML_NATIVE=ON` compiles the CPU code for the build machine: here an Intel Haswell-EP, so AVX2/FMA and no AVX-512. With every layer on the GPUs, the CPU code is mostly glue, but build on an AVX2 machine, or set `-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`, to stay close.

## 3. Server command

These are the exact arguments. Nothing is set in the environment.

```
llama-server -m /models/ornith-122b/Ornith-1.5-122B-Q4_K_M.gguf --alias ornith-1.5-122b \
  --host 0.0.0.0 --port 8000 \
  -ngl 999 -fa on -sm layer -ts 15,16,17 \
  -c 614400 -np 3 --no-kv-unified -ctk q8_0 -ctv q8_0 --ctx-checkpoints 16 \
  -b 4096 -ub 1024 -t 8 --seed 3407 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
  --jinja --metrics
```

With Docker it would be, for example:

```
docker run --gpus '"device=0,1,2"' -v /path/to/models:/models:ro -p 8000:8000 \
  --entrypoint llama-server llamacpp-sm70:e60eff95fda7-pr27997 <the arguments above>
```

Use three V100 32 GB cards in PCI bus order. The image's entrypoint is `/bin/bash`, so call `llama-server` explicitly.

**What the arguments do:**
- **Layers:** every layer is on the GPUs (`-ngl 999`). Layers are split across 3 GPUs in the proportions 15:16:17. That is a pipeline over PCIe; tensor parallel isn't possible, because 2 KV heads can't be divided three ways.
- **Sessions and KV cache:** 3 sessions of 204,800 tokens each (`-c 614400 -np 3 --no-kv-unified`). The KV cache is q8_0 and needs FlashAttention.
  - Only the 12 full-attention layers keep a KV cache, about 12.75 KiB per token, so about 2.5 GiB per full session.
  - Each session also has about 150 MiB of linear-attention state.
  - Up to 16 context checkpoints of that state are kept per session, in host RAM.
- **Prompt batches:** logical 4096, physical 1024. That is about 20% faster prompt reading than llama.cpp's default 2048/512 on this 256-expert model. A physical batch of 2048 ran the third card out of memory.
- **MTP speculative decoding** (`--spec-type draft-mtp`): **off**. llama-server skips the MTP layer's tensors ("model has unused tensor blk.48... -- ignoring"). MTP was measured on 2026-10-20 and gained nothing across concurrent streams.
- **Sampling defaults:** the model card's "precise coding" preset: temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence 0, repeat 1.0, seed 3407. Requests can override them. The model card's "general tasks" preset is temperature 1.0 with presence penalty 1.5, and that preset is **not** what's set here.
- **Reasoning:** the template keeps reasoning on by default. The log says "chat template supports preserving reasoning, it is enabled by default".

**Other values `/props` reports** (llama.cpp defaults, not set above):
- **Penalties and samplers:** `repeat_last_n` 64, `frequency_penalty` 0, `typical_p` 1, `dry_multiplier` 0, `xtc_probability` 0, `mirostat` 0, `top_n_sigma` -1, `n_predict` -1.
- **Sampler order:** penalties, dry, top_n_sigma, top_k, typ_p, top_p, min_p, xtc, temperature.
- **Endpoints:** `/slots` on, `/props` writes off, `/metrics` on.

## 4. Hardware

| | |
|---|---|
| Server | TYAN FT77C-B7079 8-GPU server (board S7079GM2NR-2T-N, BIOS V2.04.B10 of 2017-04-05) |
| CPUs | 2 × Intel Xeon E5-2660 v3 (Haswell-EP, 10 cores / 20 threads each, 2.6 GHz, 3.3 GHz turbo), 2 NUMA nodes |
| RAM | 8 × 16 GB DDR4-2133 RDIMM (Micron 36ASF2G72PZ-2G1A2): one DIMM per channel, 4 channels per socket, 64 GB per NUMA node |
| GPUs used | 3 × NVIDIA Tesla V100 32 GB PCIe, details below. A 4th V100 runs a different model, and four P100s on the other socket aren't used |
| Driver | NVIDIA 580.178.04 (driver CUDA 13.0, while the image's runtime is CUDA 12.8.1), container toolkit 1.20.1 |
| Host OS | Proxmox VE 9.2.21 (Debian 13), kernel 7.0.14-25-pve, `iommu=pt` (VT-d on, passthrough mode; GPU peer-to-peer faulted under translated IOMMU) |
| Orchestration | K3s v1.36.4+k3s1. The GPUs are claimed through NVIDIA's DRA driver, pinned by UUID |

**GPUs:** "Tesla PG500-216", PCI ID `10de:1df0` (GV100GL). These are V100 32 GB **PCIe** cards (not SXM2), an OEM/HP variant.
- **Parts:** board part 900-2G500-0350-030, GPU part 1DF0-887-A1, VBIOS 88.00.9C.00.01.
- **SMs:** 72 per card, where the stock V100 PCIe has 80.
- **Clocks:** application clocks at the default 1260 MHz, boosting to 1380 MHz max. Nothing locks them.
- **HBM2:** 1107 MHz, about 1.13 TB/s (the stock card runs about 0.9 TB/s).
- **Power:** 250 W limit (default and max).
- **Settings:** ECC on, persistence mode on, compute mode Default.
- **BAR1:** 32 GiB, mapping all of the memory.

| Model's GPU | PCI | VRAM in use (of 32,768 MiB) |
|---|---|---:|
| 0 | 04:00.0 | 29,878 MiB |
| 1 | 05:00.0 | 30,518 MiB |
| 2 | 08:00.0 | 28,968 MiB |

**PCIe:** every link on the path is trained at its maximum, **PCIe 3.0 (8 GT/s) x16**: the root ports, the switch's upstream and downstream ports, and the GPUs. That is about 15.75 GB/s per direction in theory.

```
CPU0 root port 00:02.0 (Gen3 x16) -> PLX PEX 8747 switch (02:00.0) -> 03:08.0 -> GPU0 04:00.0
                                                                   -> 03:10.0 -> GPU1 05:00.0
CPU0 root port 00:03.0 (Gen3 x16) -> PLX PEX 8747 switch (06:00.0) -> 07:08.0 -> GPU2 08:00.0
                                                                   -> 07:10.0 -> (4th V100, another model)
```

- **Topology:** all three cards hang off CPU0 (NUMA node 0, CPUs 0-9 and 20-29). In `nvidia-smi topo -m`, GPU0-GPU1 is PIX (same switch) and GPU1-GPU2 is PHB (through the CPU's root complex). No path crosses the socket interconnect.
- **NVLink:** none.
- **Peer-to-peer:** `nvidia-smi topo -p2p r` reports OK between the cards.
- **ACS:** on at the switch's downstream ports (ReqRedir+ CmpltRedir+). Peer traffic even within one switch is therefore sent up to the root port.
- **Payloads:** MaxPayload 256 B everywhere. MaxReadReq is 512 B on the GPUs and 128 B on the bridges.
- **What crosses PCIe (computed, not measured):** with a layer split, only the activations between pipeline stages cross, at the 2 stage boundaries. That is a 3,072-wide f32 vector, 12 KiB per token, so about 12 MiB per 1,024-token prompt batch. This is small against the link.
- **CPU threads:** the server runs 8 CPU threads with no CPU pinning (affinity 0-39). The pod requests 4 CPUs and 16 GiB, with a 40 GiB memory limit.

## 5. How it is served

**Request path:** clients call the server's OpenAI-compatible API (`/v1/chat/completions`) through the lab's LiteLLM proxy. LiteLLM is image `ghcr.io/berriai/litellm` v1.102.1, digest `sha256:87f34979b9f8cb274fac90ca8a4fdda07d8480de22755562a26adeb95ce20d02`.
- **Model in LiteLLM:** `openai/ornith-1.5-122b`, with `max_input_tokens` 204,800, timeout 1,800 s and no retries.
- **Global settings:** `drop_params: true`. Requests also pass through the lab's own hooks, which add a session header and coalesce streamed reasoning chunks.
- **For judge runs:** check that llama.cpp-specific fields such as `top_k`, `min_p` and `cache_prompt` actually reach the server, or send them in `extra_body`. Calling llama-server directly avoids the question.

**Measured speed:**
- **2026-10-20:**
  - one stream decodes at about 48–59 tok/s; three short streams total about 83 tok/s;
  - prompt reading runs at 456 tok/s at 29K tokens and 354 tok/s at 116K;
  - a fact planted at 116K tokens was found;
  - measured earlier that day with 4 sessions and llama.cpp's default batch: decode at about 33 tok/s at 116K tokens deep, and a cached multi-turn reply starting in about 0.9 s.
- **Real traffic, 2026-10-22 16:51Z to 2026-10-28 05:09Z** (877 requests, from the server's own timings): prompt reading had a median of 322 tok/s for 1K–10K new tokens and 426 tok/s for 10K–50K. Decode had a median of 37 tok/s per request (p10–p90 26–49), and that includes time sharing the GPUs with other streams.
- **Load time:** 13 min 19 s from a cold page cache (the weights sit on a RAID-Z2 HDD pool), and 20–33 s when the file is already cached.

## 6. Reproducing judge results

**What to keep identical** to get the same numbers:
- the weights file (its SHA-256);
- the llama.cpp commit and patch;
- `sm_70`, which means V100 cards: another GPU architecture runs different kernels;
- the KV type (q8_0), FlashAttention on, `-b 4096 -ub 1024`, and the sampling settings.

Layer split puts whole layers on each card, so the 15:16:17 split itself shouldn't change the arithmetic. That isn't tested here.

**llama.cpp is not batch-invariant.** A request's output can change when other requests share its batch (`-np 3` serves several at once) and when a cached prompt prefix is reused. For repeatable verdicts:
- send judge requests one at a time;
- use temperature 0, or a fixed `seed` per request;
- set `"cache_prompt": false`;
- fix `max_tokens`, the system prompt and the reasoning setting, and record every request field you send.

Leave the server settings as listed so the deployment stays the same, and set judge settings per request.

**Tested 2026-10-28 (section 7):** identical requests sent one after another returned byte-identical replies. **Not tested:** the same while other requests share the batch.

## 7. How SMA calls it

SMA's BrowseComp-Plus step grades its answers with this deployment (owner decision 2026-10-28). The settings are `browsecomp.judge_*` in `sma.yaml`. The endpoint and key are `SMA_JUDGE_URL` and `SMA_JUDGE_API_KEY` in the git-ignored root `.env`; a run record names the URL only as `${SMA_JUDGE_URL}` and never holds the key.

**The request:** `POST $SMA_JUDGE_URL/v1/chat/completions`, through the LiteLLM proxy, one request at a time, once per answer:

```json
{
  "model": "ornith-1.5-122b",
  "messages": [{"role": "user", "content": "<GRADER_TEMPLATE filled with the question, the model's answer and the correct answer>"}],
  "max_tokens": 4096,
  "temperature": 0.0,
  "chat_template_kwargs": {"enable_thinking": false},
  "cache_prompt": false
}
```

- **Prompt:** `GRADER_TEMPLATE` from BrowseComp-Plus `scripts_evaluation/evaluate_run.py` at commit `046949032b0328319cc9a02663a759ec601d9402`, word for word, with no system prompt. It is `GRADER_TEMPLATE` in `src/sma/browsecomp.py`, SHA-256 `71fb29f51a56cea9fe647331d525648474195fa4816f7d05632459d572419d90`, which `tests/test_browsecomp.py` checks.
- **Thinking off,** as BrowseComp-Plus runs its own judge (Qwen3-32B with `enable_thinking: false`). Its other settings differ: it samples at temperature 0.7, top-p 0.8, top-k 20, and SMA decodes greedily for repeatable verdicts.
- **Not sent:** top-p, top-k, min-p, a seed, `stream`. At temperature 0 llama.cpp picks the most likely token, so the server's sampling defaults do not apply.
- **`max_tokens` 4096** is BrowseComp-Plus's own grader limit.
- **Reading the verdict:** BrowseComp-Plus's `parse_judge_response` (`parse_judge` in `src/sma/browsecomp.py`): the `correct: yes` or `correct: no` line, plain or bold. A reply without that line counts the answer as wrong, as BrowseComp-Plus counts it, and the run warns with the count.
- **What a run keeps:** each reply's text, reasoning (none with thinking off), finish reason and usage, in the run's git-ignored bundle `quality/browsecomp_plus/trajectories.jsonl.gz`, whose SHA-256 the manifest records. `scores.json` names the judge and these settings.

**Checks on 2026-10-28,** on 99 answers from BrowseComp-Plus's published GPT-5 and o3 runs (BM25), sent through the proxy 4 at a time:

| Judge | Verdicts it gave | Wrong verdicts, checked against the answer key | Median time per answer |
|---|---|---:|---:|
| Ornith, thinking off (SMA's setting) | 98 of 99. One reply left out its `correct:` line | 0 of the 97 checked | 13 s |
| Ornith, thinking on | 98 of 99. One reply reasoned past 8,192 tokens and gave no answer | 0 of 98 | 18 s |
| LFM 2.5 24B (stopped after 32 answers) | 22 of 32. Ten replies left out the `correct:` line | 0 of 22 | 66 s |

- **How the verdicts were checked:** a "yes" whose extracted answer contains the key, or the reverse, after normalizing case and punctuation, was taken as right. Every other verdict was read by hand: all the "no" verdicts and every case where the string check and the judge disagreed.
- Thinking on and off gave the same verdict on all 97 answers that both judged. The 98th verdict with thinking off, which accepted a musician's full legal name for a stage name, was not checked.
- These answers come from strong models that follow the answer format. Small models' answers will be messier.
- **`cache_prompt` reaches the server through the proxy:** with `cache_prompt: false` a repeated request reused no prompt tokens (`timings.cache_n` 0). With caching left on, the repeat reused 662 of its 666 prompt tokens.
- **Repeatability:** three identical requests in a row gave byte-identical replies for each of 5 answers.
