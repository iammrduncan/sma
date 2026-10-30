# KernelBench on the GB10: baseline and how to call it

Written by hack-lab on 2026-10-27; baselines re-recorded at 2000 MHz on 2026-10-29, when every Spark's GPU clock was locked there. Machine identifiers are `${NAME}` placeholders. Their values are in the repository root's `.env` (git-ignored); from the root, load them with `set -a; . ./.env; set +a`.

SMA generates kernels on `${SMA_HOST}`. A separate DGX Spark, `${KERNELBENCH_HOST}`, compiles each one, checks it against KernelBench's PyTorch reference and times both on its GB10, so evaluation never competes with SMA's own model serving. The service is reached at `${KERNELBENCH_URL}`, from `${SMA_HOST}` only, and needs no key.

## Files here

| File | What |
|---|---|
| `level1-fp32-cuda_event-100.json`, `level2-…`, `level3-…` | The service's stored baselines: eager PyTorch time for every problem (KernelBench's `generate_baseline_time.py` method), with the environment they were measured in |
| `summary.csv` | One row per problem: level, id, name, mean/std/min/max ms, peak GPU memory, error |
| `environment.json` | What the service probed for these runs: GPU, driver, CUDA, PyTorch, KernelBench commit, clocks under load, sandbox |
| `settings.json` | The GB10's settings and locks, the container's limits and the image pins |
| `level1-fp32-cuda_event-100.2200mhz.json`, `level2-…`, `level3-…` | The same at 2200 MHz (2026-10-27), for reference only (another fingerprint) |
| `level1-fp32-cuda_event-100.unlocked.json` | Level 1 before any clock lock, for reference only (another fingerprint) |
| `kb_client.py` | A standard-library client: submit a batch, poll, save the results |

## Results (fp32, `cuda_event`, 100 trials; GPU clock locked at 2000 MHz)

Fingerprint `7816e61e5539da95` (2026-10-29). All three levels ran in one job on the same environment. The 2200 MHz baselines of 2026-10-27 (fingerprint `5975f7a6360573f7`) are kept as `*.2200mhz.json`.

| Level | Timed | Failed | Mean time: min / median / max | Peak GPU memory | Run |
|---|---|---|---|---|---|
| 1 | **99 of 100** | 1 | 0.615 ms (12) / 45.2 ms / 948.0 ms (68) | 24.0 GiB (39) | `kb-20261029-185237-5f74f847` |
| 2 | **100 of 100** | 0 | 4.86 ms (6) / 37.9 ms / 236.0 ms (78) | 10.1 GiB (100) | `kb-20261029-185237-5f74f847` |
| 3 | **50 of 50** | 0 | 0.671 ms (32) / 27.3 ms / 267.0 ms (31) | 16.1 GiB (31) | `kb-20261029-185237-5f74f847` |

**Problems with no time** (they fail for any kernel here, so leave them out):

- Level 1, 95 (95_CrossEntropyLoss.py): `"nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'`

**Against 2200 MHz** (249 problems timed at both): times are 2.4% longer (geometric mean; median 1.3%), from 0.953 (level 3 problem 40) to 1.11 (level 2 problem 14). Over the whole run the GPU averaged 52.9 °C (max 63.0) and 21.4 W (max 51.3), against 55.0 °C (max 71.0) and 28.4 W (max 98.1) at 2200 MHz, on another day (room temperature not measured).

## How to call it during an eval

1. **Generate kernels** with KernelBench's prompt for each problem. The service runs the problems from KernelBench at commit `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` (`GET /v1/problems?level=N` lists ids and names). A kernel is the model's whole answer: a Python module that defines `class ModelNew(nn.Module)` with the reference model's `__init__` and `forward` signature. Custom CUDA usually comes through `torch.utils.cpp_extension.load_inline`. Builds target compute capability 12.1 (`TORCH_CUDA_ARCH_LIST=12.1`).
2. **Send a batch**, one job per model and level:

   ```sh
   set -a; . ./.env; set +a
   curl -s -X POST "$KERNELBENCH_URL/v1/jobs" -H 'Content-Type: application/json' -d @job.json
   # job.json: {"level": 1, "name": "<run-id>", "kernels": [{"problem_id": 19, "label": "<model>/<quant>/sample-0", "kernel_src": "..."}], "options": {}}
   # -> 202 {"job_id": "kb-...", "status": "queued", "position": 0, "total": N, "poll": "/v1/jobs/kb-..."}
   ```

   Or `python3 kb_client.py submit --level 1 --kernels kernels.jsonl --out results.json --name <run-id>`, which polls until the job ends.
3. **Poll** `GET /v1/jobs/<job_id>` until `status` is `done` (or `failed`, `cancelled`). `results` fills in as kernels finish, in submission order. `DELETE /v1/jobs/<job_id>` cancels.
4. **Store with each SMA result:** `environment.fingerprint`, the job id and the per-kernel fields below. Compare speedups only within one fingerprint. A different fingerprint means a different GPU, driver, CUDA, PyTorch, KernelBench commit or clock lock, so record a new baseline first (`POST /v1/baselines {"levels": [1, 2, 3]}`).

**Options** (KernelBench's `eval_from_generations.py` defaults except the timeout): `num_correct_trials` 5, `num_perf_trials` 100, `timing_method` `cuda_event` (cold L2 cache each trial), `precision` `fp32` (or `fp16`, `bf16`), `backend` `cuda`, `timeout_s` 300 (60–1800) per kernel, `seed` 42, `excessive_speedup_threshold` 10. These baselines used the defaults; a baseline for another precision is `POST /v1/baselines {"levels": [1], "options": {"precision": "bf16"}}`.

**Limits:** 1,000 kernels per job, 512 KiB per kernel, 50 queued jobs (429 beyond that). One kernel is evaluated at a time, so nothing else runs on the GPU while a kernel is timed. A kernel takes about 20–40 s with its build; plan on 30–60 min for 100 kernels.

**One result per kernel:**

| Field | Meaning |
|---|---|
| `compiled`, `correct` | KernelBench's verdicts (5 correctness trials on random inputs) |
| `runtime_ms`, `ref_runtime_ms`, `speedup` | The kernel and the PyTorch reference, timed in the same process, right after each other; `speedup` = ref / kernel. KernelBench rounds to 3 significant figures |
| `baseline_ref_runtime_ms` | The stored baseline's time for that problem (these files), or null |
| `runtime_stats`, `ref_runtime_stats` | mean, std, min, max, trials |
| `excessive_speedup` | KernelBench's reward-hacking flag (speedup above the threshold) |
| `error_kind` | null if compiled, correct and timed; else `compile` (nvcc's output in `error`), `runtime` (exception, CUDA error, out of memory, or the host-memory limits below), `incorrect`, `timing`, `timeout`, or `crash` (the worker died, or the service died twice on this kernel) |
| `kernel_sha256`, `label`, `eval_seconds`, `attempts` | Bookkeeping |

**Sandbox:** each kernel runs in its own fresh process under Landlock: it writes only its scratch folder, opens no TCP connection and can't signal the server. The container has no network at all. A kernel can still lie about its own timing (it runs in the process that times it, as in KernelBench), which is what `excessive_speedup` checks.

**Other calls:** `GET /v1/environment` (the last probe), `GET /v1/health`, `GET /v1/jobs` (latest 100), `GET /v1/baselines` (all recorded), `GET /v1/baselines/<file>?fingerprint=<fp>` (another environment's).
