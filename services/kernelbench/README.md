# KernelBench evaluation service

SMA's models write CUDA kernels for [KernelBench](https://github.com/ScalingIntelligence/KernelBench)'s problems. This service takes a batch of those kernels over HTTP. It compiles each one, checks it against KernelBench's PyTorch reference, and times both on its own GPU. It returns one result per kernel, with the environment the numbers came from.

It runs on its own machine. Compiling and timing kernels next to a model server would disturb SMA's speed numbers, and model-written code should not run where models are served. The lab runs it on a second DGX Spark. The baselines in [`runs/gb10/baseline/kernelbench/`](../../runs/gb10/baseline/kernelbench/README.md) were measured with this code (only its comments and the state folder line in the Containerfile differ from that service's).

## Files

| File | What |
| --- | --- |
| `Containerfile` | The image: CUDA, PyTorch 2.9.1, KernelBench at commit `423217d9`, and the three files below |
| `server.py` | The HTTP API, the job queue and the state on disk. Standard library only |
| `worker.py` | One evaluation in its own process: a kernel, a baseline problem, or the environment probe |
| `sandbox.py` | Landlock for each worker process |
| `stubs/` | Stand-ins for the model clients KernelBench imports but never uses here |
| `kernelbench-gpu-clocks.service` | A systemd unit that locks the GPU clock |

## What you need

- A Linux machine with an NVIDIA GPU, its driver, Docker, and the NVIDIA Container Toolkit. DGX OS on a DGX Spark has all of them.
- Landlock in the kernel. With Landlock ABI 4 (Linux 6.7) a kernel cannot open TCP connections; with ABI 6 (Linux 6.12) it cannot signal processes outside its sandbox. A DGX Spark's 6.17 kernel has ABI 7. Without Landlock the worker refuses to run.
- Memory. KernelBench sizes its problems for an 80 GB H100. On a GB10's 121 GB of unified memory, 249 of Levels 1–3's 250 problems run. A 12 GB RTX 4070 SUPER ran 62 of Level 1's 100; the rest ran out of GPU memory.
- A private way in. The API has no authentication. Keep it on loopback or a private network, and reach it over SSH or a VPN such as Tailscale.

## Build

On the GPU machine:

```bash
git clone https://github.com/iammrduncan/sma.git
cd sma/services/kernelbench
docker build -t kernelbench-eval:gb10 -f Containerfile .
```

The defaults are for a GB10: CUDA 13.0 and PyTorch's `cu130` wheels, arm64. nvcc 12.8 cannot build for the GB10's compute capability 12.1. PyTorch's own kernels go up to 12.0, which runs on the GB10, so PyTorch's warning about 12.1 at startup is harmless.

### Other GPUs

For an x86_64 card that CUDA 12.8 supports, build with the CUDA 12.8 base and wheels. The lab ran this build on an RTX 4070 SUPER:

```bash
docker build -t kernelbench-eval:cu128 -f Containerfile \
  --build-arg BASE_IMAGE=docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 \
  --build-arg TORCH_CUDA=cu128 .
```

Then set `TORCH_CUDA_ARCH_LIST` to the card's compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`), for example `8.9` for an RTX 40-series card.

## Lock the GPU clock

A GPU boosts and throttles with temperature and power, so the same kernel times differently from run to run. A fixed clock keeps times comparable. On a GB10 locked at 2200 MHz, the clock held 2184–2190 MHz for 120 s of fp32 matrix multiplies, at 60–65 W and up to 72 °C. Unlocked, it fell from 2463 to 2385 MHz as it reached 79 °C at its power cap. The lock made Level 1's times 2.6% longer (geometric mean): the compute-bound problems lose the boost, and the memory-bound ones barely move.

The service does not set the clock. The host does, with `kernelbench-gpu-clocks.service`. Set `GPU` and `CLOCK_MHZ` in it, then:

```bash
sudo install -m 0644 kernelbench-gpu-clocks.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kernelbench-gpu-clocks.service
nvidia-smi -q -d CLOCK          # "Applications Clocks" and the SM clock under load
```

Pick a clock the card can hold under load. A GB10 lists no supported clocks (`nvidia-smi -q -d SUPPORTED_CLOCKS` says N/A); any clock up to `clocks.max.sm` (3003 MHz) is accepted, and 2200 MHz holds. Other cards list theirs; the RTX 4070 SUPER used 1980 MHz. Pass the same number to the service as `KERNELBENCH_GPU_CLOCK_LOCK_MHZ`. It goes into the environment fingerprint, so results with and without a lock are never compared. Every job's environment also shows the clocks sampled under load, so a lock that did not hold shows up there.

## Run

```bash
# A network with no way out: kernels reach neither the internet nor your LAN.
docker network create --internal --subnet 172.30.250.0/29 kernelbench
docker volume create kernelbench-state
docker run -d --name kernelbench --restart unless-stopped \
  --gpus all --network kernelbench --ip 172.30.250.2 \
  --read-only --tmpfs /tmp:rw,exec,nosuid,size=1g \
  -v kernelbench-state:/var/lib/kernelbench \
  --cap-drop ALL --security-opt no-new-privileges \
  --memory 96g --cpus 18 \
  -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=8 \
  -e KERNELBENCH_HOST_MEMORY_FLOOR_GIB=8 -e KERNELBENCH_GPU_CLOCK_LOCK_MHZ=2200 \
  kernelbench-eval:gb10
curl -s http://172.30.250.2:8000/v1/health
curl -s http://172.30.250.2:8000/v1/environment     # GPU, CUDA, PyTorch, clocks, sandbox
```

Pick any free subnet. The container gets no published port. From the machine that runs SMA, forward a local port to it over SSH, and point the client at it:

```bash
ssh -N -L 127.0.0.1:8000:172.30.250.2:8000 <user>@<gpu-machine> &
export KERNELBENCH_URL=http://127.0.0.1:8000
```

What the flags are for:

- `--network kernelbench`: an internal network has no route out. Landlock already stops TCP in each kernel's process; this also stops UDP and everything else.
- `--read-only` and the `/tmp` tmpfs: the image cannot change. Each kernel builds in its own folder under `/tmp`, which needs `exec` because the built extension is loaded from there. The folder is emptied after every kernel.
- `kernelbench-state`: jobs, results and baselines. They survive a restart.
- `--memory 96g`: the server stops a worker 2 GiB below this, so one large problem cannot take the whole container down. On a 128 GB Spark this leaves the host about 25 GB.
- `--cpus 18`: nvcc builds each kernel with `MAX_JOBS` parallel compiles.

The image runs as uid 10001 with no capabilities.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TORCH_CUDA_ARCH_LIST` | unset | The compute capability kernels are built for. One value builds faster. |
| `MAX_JOBS` | unset | ninja's parallel compiles per kernel |
| `KERNELBENCH_HOST_MEMORY_FLOOR_GIB` | `0` (off) | Stop a worker when the machine's available memory falls below this. Set it on unified memory (GB10), where the GPU's allocations are not charged to the container |
| `KERNELBENCH_GPU_CLOCK_LOCK_MHZ` | unset | The host's clock lock, reported in every environment and part of the fingerprint |
| `KERNELBENCH_PORT` | `8000` | The port the server listens on |
| `KERNELBENCH_KEEP_DAYS` | `30` | How long finished jobs are kept |

## Record a baseline

A baseline is the eager PyTorch time of every problem, KernelBench's `generate_baseline_time.py` method. Record one for each new environment, before you compare anything:

```bash
curl -s -X POST $KERNELBENCH_URL/v1/baselines -H 'Content-Type: application/json' -d '{"levels": [1, 2, 3]}'
```

On the locked GB10, Level 1 took 19 minutes, Level 2 19 minutes and Level 3 6 minutes. Only Level 1's problem 95 (CrossEntropyLoss) fails, inside KernelBench on any GPU: its precision cast turns the class-index targets into floats, and `nll_loss` refuses them. Results: [`runs/gb10/baseline/kernelbench/`](../../runs/gb10/baseline/kernelbench/README.md).

## Use it

Every request and response is JSON. Jobs are asynchronous: submit, then poll.

| Call | What it does |
| --- | --- |
| `POST /v1/jobs` | Queue a batch, `{level, kernels: [{problem_id, kernel_src, label?}], options?, name?}`. Answers `202 {job_id, status, position, total, poll}`. One kernel is a valid batch. |
| `GET /v1/jobs/<id>` | Status (`queued`, `running`, `done`, `failed`, `cancelled`), `progress`, `environment`, `options` and the results so far. Add `?sources=1` for the submitted sources. |
| `GET /v1/jobs` | The latest 100 jobs, without results |
| `DELETE /v1/jobs/<id>` | Cancel. A queued job goes at once; a running job stops after its current kernel. |
| `POST /v1/baselines` | `{levels: [1], options?}`: record a baseline for the current environment |
| `GET /v1/baselines`, `GET /v1/baselines/<file>` | The recorded baselines; a file for the current environment, or `?fingerprint=` for another |
| `GET /v1/problems?level=1` | Problem ids and names |
| `GET /v1/environment`, `GET /v1/health` | The last probed environment; liveness and queue length |

A kernel is the model's whole answer: a Python module that defines `class ModelNew(nn.Module)` with the reference model's `__init__` and `forward` signature. Custom CUDA usually comes in through `torch.utils.cpp_extension.load_inline`. Levels 1–3 are offered; Level 4 needs Hugging Face models.

```bash
curl -s -X POST $KERNELBENCH_URL/v1/jobs -H 'Content-Type: application/json' \
  -d '{"level": 1, "name": "run-42", "kernels": [{"problem_id": 19, "label": "model/quant/sample-0", "kernel_src": "..."}]}'
curl -s $KERNELBENCH_URL/v1/jobs/<job_id>        # poll until status is done
```

[`kb_client.py`](../../runs/gb10/baseline/kernelbench/kb_client.py) does the same from a JSONL file of kernels, with the standard library only.

**Options** are KernelBench's `eval_from_generations.py` defaults, except the timeout: `num_correct_trials` 5, `num_perf_trials` 100, `timing_method` `cuda_event` (each trial with a cold L2 cache), `precision` `fp32` (or `fp16`, `bf16`), `backend` `cuda`, `timeout_s` 300 (60–1800) per kernel, `seed` 42, `excessive_speedup_threshold` 10. A baseline takes `num_perf_trials`, `timing_method` and `precision`. Limits: 1,000 kernels per job, 512 KiB per kernel, 50 queued jobs (429 beyond that). One kernel is evaluated at a time, so nothing else uses the GPU while a kernel is timed. A kernel takes about 20–40 s on the GB10, its build included.

**One result per kernel**, in submission order:

| Field | Meaning |
| --- | --- |
| `compiled`, `correct` | KernelBench's verdicts (correctness over 5 trials with random inputs) |
| `runtime_ms`, `ref_runtime_ms`, `speedup` | The kernel and the PyTorch reference, timed in the same process, one right after the other; `speedup` is ref / kernel. KernelBench rounds times to 3 significant figures |
| `baseline_ref_runtime_ms` | The stored baseline's time for the problem, or null |
| `runtime_stats`, `ref_runtime_stats` | mean, std, min, max, trials |
| `excessive_speedup` | KernelBench's flag for a speedup above the threshold, usually a sign of reward hacking |
| `error_kind` | null when the kernel compiled, was correct and was timed. Otherwise `compile` (nvcc's output is in `error`), `runtime` (an exception, a CUDA error, out of memory, or a memory limit below), `incorrect`, `timing`, `timeout`, or `crash` (the worker died, or the service died twice on this kernel and skipped it) |
| `kernel_sha256`, `label`, `eval_seconds`, `attempts` | Bookkeeping |

**The environment** is probed at the start of every job and returned with it: the GPU, its driver and compute capability, PyTorch and its CUDA, nvcc, Python, the KernelBench commit, the clock lock, the clocks sampled under load, and the sandbox. Its `fingerprint` hashes what a speedup depends on: GPU, driver, CUDA, nvcc, PyTorch, KernelBench commit and clock lock. Compare speedups only within one fingerprint. Baselines are stored by fingerprint.

## How kernels are contained

A model-written kernel is untrusted code.

- **The container:** no network out, a read-only image, uid 10001, no capabilities, `no-new-privileges`.
- **Each kernel** runs in its own new process, which is killed with everything it started when it ends or times out. Landlock confines that process: it can write only its own scratch folder, device files and its own `/proc` entry (CUDA needs it), it cannot open a TCP connection or listen, and it cannot signal or reach processes outside its sandbox, such as the server or earlier kernels.
- **Memory:** a container's memory limit can kill the server too, so the server stops a worker 2 GiB below it and records `runtime`. KernelBench builds a problem's inputs in host memory before it copies them to the GPU; one Level 1 input is 17 GB. If the service is killed anyway, the job resumes after the restart, and a kernel the service died on twice is recorded as `crash` and skipped.
- **Unified memory:** on a GB10 the GPU's allocations come from the machine's RAM and are not charged to the container. So the server also stops a worker when the machine's available memory falls below `KERNELBENCH_HOST_MEMORY_FLOOR_GIB`.

A kernel can still read the image and its own inputs. It can also lie about its own time, because it runs in the process that times it, as in KernelBench itself. `excessive_speedup` is the check for that. One kernel's results cannot change another's.

## Kubernetes

The lab runs this image as a Kubernetes Deployment on its GB10 machine: one replica with the `Recreate` strategy, the GPU through a resource claim, requests of 8 CPUs and 16 GiB with limits of 18 CPUs and 96 GiB, the variables above, a 20 GiB volume at `/var/lib/kernelbench`, and a 1 GiB `emptyDir` at `/tmp`. The pod runs with `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`, all capabilities dropped and no service account token, under Pod Security `restricted`. A NetworkPolicy allows no egress, and the only way in is a private proxy on the node.

## Operate

- **Logs:** `docker logs kernelbench`. One line per request; a failed job keeps its error.
- **State:** jobs, results and baselines are in the `kernelbench-state` volume. Finished jobs are kept for 30 days. A job interrupted by a restart resumes where it stopped.
- **Update:** pull, rebuild the image, and recreate the container. A new KernelBench commit, PyTorch, driver or clock lock changes the fingerprint: record a new baseline before you compare.
