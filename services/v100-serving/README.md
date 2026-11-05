# V100 serving pod

The lab's V100 pod serves models for SMA; the harness (SMA itself, every benchmark client, grading, and
records) runs on the Spark. SMA starts llama.cpp there over SSH (`remote:` in the config), one run per GPU at
a time, four at once (`sma worker --slots`).

| | |
| --- | --- |
| Machine | A container reached over SSH (`V100_HOST` in the root `.env`): 4x V100 32 GB, driver 580, CUDA 12.8, 12 CPUs, 503 GB RAM, /work 3 TB |
| Topology | PCIe: GPUs 0+1 share one switch, 2+3 another; across the pairs traffic goes through the host bridge |
| Access | SSH from the Spark only, as `V100_HOST_USER` on `V100_HOST_SSH_PORT` with `V100_HOST_SSH_KEY`. Model ports (`V100_MODEL1_PORT`..`4`) are inside the pod; reach them with SSH local forwards |
| Engine | llama.cpp only. Volta has no BF16, FP8, or FP4, so vLLM and SGLang cannot serve here, and NVFP4 checkpoints cannot run |

## The pinned build

The pod has no Docker, and llama.cpp's published CUDA images are CUDA 13, which dropped Volta. So
`setup.sh` builds llama.cpp from source at release b11960 (the Spark's pinned release) for compute capability
7.0 with the pod's CUDA 12.8 toolkit:

```bash
scp -P "$V100_HOST_SSH_PORT" -i "$V100_HOST_SSH_KEY" services/v100-serving/setup.sh "$V100_HOST_USER@$V100_HOST:/work/sma/services/"
ssh -p "$V100_HOST_SSH_PORT" -i "$V100_HOST_SSH_KEY" "$V100_HOST_USER@$V100_HOST" 'bash /work/sma/services/setup.sh /work/sma'
```

It writes `/work/sma/llama.cpp-b11960/build/bin/llama-server` and `llama-perplexity`, and `build.json` with the
tag, the commit, the CUDA version, the build flags, and each binary's sha256. A run's record names that build
the way a Spark run names its image digest.

## How SMA serves here

With `remote.enabled`, a run's model server and its llama-perplexity run on the pod; everything else stays on
the Spark.

- **Model files.** SMA downloads the run's GGUF files on the pod from the run's pinned revision, into
  `/work/sma/models/<repo>@<revision>/`, once, and checks each against the sha256 of the Spark's copy.
- **The server.** `llama-server` from the pinned build, on the run's GPU (`CUDA_VISIBLE_DEVICES`), with the
  same flags as on a Spark, plus `--threads` and `--threads-batch` (`remote.threads_per_slot`, 3): the pod's
  cgroup allows 12 CPUs but shows 40, and llama.cpp would size its threads from the 40. Its folder,
  `/work/sma/servers/<name>/`, holds its log, its pid, and the GPU it runs on. An SSH local forward puts it on
  the Spark's `engine.port`, so every client reaches it as it would a local server.
- **Telemetry** is the pod's `nvidia-smi` for the run's GPU, over SSH. Each slot has its own SSH connection
  (sshd allows 10 sessions on one).
- **Records.** Runs sit under `runs/<slug>/llamacpp-v100/`, beside, never among, a Spark's. The manifest's
  `serving` names the hardware, the slot, the GPU, its model and driver, and the build's record; the engine
  image is the source build. KL compares only runs served on the same hardware.

## Slots: one run per GPU (`sma worker --slots`)

Owner decisions of 2026-11-01. A slot is one GPU with its own model port, forward port, and BrowseComp-Plus
search port:

```yaml
remote:
  enabled: true
  threads_per_slot: 3
  slots:
    - {name: v100-0, gpus: "0", remote_port_env: V100_MODEL1_PORT, local_port: 18001, search_port: 8791}
    - {name: v100-1, gpus: "1", remote_port_env: V100_MODEL2_PORT, local_port: 18002, search_port: 8792}
    - {name: v100-2, gpus: "2", remote_port_env: V100_MODEL3_PORT, local_port: 18003, search_port: 8793}
    - {name: v100-3, gpus: "3", remote_port_env: V100_MODEL4_PORT, local_port: 18004, search_port: 8794}
```

One dispatcher per serving machine, on the harness Spark: `sma worker --slots --config <yaml>`. It takes GGUF
jobs from the queue in the order they were queued. Each leg of a job is a unit; each unit runs on the first
free slot as its own process (`sma worker --run-unit JOB LEG --slot NAME`, which nobody runs by hand), and as
soon as one ends, the next unit starts on that slot.

- **A pair's legs start together.** A 4-bit leg is ready once its reference leg has a run id, which takes
  seconds; the free slots wait for it rather than go to a later job. The 4-bit leg's logprobs step then waits
  for the reference's logits (`logprobs.reference_wait_s`, 6 h). If the reference's logprobs step fails, or the
  wait runs out, KL is null and the warning says why. `sma watch` shows "waiting for the reference's logits".
- **Runs across GPUs** (owner decision 2026-11-05). A leg whose memory plan does not keep the start context on
  one V100 takes several slots, its count (`gpus`) chosen when it is queued (`runplan.choose_gpus`). The unit's
  slot name joins them (`v100-0+v100-1`): it serves on all their GPUs, with the first slot's ports and lock;
  llama-server and llama-perplexity split it by layer. The dispatcher starts it once that many slots are free,
  preferring a PCIe pair for two GPUs, and holds the slots that free up for it meanwhile. A server across GPUs
  that fails to start marks no slot down.
- **Retries.** A leg that fails, ends with failed steps, or whose process dies goes back to the queue and
  resumes its own run on whatever slot is free, up to `queue.max_attempts`, after `queue.retry_backoff_s`.
  The run's `serving` then names the new slot, and the earlier ones stay in `serving.earlier`.
- **Dropping a job it runs.** `sma queue drop JOB --config <yaml>` marks each leg of the job that is not done
  failed for good, so nothing starts it again, and sends its units SIGTERM: each tears down its server, as
  on a dispatcher stop, and its run stays resumable by hand. The other slots keep running. Once the
  dispatcher has closed the job, it moves to the queue's `removed/` folder. Until then its file stays where
  the running dispatcher reads it.
- **A slot whose server fails to start twice in a row** is marked down and skipped until the dispatcher
  restarts.
- **Restart.** The dispatcher's claim id is `dispatcher-<host_alias>`, so a restarted dispatcher knows its jobs.
  It adopts units still running, puts back legs whose process is gone (resumable), and stops servers on the pod
  that no live unit owns. SIGTERM or Ctrl-C stops every unit; each tears down its own server, and every run
  stays resumable.
- **Memory.** The dispatcher guards the harness's memory for every slot: below `memory.guard_free_gb`, read
  from the harness's slice (`memory.guard_cgroup`, its memory.max less what it uses) or else the machine's
  MemAvailable, it stops only the run using the most (its process tree plus its containers), kills that run's
  containers at once, and stops its server on the pod. The leg goes back to the queue and resumes after
  `queue.retry_backoff_s`; the run's record carries the note. A second stop fails the leg for good and writes
  it to the queue's `failed/` folder as `<job>-leg<N>.json` with both notes; the job's other legs go on. A
  run on a slot does not guard memory itself: each run alone would stop all of them.
- **Shared work, one at a time.** The speed steps (`serve_speed` and `speed`) of the pod's slots take turns in
  arrival order (`<lock_dir>/speed-v100.lock`); the other slots keep running their own steps, and the speed
  summary records it: `speed_turn` (how long the run waited) and `concurrent_load` (each other run's slot, the
  steps it ran in the speed window, its GPU's mean power, and its tokens per second; a run counts when its telemetry
  or engine counters have samples in the window). LiveCodeBench's grading
  takes a machine-wide lock inside its container, so one container grades at a time, and the speed steps hold
  that lock too while they measure: both are timed on the same CPUs, and on the GB10's slower cores 2 of 105
  problems failed the 6 s limit that pass on its faster ones. KernelBench's service already times one job at a
  time, oldest first.
- **CPUs and the harness slice on the Spark** (`harness:`): SMA's containers join `harness.cgroup_parent`;
  eval-factory suites run on `harness.cpus_untimed`; LiveCodeBench's grading and the speed client, which are
  timed, run on `harness.cpus_timed`. Units run on the untimed CPUs. `command.json` and the speed summary name
  the CPUs. With models served here, no SMA container on the Spark may use its GPU.
- **`sma watch`** shows a SLOTS section (one line per slot: its model, what it is on and how far, its decode rate
  over 5 minutes; under it, the selected slot's run whole, one line per benchmark), each queued leg with its slot,
  and KernelBench's queue. Plain `sma watch` shows these slots beside this Spark's own runs; `--config sma-v100.yaml`
  shows only the slots and the runs served on the V100s.
- **The systemd unit.** On the harness GB10 the dispatcher runs as `sma-dispatcher.service` in
  `sma-harness.slice` ([`services/kernelbench/host-setup.sh`](../kernelbench/README.md)).

Runs across two GPUs (a pair on one PCIe switch) are not built: every model so far fits one V100 (owner
decision 2026-11-01).
