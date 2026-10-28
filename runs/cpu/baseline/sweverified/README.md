# SWE-bench Verified-mini on the CPU runner: gold-patch baseline and how to call it

Written by hack-lab on 2026-10-27. Machine identifiers are `${NAME}` placeholders. Their values are in `runs/.env`; load them with `set -a; . runs/.env; set +a`.

SWE-bench's task containers are x86_64, and their arm64 support is experimental, so they don't run on the Spark. `${SWE_RUNNER_HOST}` is an x86_64 Docker host for them. SMA logs in as `${SWE_RUNNER_USER}` with `${SWE_RUNNER_SSH_KEY}`; the user is in the docker group, without sudo. The runner reaches the internet but nothing in the lab, and only `${SMA_HOST}` (and the owner) can reach it, on SSH only.

KernelBench's baseline times the reference code. SWE-bench's counterpart is the **gold-patch run**: every task is evaluated with its own reference patch (`--predictions_path gold`), so each one should resolve. That shows each task image, its tests and the harness work on this runner, and gives each task's evaluation time here.

## Files here

| File | What |
|---|---|
| `gold-baseline.report.json` | The harness's report for the run (`run_id` `gold-baseline`) |
| `instances.csv` | One row per task: instance id, resolved, FAIL_TO_PASS and PASS_TO_PASS tests passed, the harness's seconds for that task, the image digest |
| `settings.json` | The runner (CPU, memory, disk, Docker), the tool versions and the dataset and image pins |

## Results: gold patches, one worker

Run `gold-baseline`, started 2026-10-27T22:51:32Z, swebench 5.0.2, SMA's 50-task Verified-mini file.

| | |
|---|---|
| Resolved | **50 of 50** |
| Unresolved / errors / empty | 0 / 0 / 0 |
| Seconds per task | min 27, median 36, max 45; total 29 min |
| django | 25 tasks, median 39 s |
| sphinx-doc | 25 tasks, median 30 s |

Every task's image, its tests and the harness work on this runner: a model's patch that fails a task fails on its own merits, not on the environment. Times are the harness's for one task with nothing else running (start of the container to its removal, tests included). Runs with more workers were not measured.

## How to call it during an eval

The model server on `${SMA_HOST}` listens on 127.0.0.1 only. SMA reaches it from the runner through its own SSH connection, as a reverse forward to the runner's loopback. Then the agent runs on the runner, with its task containers there, and the harness grades the patches there. That's how SMA's first trial ran (`~/sma-swe`, 2026-10-27).

```sh
set -a; . runs/.env; set +a
R="${SWE_RUNNER_USER}@${SWE_RUNNER_HOST}"
ssh -i "$SWE_RUNNER_SSH_KEY" "$R" docker info --format '{{.ServerVersion}}'          # reachable

# 1. The model server's port on the runner's loopback, for as long as the eval runs
ssh -i "$SWE_RUNNER_SSH_KEY" -N -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:${SMA_MODEL_PORT}:127.0.0.1:${SMA_MODEL_PORT}" "$R" &

# 2. The agent, on the runner: mini-swe-agent over the 50 Verified-mini tasks
ssh -i "$SWE_RUNNER_SSH_KEY" "$R" 'cd ~/sma-swe && .venv/bin/mini-extra swebench \
    --subset data/verified-mini --split test --output runs/<run-id> --workers 4 \
    -c swebench.yaml -c model.model_name=hosted_vllm/<served-model-name> \
    -c model.model_kwargs.api_base=http://127.0.0.1:'"$SMA_MODEL_PORT"'/v1 \
    -c model.model_kwargs.temperature=0 -c agent.step_limit=250'

# 3. Grade the patches with the harness, on the runner
ssh -i "$SWE_RUNNER_SSH_KEY" "$R" 'cd ~/sma-swe/eval && ../.venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name ../data/verified-mini.jsonl --split test \
    --predictions_path ../runs/<run-id>/preds.json --run_id <run-id> --max_workers 4 --timeout 1800'

# 4. Bring the report back
scp -i "$SWE_RUNNER_SSH_KEY" "$R:sma-swe/eval/*.<run-id>.json" runs/...
```

- **Task images** are already on the runner, pinned by digest (`~/sma-swe/images.txt`, pulled by `~/sma-swe/pull-images.sh`) and tagged `:latest`, the name mini-swe-agent and the harness look up. Nothing is pulled during an eval. Docker Hub limits anonymous pulls, so pull new images one at a time with that script.
- **Concurrency:** the gold run used one worker, for clean per-task times. The runner has 10 vCPUs and 23 GiB of memory; SWE-bench suggests at most 75% of the cores as workers, so 4–6 at once.
- **Containers:** the harness names them `sweb.eval.<instance>.<run_id>` and removes them when done; mini-swe-agent's are `minisweagent-<id>` with `--rm`. `docker ps -a` on the runner shows anything left behind.
- **Disk:** about 200 GB free; the 50 images take about 35 GB. `docker image prune` removes dangling layers only.
- **Settings still open for the owner** (SMA's `docs/serving-design.md`): step limit (250 suggested), sampling and thinking mode. Agent runs are not part of this baseline.
