# SWE-bench runner

SMA scores coding agents on [SWE-bench Verified](https://www.swebench.com/)'s 50-task mini set ([MariusHobbhahn/swe-bench-verified-mini](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)). Each task runs in its own Docker image, and those images are x86_64: SWE-bench's arm64 support is experimental, so the tasks do not run on a DGX Spark. The runner is a separate x86_64 Docker host. The agent ([mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)) and SWE-bench's harness run there, and the model stays on the Spark.

This folder sets up a runner the same way as the lab's, where the gold-patch baseline in [`runs/cpu/baseline/sweverified/`](../../runs/cpu/baseline/sweverified/README.md) ran.

## Files

| File | What |
| --- | --- |
| `setup.sh` | Sets up the runner: the venv, the task file and the task images. Safe to run again |
| `requirements.in`, `requirements.txt` | swebench 5.0.2, mini-swe-agent 2.4.6 and LiteLLM 1.106.2; `requirements.txt` pins every package with its hashes, for Python 3.13 on x86_64 Linux |
| `build_dataset.py` | Builds the task file from both datasets at pinned revisions, and checks it against the baseline's |
| `images.txt` | The 50 task images, by digest |
| `pull-images.sh` | Pulls them one at a time and tags each as `:latest`, the name mini-swe-agent and the harness look up |

## What you need

- An x86_64 Linux machine or VM with Docker Engine. The lab's has 10 vCPUs, 24 GB of memory and a 250 GB disk.
- About 35 GB of disk for the 50 images, plus room to work. `pull-images.sh` stops while Docker's disk has less than 30 GB free.
- A user in the docker group to run the tasks. That group is root on the machine, so use a machine or VM that holds nothing else.
- [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs Python 3.13 if the machine does not have it.
- Internet access, for PyPI, Hugging Face and Docker Hub. The runner needs nothing on your own network. The lab's reaches the internet and nothing on the lab's network, and only the Spark reaches it, over SSH.

## Set up

As the runner's user:

```bash
git clone https://github.com/iammrduncan/sma.git
sma/services/swe-runner/setup.sh          # or SWE_RUNNER_DIR=/srv/sma-swe sma/services/swe-runner/setup.sh
```

It refuses a machine that is not x86_64, or has no Docker or uv. Then, in `~/sma-swe`:

1. It makes `.venv` with Python 3.13 and syncs it to `requirements.txt`, checking every hash.
2. It downloads both datasets at their pinned revisions, checks their SHA-256, and writes `data/verified-mini.jsonl` for the harness and `data/verified-mini/test.jsonl` for mini-swe-agent. The rows come from SWE-bench_Verified, which names each task's image, eval script and log parser. The script stops unless the file is byte for byte the one the baseline ran on, and `images.txt` lists exactly its images.
3. It pulls the 50 images by digest, waiting 30 s between pulls. Docker Hub limits pulls without a login; if pulls fail with `toomanyrequests`, run `docker login` and run `setup.sh` again.

## Check it: the gold-patch baseline

The harness can grade every task with its own reference patch. Each one should resolve, which shows that the images, the tests and the harness work on this machine:

```bash
cd ~/sma-swe/eval
../.venv/bin/python -m swebench.harness.run_evaluation --dataset_name ../data/verified-mini.jsonl --split test \
  --predictions_path gold --run_id gold-baseline --max_workers 1 --timeout 1800
```

It writes `gold.gold-baseline.json`. On the lab's runner, 50 of 50 resolved, at 27–45 s per task (median 36 s) and 29 minutes in all, one task at a time.

## Run an eval from the Spark

The Spark's model server listens on 127.0.0.1 only. SMA reaches it from the runner through its own SSH connection, as a reverse forward to the runner's loopback. The agent then runs on the runner and calls the model through the forward, the task containers run there, and the harness grades the patches there.

```bash
R=<user>@<runner>; PORT=8000                     # the runner's login; the model server's port on the Spark
ssh "$R" docker info --format '{{.ServerVersion}}'

# 1. The model server's port on the runner's loopback, for as long as the eval runs.
ssh -N -o ExitOnForwardFailure=yes -R "127.0.0.1:$PORT:127.0.0.1:$PORT" "$R" &

# 2. The agent, on the runner: mini-swe-agent over the 50 tasks.
ssh "$R" "cd ~/sma-swe && .venv/bin/mini-extra swebench \
    --subset data/verified-mini --split test --output runs/<run-id> --workers 4 \
    -c swebench.yaml -c model.model_name=hosted_vllm/<served-model-name> \
    -c model.model_kwargs.api_base=http://127.0.0.1:$PORT/v1 \
    -c model.model_kwargs.temperature=0 -c agent.step_limit=250"

# 3. Grade the patches with the harness, on the runner.
ssh "$R" "cd ~/sma-swe/eval && ../.venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name ../data/verified-mini.jsonl --split test \
    --predictions_path ../runs/<run-id>/preds.json --run_id <run-id> --max_workers 4 --timeout 1800"

# 4. Bring the report back.
scp "$R:sma-swe/eval/*.<run-id>.json" .
```

The agent's settings (step limit, sampling, thinking mode) are SMA's to decide; see [docs/serving-design.md](../../docs/serving-design.md#not-decided-here).

## Notes

- **Workers:** the baseline used one, for clean per-task times. SWE-bench suggests at most 75% of the cores. On 10 vCPUs and 24 GB, 4–6 at once.
- **Containers:** the harness names its containers `sweb.eval.<task>.<run-id>` and removes them when it is done. mini-swe-agent's are `minisweagent-<id>`, started with `--rm`. `docker ps -a` shows anything left behind.
- **Disk:** `docker image prune` removes only dangling layers. The task images stay.
- **LiteLLM:** mini-swe-agent calls the model through LiteLLM. Its 1.107.0 release carried a credential stealer ([BerriAI/litellm#47826](https://github.com/BerriAI/litellm/issues/47826)), so `requirements.txt` holds it at 1.106.2, the last release before. Do not raise it without checking what changed.
