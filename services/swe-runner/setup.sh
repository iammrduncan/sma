#!/bin/sh
# Set up this x86_64 Docker host as SMA's SWE-bench runner (README.md). Run it as the user who will
# run the tasks, a member of the docker group. It is safe to run again: the venv is synced to
# requirements.txt, the dataset is rebuilt and checked, and images already present are kept.
#   SWE_RUNNER_DIR   where the runner's files go (default ~/sma-swe)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dir=${SWE_RUNNER_DIR:-$HOME/sma-swe}

if [ "$(uname -m)" != x86_64 ]; then
  echo "SWE-bench's task images are x86_64 only, and this machine is $(uname -m). Use an x86_64 host." >&2; exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is not reachable as $(id -un). Install Docker Engine and add this user to the docker group." >&2; exit 1
fi
if ! command -v uv >/dev/null; then
  echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1
fi

mkdir -p "$dir/eval" "$dir/runs"
cp "$here/requirements.txt" "$here/build_dataset.py" "$here/images.txt" "$here/pull-images.sh" "$dir/"
cd "$dir"
[ -x .venv/bin/python ] || uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python --require-hashes requirements.txt
.venv/bin/python build_dataset.py
./pull-images.sh
echo "Ready in $dir. Check it with the gold-patch baseline (README.md)."
