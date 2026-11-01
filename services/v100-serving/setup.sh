#!/usr/bin/env bash
# Build SMA's pinned llama.cpp for the V100 serving pod (Volta, compute capability 7.0).
#
# The pod has no Docker, and llama.cpp's published CUDA images are CUDA 13, which dropped Volta. So SMA
# serves from a source build at the same release the Spark runs (b11960), compiled for sm_70 with the pod's
# CUDA 12.8 toolkit. Run it on the pod as the SMA user; it needs git, cmake, ninja, gcc, and nvcc.
#
#   bash setup.sh [ROOT]        # ROOT defaults to /work/sma
#
# It writes ROOT/llama.cpp-<tag>/build/bin/{llama-server,llama-perplexity} and ROOT/llama.cpp-<tag>/build.json:
# the tag, the commit, the CUDA version, the build flags, and each binary's sha256. Running it again with the
# binaries in place only checks the commit and prints build.json.
set -euo pipefail

TAG="b11960"
COMMIT_EXPECTED="8642541b255bfb5dc39cd351a178be09013a55d6"  # The tag's commit at the first build (2026-11-01), so a moved tag is caught.
ROOT="${1:-/work/sma}"
SRC="$ROOT/llama.cpp-$TAG"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
FLAGS=(
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=70
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF
  -DGGML_NATIVE=OFF
  -DBUILD_SHARED_LIBS=OFF
)

mkdir -p "$ROOT"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
if [ -n "$COMMIT_EXPECTED" ] && [ "$COMMIT" != "$COMMIT_EXPECTED" ]; then
  echo "llama.cpp $TAG is at $COMMIT, not $COMMIT_EXPECTED. The tag moved; check it before building." >&2
  exit 1
fi

BIN="$SRC/build/bin"
if [ ! -x "$BIN/llama-server" ] || [ ! -x "$BIN/llama-perplexity" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" "${FLAGS[@]}"
  cmake --build "$SRC/build" --target llama-server llama-perplexity -j "$(nproc)"
fi

python3 - "$SRC" "$TAG" "$COMMIT" "$CUDA_HOME" "${FLAGS[@]}" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path

src, tag, commit, cuda_home, *flags = sys.argv[1:]
bins = {}
for name in ("llama-server", "llama-perplexity"):
    path = Path(src) / "build" / "bin" / name
    bins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
nvcc = subprocess.run([f"{cuda_home}/bin/nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
record = {"tag": tag, "commit": commit, "cuda": nvcc, "cuda_architectures": "70", "flags": flags, "sha256": bins}
Path(src, "build.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
EOF
