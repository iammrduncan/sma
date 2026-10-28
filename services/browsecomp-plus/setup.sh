#!/bin/sh
# Set up this x86_64 machine as SMA's BrowseComp-Plus runner (README.md): a JDK, the venv, the pinned BM25
# index, the tokenizer, the questions, and the search server. It is safe to run again: the venv is synced
# to requirements.txt, and files that already match their pins are kept.
#   BCP_RUNNER_DIR   where the runner's files go (default ~/sma-bcp)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dir=${BCP_RUNNER_DIR:-$HOME/sma-bcp}
# Temurin 21.0.5+11, the Java the index was built with (Lucene 9.9.1 on Java 21.0.5).
jdk_url=https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.5%2B11/OpenJDK21U-jdk_x64_linux_hotspot_21.0.5_11.tar.gz
jdk_sha256=3c654d98404c073b8a7e66bffb27f4ae3e7ede47d13284c132d40a83144bfd8c

if [ "$(uname -m)" != x86_64 ]; then
  echo "requirements.txt and the JDK are pinned for x86_64, and this machine is $(uname -m)." >&2; exit 1
fi
if ! command -v uv >/dev/null; then
  echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1
fi
free_kb=$(df -Pk "$(dirname "$dir")" | awk 'NR==2 {print $4}')
if [ "$free_kb" -lt 10000000 ]; then
  echo "Less than 10 GB free for $dir. The index, the venv, and the JDK take about 4 GB." >&2; exit 1
fi

mkdir -p "$dir/runs"
cp "$here/requirements.txt" "$here/prepare.py" "$here/search_server.py" "$dir/"
cd "$dir"
if [ ! -x jdk/bin/java ]; then
  curl -fsSL -o jdk.tar.gz "$jdk_url"
  echo "$jdk_sha256  jdk.tar.gz" | sha256sum -c -
  rm -rf jdk && mkdir jdk && tar -xzf jdk.tar.gz -C jdk --strip-components=1 && rm jdk.tar.gz
fi
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --require-hashes --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match requirements.txt
.venv/bin/python prepare.py
# A search through the real server path, so a broken JDK or index shows up here and not in a run.
JAVA_HOME="$dir/jdk" .venv/bin/python - <<'EOF'
import json
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher("indexes/bm25")
hits = searcher.search("deep research agent", 5)
assert searcher.num_docs == 100195, searcher.num_docs
assert len(hits) == 5 and json.loads(hits[0].lucene_document.get("raw"))["contents"]
print(f"search check: {searcher.num_docs} documents, 5 hits")
EOF
echo "Ready in $dir."
