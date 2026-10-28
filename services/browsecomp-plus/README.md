# BrowseComp-Plus runner

SMA scores deep research with [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) (MIT): 830 hard questions, each answerable from a fixed set of 100,195 web documents. The model searches those documents with a BM25 tool until it answers, and a judge grades the answer. This folder sets up the machine that holds the documents' index and answers the searches. The model stays on the Spark, and the index takes nothing from it.

The lab runs this on the same x86_64 VM as the [SWE-bench runner](../swe-runner/README.md), in its own folder, `~/sma-bcp`. SMA reaches it with the same `SWE_RUNNER_*` names in the root `.env`.

## Files

| File | What |
| --- | --- |
| `setup.sh` | Sets up the runner: a JDK, the venv, the pinned files, and a search check. Safe to run again |
| `requirements.in`, `requirements.txt` | Pyserini 1.2.0, torch (CPU build), transformers, openai and numpy at the versions in BrowseComp-Plus's own `uv.lock`, and what `prepare.py` needs. `requirements.txt` pins every package with its hashes, for Python 3.11 on x86_64 Linux |
| `prepare.py` | Downloads the pinned index, tokenizer, questions and qrels, and checks each against its SHA-256 |
| `search_server.py` | BrowseComp-Plus's search tool as a small HTTP server on the runner's loopback |

## What you need

- An x86_64 Linux machine with [uv](https://docs.astral.sh/uv/getting-started/installation/), curl and about 10 GB free. It needs no GPU and no Docker. On the lab's runner the search server used about 0.7 GB of memory, plus the index in the page cache.
- Internet access, for PyPI, the PyTorch CPU index, GitHub and Hugging Face.

## Set up

As the runner's user:

```bash
sma/services/browsecomp-plus/setup.sh          # or BCP_RUNNER_DIR=/srv/sma-bcp sma/services/browsecomp-plus/setup.sh
```

In `~/sma-bcp`, it:

1. Unpacks Temurin JDK 21.0.5+11, checked against its SHA-256. The index was built with Lucene 9.9.1 on Java 21.0.5.
2. Makes `.venv` with Python 3.11 and syncs it to `requirements.txt`, checking every hash.
3. Runs `prepare.py`:
   - **`indexes/bm25`:** the prebuilt BM25 index, [`Tevatron/browsecomp-plus-indexes`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes) at `b3f37f70`, 15 files and 2.17 GB. Each file is checked against the SHA-256 that commit records.
   - **`data/tokenizer`:** Qwen3-0.6B's tokenizer at `c1899de2`, which cuts each search hit to 512 tokens, as the client does.
   - **`data/queries.jsonl`:** the 830 questions and answers from [`Tevatron/browsecomp-plus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus) at `144cff8e`, decrypted. Only the query and answer columns are read, not the 2.8 GB of documents bundled with them. The file is checked against a pinned SHA-256 and readable only by its owner.
   - **`data/qrel_evidence.txt`, `data/qrel_golds.txt`:** which documents hold each question's evidence and its answer, from the repo at `04694903`.
4. Opens the index and runs one search.

BrowseComp-Plus asks that its questions and answers never appear as plain text online. They stay in `data/queries.jsonl` on the runner and, during a run, in SMA's memory and in the run's git-ignored files.

## The search server

SMA starts it for each run and stops it at the end. On the lab's runner (2026-10-28) it opened the index in 14 s, and a search through SMA's SSH forward took 0.5 to 0.7 s:

```bash
cd ~/sma-bcp && .venv/bin/python search_server.py --index indexes/bm25 --tokenizer data/tokenizer \
    --port 8790 --snippet-max-tokens 512 --tag runs/<run-id>
```

- `GET /health` answers `{"ok": true, "docs": 100195, ...}` once the index is open.
- `POST /search {"query": ..., "k": 5}` answers `{"output": ...}`, where `output` is exactly the text BrowseComp-Plus's client gives the model: a JSON list of the top `k` hits, each with `docid`, `score` and `snippet`.

The search is BrowseComp-Plus's own: Pyserini's `LuceneSearcher` with BM25 at k1=0.9 and b=0.4 (`searcher/searchers/bm25_searcher.py`), and each hit's text cut to its first 512 tokens with Qwen3-0.6B's tokenizer (`search_agent/glm_zai_client.py`).

The server listens on 127.0.0.1 only and handles one request at a time. It logs no query text, because the queries come from the questions. SMA reaches it through its own SSH connection, as a forward from the Spark's loopback, port 8790 by default (`browsecomp.search_port` and `browsecomp.local_port`).

## How SMA uses it

See the README's run steps and `sma run <model> --dry-run`. In short:

1. On the quality server, SMA asks 100 of the 830 questions, the same seeded 100 for every model, 4 at a time, as BrowseComp-Plus's chat client does.
2. After the speed benchmark, the judge grades each final answer with BrowseComp-Plus's grader prompt. The judge is [`judge-info.md`](../../judge-info.md).
3. The headline is accuracy over the 100. Evidence recall and search calls need no judge.
