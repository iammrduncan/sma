"""BrowseComp-Plus's BM25 search tool as a small HTTP server on the runner's loopback (README.md).

The search is BrowseComp-Plus's own at 04694903: searcher/searchers/bm25_searcher.py (Pyserini's
LuceneSearcher over the prebuilt index, BM25 k1=0.9 b=0.4) and search_agent/glm_zai_client.py
SearchToolHandler._search (each hit's text cut to its first N tokens with Qwen3-0.6B's tokenizer). The
reply's "output" is the exact text the client gives the model.

  GET  /health                         -> {"ok": true, "docs": <documents in the index>}
  POST /search {"query": str, "k": int} -> {"output": <JSON text of [{docid, score, snippet}]>}

One request at a time. It listens on 127.0.0.1 only; SMA reaches it through an SSH forward. It logs no
query text, because the queries come from questions that must not appear as plain text online.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

here = Path(__file__).resolve().parent
if not os.environ.get("JAVA_HOME") and (here / "jdk").is_dir():
    os.environ["JAVA_HOME"] = str(here / "jdk")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", required=True, help="the Lucene index folder")
    parser.add_argument("--tokenizer", required=True, help="the Qwen3-0.6B tokenizer folder")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--tag", default="", help="names the run, so SMA can stop this server by its command line")
    args = parser.parse_args()

    from pyserini.search.lucene import LuceneSearcher
    from transformers import AutoTokenizer

    searcher = LuceneSearcher(str(Path(args.index)))
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.tokenizer))) if args.snippet_max_tokens > 0 else None
    docs = searcher.num_docs
    searches = 0

    def search(query: str, k: int) -> str:
        results = []
        for hit in searcher.search(query, k):
            text = json.loads(hit.lucene_document.get("raw"))["contents"]
            snippet = text
            if tokenizer is not None:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > args.snippet_max_tokens:
                    snippet = tokenizer.decode(tokens[: args.snippet_max_tokens], skip_special_tokens=True)
            results.append({"docid": hit.docid, "score": hit.score, "snippet": snippet})
        return json.dumps(results, indent=2)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            if self.path == "/health":
                self._reply(200, {"ok": True, "docs": docs, "searches": searches})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            nonlocal searches
            if self.path != "/search":
                self._reply(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                query, k = str(body["query"]), int(body.get("k", 5))
            except (ValueError, KeyError, TypeError) as exc:
                self._reply(400, {"error": f"expected {{query, k}}: {type(exc).__name__}"})
                return
            output = search(query, k)
            searches += 1
            self._reply(200, {"output": output})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server's name
            return  # No request lines: they would hold the query text.

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"search server ready on 127.0.0.1:{args.port}: {docs} documents, snippets of {args.snippet_max_tokens} tokens", flush=True)
    try:
        server.serve_forever()
    finally:
        print(f"search server stopped after {searches} searches", flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
