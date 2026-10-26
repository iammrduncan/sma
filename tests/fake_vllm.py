"""A tiny OpenAI-compatible server for tests. It does not run a model."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeVLLM:
    def __init__(self, port: int) -> None:
        self.port = port
        self.nondeterministic = False
        self.counter = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/metrics"):
                    owner.counter += 1
                    body = (
                        f"vllm:generation_tokens_total {owner.counter}\n"
                        f"vllm:prompt_tokens_total {owner.counter}\n"
                        f"vllm:time_to_first_token_seconds_sum {owner.counter}\n"
                        f"vllm:time_to_first_token_seconds_count {owner.counter}\n"
                    )
                    self._send(200, body.encode(), "text/plain")
                    return
                if self.path.startswith("/v1/models"):
                    self._send(200, json.dumps({"data": [{"id": "fake"}]}).encode())
                    return
                self._send(200, b'{"status":"ok"}')

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode() or "{}")
                except json.JSONDecodeError:
                    self._send(400, b'{"error":"bad json"}')
                    return
                if not isinstance(payload, dict):
                    self._send(400, b'{"error":"bad payload"}')
                    return
                body = completion_body(payload, nondeterministic=owner.nondeterministic)
                self._send(200, json.dumps(body).encode())

            def _send(self, code: int, data: bytes, content_type: str = "application/json") -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        ThreadingHTTPServer.allow_reuse_address = True
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        deadline = time.time() + 2
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.01)
        raise RuntimeError(f"fake vLLM did not listen on {self.port}")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None


def completion_body(payload: dict[str, Any], *, nondeterministic: bool) -> dict[str, Any]:
    prompt = payload.get("prompt")
    top_k = int(payload.get("prompt_logprobs") or 0)
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
        return {
            "choices": [
                {
                    "text": "",
                    "prompt_logprobs": _prompt_logprobs([int(token) for token in prompt], top_k),
                }
            ]
        }
    if isinstance(prompt, str):
        prompts = [prompt]
    elif isinstance(prompt, list):
        prompts = [str(item) for item in prompt]
    else:
        prompts = [""]
    texts = []
    for index, item in enumerate(prompts):
        if nondeterministic:
            texts.append(f"ok:{item}:{index}")
        else:
            texts.append(f"ok:{item}")
    return {"choices": [{"text": text, "index": index} for index, text in enumerate(texts)]}


def _prompt_logprobs(token_ids: list[int], top_k: int) -> list[dict[str, Any] | None]:
    k = max(1, top_k)
    rows: list[dict[str, Any] | None] = [None]
    for index in range(1, len(token_ids)):
        true_id = token_ids[index]
        ids = [true_id] + [true_id + 1 + offset for offset in range(k - 1)]
        raw = [0.5 ** (offset + 1) for offset in range(k)]
        scale = 0.9 / sum(raw)
        entry = {}
        for rank, token_id in enumerate(ids):
            prob = raw[rank] * scale
            entry[str(token_id)] = {"logprob": math.log(prob), "rank": rank + 1}
        rows.append(entry)
    return rows
