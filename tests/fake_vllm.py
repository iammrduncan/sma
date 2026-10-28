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
        # Chat requests as sent (KernelBench), for tests to check sampling and thinking fields.
        self.chat_requests: list[dict[str, Any]] = []
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
                        'vllm:generation_tokens_total{model_name="fake",engine="0"} '
                        f"{owner.counter}\n"
                        'vllm:prompt_tokens_total{model_name="fake",engine="0"} '
                        f"{owner.counter}\n"
                        'vllm:time_to_first_token_seconds_sum{model_name="fake"} '
                        f"{owner.counter}\n"
                        'vllm:time_to_first_token_seconds_count{model_name="fake"} '
                        f"{owner.counter}\n"
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
                if self.path.startswith("/v1/chat/completions") and payload.get("tools"):
                    # BrowseComp-Plus's agent: one search, then an answer in its format.
                    owner.chat_requests.append(payload)
                    searched = any(m.get("role") == "tool" for m in payload.get("messages") or [])
                    if searched:
                        message = {"role": "assistant", "reasoning": "The hits name the city.",
                                   "content": "Explanation: It is Paris [11].\nExact Answer: Paris\nConfidence: 90%"}
                        finish = "stop"
                    else:
                        message = {"role": "assistant", "reasoning": "Search first.", "content": None, "tool_calls": [
                            {"id": "call-1", "type": "function", "function": {"name": "search", "arguments": '{"query": "capital city"}'}}]}
                        finish = "tool_calls"
                    body = {"choices": [{"message": message, "finish_reason": finish}],
                            "usage": {"prompt_tokens": 300, "completion_tokens": 40}}
                    self._send(200, json.dumps(body).encode())
                    return
                if self.path.startswith("/v1/chat/completions"):
                    owner.chat_requests.append(payload)
                    content = "<think>\nPlan the kernel.\n</think>\n\n```python\nclass ModelNew:\n    pass\n```\n"
                    body = {
                        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    }
                    self._send(200, json.dumps(body).encode())
                    return
                if self.path.startswith("/generate"):
                    ids = payload.get("input_ids") or []
                    k = int(payload.get("top_logprobs_num") or 1)
                    self._send(200, json.dumps(sglang_generate_body([int(i) for i in ids], k)).encode())
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


def sglang_generate_body(token_ids: list[int], top_k: int) -> dict[str, Any]:
    rows = _prompt_logprobs(token_ids, top_k)
    input_top = []
    input_token = []
    for index, row in enumerate(rows):
        if row is None:
            input_top.append(None)
            input_token.append(None)
            continue
        pairs = []
        for key, value in row.items():
            pairs.append([value["logprob"], int(key)])
        input_top.append(pairs)
        true = token_ids[index]
        true_lp = row.get(str(true), {}).get("logprob")
        input_token.append([true_lp, true] if true_lp is not None else None)
    return {"meta_info": {"input_top_logprobs": input_top, "input_token_logprobs": input_token}}


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
