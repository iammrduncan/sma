"""The lab's judge model, an OpenAI-compatible chat endpoint (owner decision 2026-10-27: judges are allowed).

The endpoint and its key come from the root .env (SMA_JUDGE_URL, SMA_JUDGE_API_KEY), never from sma.yaml or
a run folder. The model is set in sma.yaml, and every score it grades names it: scores are compared only
within one judge.
"""

from __future__ import annotations

import os
from typing import Any

from sma.httpapi import request_json

URL_ENV, KEY_ENV = "SMA_JUDGE_URL", "SMA_JUDGE_API_KEY"


class Judge:
    def __init__(self, *, url: str, key: str, model: str, timeout_s: float, attempts: int, backoff_s: float) -> None:
        base = url.rstrip("/")
        self.url = (base[: -len("/v1")] if base.endswith("/v1") else base) + "/v1/chat/completions"
        self.key, self.model = key, model
        self.timeout_s, self.attempts, self.backoff_s = timeout_s, attempts, backoff_s

    def ask(
        self, prompt: str, *, max_tokens: int, temperature: float | None, extra_body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """One user message. The reply's text, its reasoning when the server returns it apart, and usage.

        extra_body is merged into the request (chat_template_kwargs, llama.cpp's cache_prompt).
        """
        body: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra_body or {})
        reply = request_json(
            "POST",
            self.url,
            timeout_s=self.timeout_s,
            payload=body,
            attempts=self.attempts,
            backoff_s=self.backoff_s,
            headers={"Authorization": f"Bearer {self.key}"},
        )
        choice = ((reply or {}).get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "model": (reply or {}).get("model") or self.model,
            "content": message.get("content"),
            "reasoning": message.get("reasoning_content") or message.get("reasoning"),
            "finish_reason": choice.get("finish_reason"),
            "usage": (reply or {}).get("usage"),
        }


def judge_from_env(model: str, *, timeout_s: float, attempts: int, backoff_s: float) -> Judge | None:
    url, key = os.environ.get(URL_ENV), os.environ.get(KEY_ENV)
    if not url or not key:
        return None
    return Judge(url=url, key=key, model=model, timeout_s=timeout_s, attempts=attempts, backoff_s=backoff_s)
