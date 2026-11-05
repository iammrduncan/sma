"""The judge model, an OpenAI-compatible chat endpoint (owner decision 2026-10-27: judges are allowed).

A lab endpoint comes from the root .env (SMA_JUDGE_URL), never from sma.yaml or a run folder. A public
service (OpenRouter, owner decision 2026-11-04) is named in sma.yaml as browsecomp.judge_url. The key is
always in .env, under browsecomp.judge_key_env, which SMA scrubs from records. The model is set in sma.yaml,
and every score it grades names it: scores are compared only within one judge.
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

        extra_body is merged into the request (OpenRouter's reasoning and provider settings).
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


def judge_endpoint(url: str | None, key_env: str) -> tuple[str | None, str | None]:
    """The judge's base URL and key: browsecomp.judge_url when set, else SMA_JUDGE_URL; the key from key_env."""
    return (url or os.environ.get(URL_ENV) or None), (os.environ.get(key_env) or None)


def judge_needs(url: str | None, key_env: str) -> str:
    """The .env names the judge needs, for an error that says what to set."""
    return key_env if url else f"{URL_ENV} and {key_env}"


def judge_from_env(
    model: str, *, url: str | None = None, key_env: str = KEY_ENV, timeout_s: float, attempts: int, backoff_s: float
) -> Judge | None:
    base, key = judge_endpoint(url, key_env)
    if not base or not key:
        return None
    return Judge(url=base, key=key, model=model, timeout_s=timeout_s, attempts=attempts, backoff_s=backoff_s)
