"""Small HTTP client for the local vLLM server. Every call has a timeout."""

from __future__ import annotations

from typing import Any

import httpx

from sma.errors import SmaError
from sma.scrub import scrub_text


class ApiError(SmaError):
    pass


def request_json(
    method: str,
    url: str,
    *,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
    attempts: int = 3,
    backoff_s: float = 0.5,
) -> Any:
    from sma.retry import call_with_retry

    def _once() -> Any:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.request(method, url, json=payload)
        except httpx.TimeoutException as exc:
            raise ApiError(f"timed out calling {url}") from exc
        except httpx.HTTPError as exc:
            raise ApiError(f"HTTP error calling {url}: {scrub_text(str(exc))}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            raise ApiError(f"{url} returned {response.status_code}: {scrub_text(response.text)[:400]}")
        if response.status_code >= 400:
            raise SmaError(
                f"{url} returned {response.status_code}: {scrub_text(response.text)[:600]}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise SmaError(f"{url} did not return JSON") from exc

    return call_with_retry(
        _once,
        attempts=attempts,
        backoff_s=backoff_s,
        factor=2.0,
        what=f"{method} {url}",
        retry_on=(ApiError,),
    )


def wait_until_healthy(
    base_url: str,
    *,
    deadline_s: float,
    request_timeout_s: float,
    poll_s: float,
) -> None:
    import time

    deadline = time.monotonic() + deadline_s
    last = "no response"
    while time.monotonic() < deadline:
        try:
            request_json(
                "GET",
                base_url.rstrip("/") + "/health",
                timeout_s=request_timeout_s,
                attempts=1,
            )
            models = request_json(
                "GET",
                base_url.rstrip("/") + "/v1/models",
                timeout_s=request_timeout_s,
                attempts=1,
            )
            if isinstance(models, dict) and models.get("data"):
                return
            last = "health passed but /v1/models was empty"
        except (SmaError, OSError) as exc:
            last = scrub_text(str(exc))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_s, remaining))
    raise SmaError(f"vLLM did not become healthy within {deadline_s:.0f}s: {last}")


def scrape_metrics(base_url: str, timeout_s: float) -> str:
    url = base_url.rstrip("/") + "/metrics"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise SmaError(f"failed to scrape {url}: {scrub_text(str(exc))}") from exc
    if response.status_code != 200:
        raise SmaError(f"{url} returned {response.status_code}")
    return response.text
