"""Small HTTP client for the local vLLM server. Every call has a timeout."""

from __future__ import annotations

from typing import Any, Callable

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
    headers: dict[str, str] | None = None,
) -> Any:
    from sma.retry import call_with_retry

    def _once() -> Any:
        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.request(method, url, json=payload, headers=headers)
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
    engine_label: str = "server",
    health_paths: tuple[str, ...] = ("/health", "/v1/models"),
    alive: Callable[[], bool] | None = None,
    on_wait: Callable[[float], None] | None = None,
    report_every_s: float = 15.0,
) -> None:
    """Poll until the server answers. Stop early when alive() says the server process is gone.

    on_wait(elapsed_s) is called about every report_every_s while the server is not ready yet.
    """
    import time

    started = time.monotonic()
    deadline = started + deadline_s
    reported = started
    last = "no response"
    while time.monotonic() < deadline:
        try:
            for path in health_paths:
                body = request_json(
                    "GET",
                    base_url.rstrip("/") + path,
                    timeout_s=request_timeout_s,
                    attempts=1,
                )
                if path.endswith("/models"):
                    if not (isinstance(body, dict) and body.get("data")):
                        last = f"{path} was empty"
                        raise SmaError(last)
            return
        except (SmaError, OSError) as exc:
            last = scrub_text(str(exc))
        if alive is not None and not alive():
            raise SmaError(f"{engine_label} exited before it became healthy. Last health check: {last}")
        now = time.monotonic()
        if on_wait is not None and now - reported >= report_every_s:
            reported = now
            on_wait(now - started)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_s, remaining))
    raise SmaError(f"{engine_label} did not become healthy within {deadline_s:.0f}s: {last}")


def scrape_metrics(base_url: str, timeout_s: float, *, path: str = "/metrics") -> str:
    url = base_url.rstrip("/") + path
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise SmaError(f"failed to scrape {url}: {scrub_text(str(exc))}") from exc
    if response.status_code != 200:
        raise SmaError(f"{url} returned {response.status_code}")
    return response.text
