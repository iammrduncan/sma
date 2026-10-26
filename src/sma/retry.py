"""Bounded retries with exponential backoff for external calls."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from sma.errors import SmaError
from sma.logsetup import log_event
from sma.scrub import scrub_text

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    backoff_s: float,
    factor: float,
    what: str,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    if attempts < 1:
        raise SmaError(f"{what}: attempts must be >= 1")
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if attempt >= attempts:
                break
            delay = backoff_s * (factor ** (attempt - 1))
            log_event(
                "retry",
                what=what,
                attempt=attempt,
                sleep_s=round(delay, 3),
                error=scrub_text(str(exc)),
            )
            if delay > 0:
                time.sleep(delay)
    detail = scrub_text(str(last)) if last else "unknown error"
    raise SmaError(f"{what} failed after {attempts} attempts: {detail}") from last
