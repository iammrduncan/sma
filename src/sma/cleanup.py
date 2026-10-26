"""Page-cache drop and idle-until-cool. Both are recorded; neither is guessed."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from sma.scrub import scrub_text


def drop_page_cache() -> dict[str, object]:
    """Drop the Linux page cache when the process is allowed to. Never raises."""
    try:
        subprocess.run(["sync"], timeout=60, check=False)
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
    except PermissionError as exc:
        return {
            "attempted": True,
            "permitted": False,
            "ok": False,
            "error": scrub_text(str(exc)),
        }
    except OSError as exc:
        return {
            "attempted": True,
            "permitted": False,
            "ok": False,
            "error": scrub_text(str(exc)),
        }
    return {"attempted": True, "permitted": True, "ok": True, "error": None}


def idle_until_cool(
    read_temp: Callable[[], float | None],
    *,
    target_c: float,
    timeout_s: float,
    poll_s: float,
) -> dict[str, object]:
    started = time.time()
    deadline = started + max(0.0, timeout_s)
    first = read_temp()
    last = first
    reached = first is not None and first <= target_c
    while not reached and time.time() < deadline:
        delay = min(poll_s, max(0.0, deadline - time.time()))
        if delay > 0:
            time.sleep(delay)
        last = read_temp()
        if last is not None and last <= target_c:
            reached = True
            break
    return {
        "target_c": target_c,
        "start_c": first,
        "end_c": last,
        "reached": reached,
        "elapsed_s": round(time.time() - started, 3),
    }
