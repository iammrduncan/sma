"""Watch an engine server: follow its log, report startup, check KV capacity, explain a mid-step stop."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from sma.logsetup import log_event

# Steps that send requests to an engine server, and the server each one uses.
STEP_SERVER = {"determinism": "quality", "logprobs": "quality", "quality": "quality", "speed": "speed"}

# vLLM tags each line with its process, for example "(EngineCore pid=308) ".
_PROCESS_TAG = re.compile(r"^\([A-Za-z_]+ pid=\d+\)\s*")
# vLLM's startup line, for example "GPU KV cache size: 886,656 tokens".
_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def server_log_path(run_dir: Path, mode: str) -> Path:
    return run_dir / "logs" / f"server-{mode}.log"


def start_following(runtime: Any, name: str, run_dir: Path, mode: str) -> Any | None:
    """Copy the server's output into its log file while it runs. A failure here never stops a run."""
    path = server_log_path(run_dir, mode)
    try:
        following = runtime.follow_logs(name, path)
    except Exception as exc:
        log_event("server_log_unavailable", mode=mode, error=str(exc))
        return None
    log_event("server_log", mode=mode, path=path)
    return following


def tail_lines(path: Path, count: int) -> list[str]:
    """The last non-empty lines of a log, without vLLM's process tag. Progress bars keep their last frame."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 16384))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines: list[str] = []
    for raw in text.splitlines():
        line = _PROCESS_TAG.sub("", raw.split("\r")[-1].strip())
        if line:
            lines.append(line[:200])
    return lines[-count:]


def report_wait(run_dir: Path, mode: str, elapsed_s: float) -> None:
    latest = tail_lines(server_log_path(run_dir, mode), 1)
    log_event("server_wait", mode=mode, elapsed_s=round(elapsed_s), line=latest[0] if latest else "(no output yet)")


def stopped_server_note(runtime: Any, *, name: str, mode: str, log_path: Path, poll_s: float) -> str | None:
    """Explain a step failure caused by the server stopping. None while the server still runs.

    A server that is shutting down still answers "running" for a moment, so this checks a few times.
    """
    for _ in range(3):
        if not runtime.is_running(name):
            break
        time.sleep(poll_s)
    else:
        return None
    if name not in runtime.list_sma():
        return None
    code = runtime.exit_code(name)
    tail = "\n".join("    " + line for line in tail_lines(log_path, 6))
    return (
        f"The {mode} server stopped while this step was running "
        f"(exit code {code if code is not None else 'unknown'}). Its log is {log_path}. Last lines:\n"
        f"{tail}\n"
        "If the log shows a clean shutdown and no error, something on the host stopped the server. "
        "An out-of-memory guard such as earlyoom does that; check `journalctl -u earlyoom`."
    )


def needed_kv_tokens(cfg: Any) -> int:
    """The most tokens SMA can have in flight at once: its highest concurrency at full context."""
    widest = max([cfg.quality.concurrency, *cfg.speed.sweep_concurrency, 1])
    return int(widest * cfg.engine.max_model_len)


def kv_capacity_warning(log_text: str, needed_tokens: int) -> str | None:
    """A warning when the KV cache vLLM reported cannot hold SMA's widest load. None when it can."""
    found = _KV_TOKENS.findall(log_text)
    if not found:
        return None
    tokens = int(found[-1].replace(",", ""))
    if tokens >= needed_tokens:
        return None
    return (
        f"The KV cache holds {tokens:,} tokens. SMA's widest load can need {needed_tokens:,} "
        "(highest concurrency times engine.max_model_len). Requests past that wait or are preempted, "
        "which changes speed numbers. Raise engine.kv_cache_gib."
    )
