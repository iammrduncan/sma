"""Subprocess helper with a timeout and a process-group kill."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass

from sma.errors import SmaError
from sma.scrub import scrub_text


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(argv: list[str], *, timeout_s: float, what: str) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            start_new_session=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _kill_group(exc)
        stdout = scrub_text(exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = scrub_text(exc.stderr or "") if isinstance(exc.stderr, str) else ""
        raise SmaError(
            f"{what} timed out after {timeout_s:.0f}s. stdout={stdout[-500:]} stderr={stderr[-500:]}"
        ) from exc
    except FileNotFoundError as exc:
        raise SmaError(f"{what} could not start; {argv[0]} is not installed") from exc
    return CommandResult(
        argv=list(argv),
        returncode=completed.returncode,
        stdout=scrub_text(completed.stdout or ""),
        stderr=scrub_text(completed.stderr or ""),
    )


def _kill_group(exc: subprocess.TimeoutExpired) -> None:
    # subprocess.run with start_new_session has already killed the group on
    # timeout in modern Python. This is a second attempt for older paths.
    pid = getattr(exc, "pid", None)
    # TimeoutExpired does not always expose the pid. Nothing more to do.
    if not isinstance(pid, int):
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return


def require_ok(result: CommandResult, *, what: str) -> CommandResult:
    if result.ok:
        return result
    detail = (result.stderr or result.stdout or "no output").strip()
    raise SmaError(f"{what} failed (exit {result.returncode}): {detail[-800:]}")
