"""Subprocess helper with a timeout and a process-group kill."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from sma.errors import SmaError
from sma.logsetup import log_event
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


def run_command(
    argv: list[str],
    *,
    timeout_s: float,
    what: str,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            start_new_session=True,
            check=False,
            cwd=cwd,
            env={**os.environ, **extra_env} if extra_env else None,
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


def run_command_streaming(argv: list[str], *, timeout_s: float, what: str) -> CommandResult:
    """Run a command and log each stderr line as it arrives. Used for image pulls."""
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise SmaError(f"{what} could not start; {argv[0]} is not installed") from exc
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def _read(pipe: IO[str] | None, sink: list[str], *, progress: bool) -> None:
        if pipe is None:
            return
        for line in iter(pipe.readline, ""):
            sink.append(line)
            if progress and line.strip():
                log_event("image_pull_progress", line=scrub_text(line.strip())[:240])
        pipe.close()

    stdout_thread = threading.Thread(
        target=_read, args=(process.stdout, stdout_parts), kwargs={"progress": False}, daemon=True
    )
    stderr_thread = threading.Thread(
        target=_read, args=(process.stderr, stderr_parts), kwargs={"progress": True}, daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        raise SmaError(f"{what} timed out after {timeout_s:.0f}s") from exc
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    return CommandResult(
        argv=list(argv),
        returncode=returncode,
        stdout=scrub_text("".join(stdout_parts)),
        stderr=scrub_text("".join(stderr_parts)),
    )


@dataclass
class Following:
    """A long-running command whose output is being copied into a file."""

    process: subprocess.Popen[str]
    thread: threading.Thread
    path: Path

    def stop(self, timeout_s: float = 10.0) -> None:
        """Let the command end on its own for up to timeout_s, then end it. Joins the copier."""
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.thread.join(timeout=5)


def follow_to_file(argv: list[str], path: Path, *, what: str) -> Following:
    """Start a command and copy its stdout and stderr, interleaved as they arrive and scrubbed, into path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise SmaError(f"{what} could not start; {argv[0]} is not installed") from exc

    def _copy() -> None:
        pipe = process.stdout
        if pipe is None:
            return
        with path.open("w", encoding="utf-8") as sink:
            for line in iter(pipe.readline, ""):
                sink.write(scrub_text(line))
                sink.flush()
        pipe.close()

    thread = threading.Thread(target=_copy, name=f"follow {what}", daemon=True)
    thread.start()
    return Following(process=process, thread=thread, path=path)


def require_ok(result: CommandResult, *, what: str) -> CommandResult:
    if result.ok:
        return result
    detail = (result.stderr or result.stdout or "no output").strip()
    raise SmaError(f"{what} failed (exit {result.returncode}): {detail[-800:]}")
