"""File locks: one per GPU so two SMA processes cannot share it, and turns in arrival order (FifoLock)."""

from __future__ import annotations

import os
from pathlib import Path

from sma.errors import SmaError


def lock_held(path: Path) -> bool | None:
    """Whether some process holds a flock on path, read from /proc/locks without touching the lock (sma watch only
    reads). None when that cannot be told: no such file, or no /proc/locks."""
    try:
        st = os.stat(path)
        text = Path("/proc/locks").read_text()
    except OSError:
        return None
    ident = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    for line in text.splitlines():
        # "1: FLOCK  ADVISORY  WRITE 2509 103:02:2236626 0 EOF"; a blocked waiter's line has "->" after the number.
        fields = [field for field in line.split() if field != "->"]
        if len(fields) > 5 and fields[1] == "FLOCK" and fields[5] == ident:
            return True
    return False


class GpuLock:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise SmaError(
                f"GPU lock is held ({self.path}). Another SMA process is using this Spark. "
                "One model runs at a time; there is no tensor parallelism across processes."
            ) from exc
        except OSError as exc:
            os.close(fd)
            raise SmaError(f"Could not take the GPU lock at {self.path}: {exc}") from exc
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "GpuLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class FifoLock:
    """Turns in arrival order among processes on one machine (the speed steps of the serving machine's slots).

    Each waiter writes a ticket named by its arrival time and pid in <path>.d/. It takes its turn when its ticket is
    the oldest whose process is alive, then holds an flock on <path> while it runs, so even two waiters that both
    think they are first cannot overlap. A ticket whose process died is removed by the next waiter.
    """

    def __init__(self, path: Path, *, poll_s: float = 2.0) -> None:
        self.path = Path(path)
        self.tickets = self.path.with_name(self.path.name + ".d")
        self.poll_s = poll_s
        self._ticket: Path | None = None
        self._fd: int | None = None

    def acquire(self) -> None:
        import fcntl
        import time

        self.tickets.mkdir(parents=True, exist_ok=True)
        self._ticket = self.tickets / f"{time.time_ns():020d}-{os.getpid()}"
        self._ticket.write_text("", encoding="utf-8")
        try:
            while self._first() != self._ticket.name:
                time.sleep(self.poll_s)
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._fd = fd
        except BaseException:
            self._ticket.unlink(missing_ok=True)
            self._ticket = None
            raise

    def _first(self) -> str | None:
        for ticket in sorted(self.tickets.iterdir()):
            pid = ticket.name.rsplit("-", 1)[-1]
            if pid.isdigit() and _alive(int(pid)):
                return ticket.name
            ticket.unlink(missing_ok=True)
        return None

    def release(self) -> None:
        if self._fd is not None:
            import fcntl

            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        if self._ticket is not None:
            self._ticket.unlink(missing_ok=True)
            self._ticket = None

    def __enter__(self) -> "FifoLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SharedFileLock:
    """A blocking flock on a file other processes and containers share: LiveCodeBench's grading lock, which
    its patch takes inside each container on the same file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None
