"""One exclusive file lock per Spark so two SMA processes cannot share the GPU."""

from __future__ import annotations

import os
from pathlib import Path

from sma.errors import SmaError


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
