"""SMA's grading turn for LiveCodeBench (sitecustomize.py and regrade.py use it).

SMA_GRADING_LOCK: one container on the machine grades at a time, by an flock on a file SMA mounts in every
LiveCodeBench container, so runs grading together cannot push a test over its 6 s limit. SMA_GRADING_CPUS:
grading runs on those CPUs (the harness's timed CPUs), and the grading processes inherit them. With neither
set, nothing changes.
"""

import sys


def cpu_list(text):
    """A CPU list ("5-9", "0-4,10") as a set of CPU numbers."""
    cpus = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


class grading_turn:
    """SMA_GRADING_LOCK: one container on the machine grades at a time (an flock on a file SMA mounts in every
    LiveCodeBench container), so runs grading together cannot push a test over its time limit. SMA_GRADING_CPUS:
    grading runs on those CPUs, which the grading processes inherit. Neither set: nothing changes."""

    def __enter__(self):
        import os
        import time

        self.fd, self.previous = None, None
        lock = os.environ.get("SMA_GRADING_LOCK")
        if lock:
            import fcntl

            self.fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o666)
            started = time.monotonic()
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            print(f"SMA patch: grading lock {lock} taken after {time.monotonic() - started:.0f} s", file=sys.stderr, flush=True)
        cpus = os.environ.get("SMA_GRADING_CPUS")
        if cpus:
            self.previous = os.sched_getaffinity(0)
            os.sched_setaffinity(0, cpu_list(cpus))
            print(f"SMA patch: grading on CPUs {cpus}", file=sys.stderr, flush=True)
        return self

    def __exit__(self, *exc):
        import os

        if self.previous is not None:
            os.sched_setaffinity(0, self.previous)
        if self.fd is not None:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        return False
