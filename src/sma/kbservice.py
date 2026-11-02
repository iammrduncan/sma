"""Client for the KernelBench service that compiles, checks, and times kernels on a GB10, one job at a time.

The API is in runs/gb10/baseline/kernelbench/README.md. The URL comes from KERNELBENCH_URL in the root
.env, so it never goes into sma.yaml; run records name it ${KERNELBENCH_URL}.
"""

from __future__ import annotations

import os
from typing import Any

from sma.errors import SmaError
from sma.httpapi import ApiError, request_json

URL_ENV = "KERNELBENCH_URL"


class QueueFull(SmaError):
    """The service already holds its most queued jobs (429)."""

_DONE = {"done", "failed", "cancelled"}


class KernelBenchService:
    def __init__(self, url: str, *, timeout_s: float = 60.0, attempts: int = 4) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.attempts = attempts

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        return request_json(method, self.url + path, timeout_s=self.timeout_s, payload=payload, attempts=self.attempts, backoff_s=5.0)

    def environment(self) -> dict[str, Any]:
        return self._call("GET", "/v1/environment")

    def submit(self, *, level: int, name: str, kernels: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Queue a job. The reply: job_id, and position (jobs ahead of it, the running one included). A full queue
        (429, 50 queued) raises QueueFull after the client's retries."""
        try:
            reply = self._call("POST", "/v1/jobs", {"level": level, "name": name, "kernels": kernels, "options": options})
        except ApiError as exc:
            if "returned 429" in str(exc):
                raise QueueFull(str(exc)) from exc
            raise
        job_id = reply.get("job_id") if isinstance(reply, dict) else None
        if not job_id:
            raise SmaError("The KernelBench service accepted the job but returned no job_id.")
        return {**reply, "job_id": str(job_id)}

    def job(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/jobs/{job_id}")

    def jobs(self) -> list[dict[str, Any]]:
        """The service's latest 100 jobs, newest first, without results: anyone's, not only SMA's."""
        reply = self._call("GET", "/v1/jobs")
        rows = reply.get("jobs") if isinstance(reply, dict) else None
        return [row for row in rows or [] if isinstance(row, dict)]

    def baseline_ids(self, *, level: int, options: dict[str, Any], fingerprint: str) -> set[int]:
        """The problems with a stored eager-PyTorch time for this level, options, and environment."""
        name = f"level{level}-{options['precision']}-{options['timing_method']}-{options['num_perf_trials']}.json"
        payload = self._call("GET", f"/v1/baselines/{name}?fingerprint={fingerprint}")
        problems = payload.get("problems") if isinstance(payload, dict) else None
        if not isinstance(problems, list):
            raise SmaError(f"The KernelBench service has no baseline {name} for environment {fingerprint}.")
        return {int(item["problem_id"]) for item in problems if isinstance(item, dict) and item.get("stats")}


def service_from_env() -> KernelBenchService | None:
    url = os.environ.get(URL_ENV)
    return KernelBenchService(url) if url else None


def finished(job: dict[str, Any]) -> bool:
    return str(job.get("status")) in _DONE


def queue_position(jobs: list[dict[str, Any]], job_id: str) -> int | None:
    """How many jobs are ahead of job_id in the service's FIFO: the running job, then queued jobs created before
    it. None when job_id is not queued."""
    mine = next((row for row in jobs if row.get("id") == job_id), None)
    if mine is None or mine.get("status") != "queued":
        return None
    created = str(mine.get("created_at") or "")
    ahead = [row for row in jobs if row.get("status") == "running"
             or (row.get("status") == "queued" and str(row.get("created_at") or "") < created)]
    return len(ahead)
