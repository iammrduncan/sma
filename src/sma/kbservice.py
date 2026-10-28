"""Client for the KernelBench service that compiles, checks, and times kernels on a separate GB10.

The API is in runs/gb10/baseline/kernelbench/README.md. The URL comes from KERNELBENCH_URL in the root
.env, so it never goes into sma.yaml; run records name it ${KERNELBENCH_URL}.
"""

from __future__ import annotations

import os
from typing import Any

from sma.errors import SmaError
from sma.httpapi import request_json

URL_ENV = "KERNELBENCH_URL"
_DONE = {"done", "failed", "cancelled"}


class KernelBenchService:
    def __init__(self, url: str, *, timeout_s: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        return request_json(method, self.url + path, timeout_s=self.timeout_s, payload=payload, attempts=4, backoff_s=5.0)

    def environment(self) -> dict[str, Any]:
        return self._call("GET", "/v1/environment")

    def submit(self, *, level: int, name: str, kernels: list[dict[str, Any]], options: dict[str, Any]) -> str:
        reply = self._call("POST", "/v1/jobs", {"level": level, "name": name, "kernels": kernels, "options": options})
        job_id = reply.get("job_id") if isinstance(reply, dict) else None
        if not job_id:
            raise SmaError("The KernelBench service accepted the job but returned no job_id.")
        return str(job_id)

    def job(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/jobs/{job_id}")

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
