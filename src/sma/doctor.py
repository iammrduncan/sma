"""Preflight checks. Nothing here starts a model or pulls an image."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from sma.commands import image_ref
from sma.config import Config, hf_token
from sma.errors import SmaError
from sma.identity import assert_alias_safe
from sma.proc import CommandResult, run_command
from sma.scrub import assert_text_clean


@dataclass
class Check:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _try_command(argv: list[str], *, timeout_s: float, what: str) -> CommandResult:
    try:
        return run_command(argv, timeout_s=timeout_s, what=what)
    except SmaError as exc:
        return CommandResult(argv=list(argv), returncode=127, stdout="", stderr=str(exc))


def _disk_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def default_checks(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    try:
        assert_alias_safe(cfg.host_alias)
        checks.append(Check("host_alias", "ok", cfg.host_alias))
    except Exception as exc:
        checks.append(Check("host_alias", "fail", str(exc)))

    docker = _try_command(["docker", "info"], timeout_s=20, what="docker info")
    if docker.ok:
        checks.append(Check("docker", "ok", "daemon reachable"))
    else:
        checks.append(Check("docker", "fail", (docker.stderr or docker.stdout or "docker info failed")[-300:]))

    if docker.ok:
        inspected = _try_command(
            ["docker", "image", "inspect", image_ref(cfg)],
            timeout_s=30,
            what="docker image inspect",
        )
        if inspected.ok:
            checks.append(Check("ngc_image", "ok", image_ref(cfg)))
        else:
            checks.append(
                Check(
                    "ngc_image",
                    "fail",
                    f"{image_ref(cfg)} is not local. Pull it before a run. Doctor does not pull.",
                )
            )
        leftovers = _try_command(
            ["docker", "ps", "-a", "--filter", f"name={cfg.engine.container_prefix}-", "--format", "{{.Names}}"],
            timeout_s=20,
            what="docker ps",
        )
        names = [line.strip() for line in leftovers.stdout.splitlines() if line.strip()] if leftovers.ok else []
        if names:
            checks.append(Check("containers", "warn", "leftover SMA containers: " + ", ".join(names)))
        else:
            checks.append(Check("containers", "ok", "no SMA containers"))
    else:
        checks.append(Check("ngc_image", "fail", "docker is required to inspect the NGC image"))

    gpu = _try_command(["nvidia-smi", "-L"], timeout_s=20, what="nvidia-smi")
    if gpu.ok and gpu.stdout.strip():
        checks.append(Check("gpu", "ok", gpu.stdout.strip().splitlines()[0][:200]))
        procs = _try_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ],
            timeout_s=20,
            what="nvidia-smi processes",
        )
        busy = [line.strip() for line in procs.stdout.splitlines() if line.strip()] if procs.ok else []
        if not procs.ok:
            checks.append(Check("gpu_idle", "fail", (procs.stderr or "could not query compute apps")[-300:]))
        elif busy:
            checks.append(Check("gpu_idle", "fail", "GPU is not idle: " + "; ".join(busy[:5])))
        else:
            checks.append(Check("gpu_idle", "ok", "no compute processes"))
    else:
        detail = (gpu.stderr or gpu.stdout or "nvidia-smi failed")[-300:]
        checks.append(Check("gpu", "fail", detail))
        checks.append(Check("gpu_idle", "fail", "GPU was not found"))

    try:
        free = _disk_gb(cfg.runs_path())
        if free < cfg.doctor.min_free_gb:
            checks.append(
                Check("disk", "fail", f"{free:.1f} GB free, need at least {cfg.doctor.min_free_gb:.0f} GB")
            )
        else:
            checks.append(Check("disk", "ok", f"{free:.1f} GB free"))
    except OSError as exc:
        checks.append(Check("disk", "fail", str(exc)))

    # Fetch refuses a snapshot path that contains the hostname, for example /home/<user> when the
    # user name is the hostname. Same rule, checked before a run.
    cache = str(cfg.cache_path())
    try:
        assert_text_clean(cache, "Hugging Face cache path")
        checks.append(Check("hf_cache", "ok", cache))
    except SmaError as exc:
        checks.append(Check("hf_cache", "fail", str(exc)))

    checks.append(
        Check(
            "hf_token",
            "ok" if hf_token(cfg) else "fail",
            f"{cfg.hf.token_env} is set" if hf_token(cfg) else f"{cfg.hf.token_env} is not set",
        )
    )
    if shutil.which(cfg.quality.executable):
        checks.append(Check("quality_evaluator", "ok", cfg.quality.executable))
    else:
        checks.append(
            Check(
                "quality_evaluator",
                "warn",
                f"{cfg.quality.executable} is not on PATH. Quality suites will fail until it is installed "
                "(`uv sync --extra quality`).",
            )
        )
    return checks


def format_checks(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        lines.append(f"{check.status:4}  {check.name:12}  {check.detail}")
    return "\n".join(lines) + "\n"


def failed(checks: list[Check]) -> bool:
    return any(check.status == "fail" for check in checks)
