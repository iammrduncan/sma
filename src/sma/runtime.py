"""Docker runtime for the NGC vLLM container. Tests use a fake with the same methods."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sma.commands import (
    docker_inspect_argv,
    docker_logs_argv,
    docker_pull_argv,
    docker_rm_argv,
    docker_stop_argv,
)
from sma.config import Config
from sma.errors import SmaError
from sma.logsetup import log_event
from sma.proc import require_ok, run_command
from sma.retry import call_with_retry
from sma.scrub import scrub_text


@dataclass
class StartSpec:
    name: str
    docker_argv: list[str]
    env: dict[str, str]
    serve_argv: list[str]
    port: int
    mode: str


@dataclass
class DockerRuntime:
    cfg: Config
    commands: list[list[str]] = field(default_factory=list)

    def ensure_image(self) -> str | None:
        inspect = docker_inspect_argv(self.cfg)
        self.commands.append(inspect)
        looked = run_command(inspect, timeout_s=self.cfg.timeouts.docker_s, what="docker image inspect")
        if not looked.ok:
            pull = docker_pull_argv(self.cfg)

            def _pull() -> None:
                self.commands.append(pull)
                require_ok(
                    run_command(pull, timeout_s=self.cfg.timeouts.docker_s, what="docker pull"),
                    what="docker pull",
                )

            call_with_retry(
                _pull,
                attempts=self.cfg.retries.attempts,
                backoff_s=self.cfg.retries.backoff_s,
                factor=self.cfg.retries.backoff_factor,
                what="docker pull",
            )
            looked = require_ok(
                run_command(inspect, timeout_s=self.cfg.timeouts.docker_s, what="docker image inspect"),
                what="docker image inspect",
            )
        return _digest_from_inspect(looked.stdout)

    def start(self, spec: StartSpec) -> None:
        self.commands.append(spec.docker_argv)
        log_event("docker_run", name=spec.name, mode=spec.mode)
        require_ok(
            run_command(spec.docker_argv, timeout_s=self.cfg.timeouts.docker_s, what="docker run"),
            what=f"docker run {spec.name}",
        )
        state = run_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", spec.name],
            timeout_s=self.cfg.timeouts.command_s,
            what="docker inspect state",
        )
        if state.ok and state.stdout.strip().lower() == "false":
            logs = self.logs(spec.name)
            self.stop(spec.name)
            raise SmaError(f"Container {spec.name} exited during start. Logs: {logs[-800:]}")

    def stop(self, name: str) -> None:
        stop = docker_stop_argv(name, self.cfg.engine.server_stop_timeout_s)
        remove = docker_rm_argv(name)
        self.commands.append(stop)
        self.commands.append(remove)
        run_command(stop, timeout_s=self.cfg.engine.server_stop_timeout_s + 15, what="docker stop")
        run_command(remove, timeout_s=self.cfg.timeouts.command_s, what="docker rm")

    def logs(self, name: str) -> str:
        argv = docker_logs_argv(name)
        self.commands.append(argv)
        result = run_command(argv, timeout_s=self.cfg.timeouts.command_s, what="docker logs")
        text = scrub_text((result.stdout or "") + (result.stderr or ""))
        if len(text) > 2_000_000:
            return text[-2_000_000:]
        return text

    def is_running(self, name: str) -> bool:
        result = run_command(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            timeout_s=self.cfg.timeouts.command_s,
            what="docker inspect",
        )
        return result.ok and result.stdout.strip().lower() == "true"

    def list_sma(self) -> list[str]:
        result = run_command(
            ["docker", "ps", "-a", "--filter", f"name={self.cfg.engine.container_prefix}-", "--format", "{{.Names}}"],
            timeout_s=self.cfg.timeouts.command_s,
            what="docker ps",
        )
        if not result.ok:
            raise SmaError(f"docker ps failed: {result.stderr[-400:]}")
        prefix = self.cfg.engine.container_prefix + "-"
        return [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(prefix)]


def _digest_from_inspect(stdout: str) -> str | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and "@sha256:" in item:
                return item.split("@", 1)[1]
        return str(parsed[0]) if parsed else None
    if isinstance(parsed, str) and "@sha256:" in parsed:
        return parsed.split("@", 1)[1]
    return None


def save_container_logs(runtime: DockerRuntime, name: str, path: Path) -> None:
    from sma.atomic import atomic_write_text

    try:
        text = runtime.logs(name)
    except SmaError as exc:
        text = f"could not read logs: {exc}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text if text.endswith("\n") or text == "" else text + "\n")
