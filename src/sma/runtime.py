"""Docker runtime for the NGC vLLM container. Tests use a fake with the same methods."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from sma.proc import Following, follow_to_file, require_ok, run_command
from sma.retry import call_with_retry
from sma.scrub import scrub_text


_GGUF_ARCHITECTURES = (
    "from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES as names\n"
    "print(' '.join(sorted(names)))"
)


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

    def ensure_image(self, image: str | None = None) -> str | None:
        from sma.commands import image_ref

        ref = image or image_ref(self.cfg)
        inspect = ["docker", "image", "inspect", ref, "--format", "{{json .RepoDigests}}"]
        self.commands.append(inspect)
        looked = run_command(inspect, timeout_s=self.cfg.timeouts.docker_s, what="docker image inspect")
        if not looked.ok:
            pull = ["docker", "pull", ref]

            def _pull() -> None:
                self.commands.append(pull)
                # A cold pull of a multi-GB image outlasts the short docker command timeout.
                require_ok(
                    run_command(pull, timeout_s=self.cfg.timeouts.image_pull_s, what="docker pull"),
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

    def kill(self, name: str) -> None:
        """SIGKILL the container now (docker kill). It stays, so its exit code and logs can still be read."""
        argv = ["docker", "kill", name]
        self.commands.append(argv)
        run_command(argv, timeout_s=self.cfg.timeouts.command_s, what="docker kill")

    def logs(self, name: str) -> str:
        argv = docker_logs_argv(name)
        self.commands.append(argv)
        result = run_command(argv, timeout_s=self.cfg.timeouts.command_s, what="docker logs")
        text = scrub_text((result.stdout or "") + (result.stderr or ""))
        if len(text) > 2_000_000:
            return text[-2_000_000:]
        return text

    def follow_logs(self, name: str, path: Path) -> Following:
        """Copy the container's output into path while it runs. Ends when the container stops."""
        argv = docker_logs_argv(name, follow=True)
        self.commands.append(argv)
        return follow_to_file(argv, path, what="docker logs --follow")

    def exit_code(self, name: str) -> int | None:
        result = run_command(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", name],
            timeout_s=self.cfg.timeouts.command_s,
            what="docker inspect",
        )
        try:
            return int(result.stdout.strip()) if result.ok else None
        except ValueError:
            return None

    def gguf_architectures(self, image: str) -> set[str] | None:
        """GGUF architectures whose tokenizer the image's transformers reads: vLLM's bench client loads a
        GGUF's tokenizer through it. None when the image does not say."""
        result = run_command(
            ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3", image, "-c", _GGUF_ARCHITECTURES],
            timeout_s=self.cfg.timeouts.docker_s,
            what="transformers GGUF architectures",
        )
        lines = result.stdout.strip().splitlines() if result.ok else []
        return set(lines[-1].split()) if lines else None

    def started_at(self, name: str) -> datetime | None:
        """When the container last started (docker's State.StartedAt), or None when docker does not say."""
        result = run_command(
            ["docker", "inspect", "-f", "{{.State.StartedAt}}", name],
            timeout_s=self.cfg.timeouts.command_s,
            what="docker inspect",
        )
        match = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", result.stdout.strip()) if result.ok else None
        if not match:
            return None
        fraction = (match.group(2) or ".0")[:7]  # Python takes microseconds; docker gives nanoseconds.
        return datetime.fromisoformat(match.group(1) + fraction).replace(tzinfo=timezone.utc)

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
