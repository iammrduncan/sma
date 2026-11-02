"""Serving on another machine: the lab's V100 pod (services/v100-serving).

The harness, every benchmark client, grading, and the records stay on this machine. llama-server and
llama-perplexity run on the serving machine as processes started over SSH, from the pinned source build
there (it has no Docker), and an SSH local forward puts the server on this machine's engine.port, so every
client reaches it at base_url as it would a local server. The model files are downloaded there from the
same pinned revision and checked against the hashes of this machine's copy.
"""

from __future__ import annotations

import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sma.config import Config
from sma.errors import SmaError
from sma.logsetup import log_event
from sma.proc import Following, follow_to_file
from sma.records import model_slug
from sma.scrub import scrub_text
from sma.swebench import SshRunner, start_script

# Under remote.root: one folder per server (its log, pid, and exit status), and the model files by revision.
SERVERS = "servers"
MODELS = "models"
_LOG_TAIL_BYTES = 2_000_000


def remote_runner(cfg: Config) -> SshRunner:
    """The serving machine named in the root .env."""
    rc = cfg.remote
    names = (rc.host_env, rc.user_env, rc.port_env, rc.key_env)
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise SmaError(
            f"remote.enabled needs {', '.join(missing)} in the root .env (services/v100-serving/README.md)."
        )
    return SshRunner(
        host=os.environ[rc.host_env],
        user=os.environ[rc.user_env],
        key=os.environ[rc.key_env],
        remote_dir=rc.root,
        label="serving machine",
        port=int(os.environ[rc.port_env]),
        # One shared connection per slot: sshd allows 10 sessions on one connection, and each run holds a log
        # follower and asks nvidia-smi every second. The dispatcher has its own.
        control_dir=Path("~/.local/share/sma/ssh").expanduser() / (rc.slot or "shared"),
    )


def remote_serve_args(serve_argv: list[str], *, model_dir: str, port: int) -> list[str]:
    """llama-server's flags as SMA builds them for its container, for the serving machine: the model from
    model_dir instead of the /model mount, and the server on its own loopback at port (the forward's end)."""
    args = list(serve_argv)
    for index, value in enumerate(args[:-1]):
        following = args[index + 1]
        if value == "-m" and following.startswith("/model/"):
            args[index + 1] = f"{model_dir}/{following.removeprefix('/model/')}"
        elif value == "--host":
            args[index + 1] = "127.0.0.1"
        elif value == "--port":
            args[index + 1] = str(port)
    return args


def thread_args(cfg: Config) -> list[str]:
    """llama.cpp's thread flags on the serving machine (remote.threads_per_slot): the pod's 12 CPUs shared by its
    slots, where llama.cpp would size its threads from the 40 CPUs the pod shows."""
    count = cfg.remote.threads_per_slot
    return ["--threads", str(count), "--threads-batch", str(count)] if count else []


class RemoteRuntime:
    """DockerRuntime's interface for servers on the serving machine. A name is SMA's container name; each
    server has a folder <root>/servers/<name> with its log, the wrapper's pid and exit status, and the
    server's own pid (llama.pid)."""

    def __init__(self, cfg: Config, runner: Any = None, local: Any = None) -> None:
        from sma.runtime import DockerRuntime

        self.cfg = cfg
        self.runner = runner or remote_runner(cfg)
        # Everything but the model servers stays here in Docker: the eval-factory suites, the speed client.
        self.local = local or DockerRuntime(cfg)
        self.forwards: dict[str, Following] = {}
        self.commands: list[list[str]] = []

    @staticmethod
    def served(name: str) -> bool:
        """Whether a name is a model server, which runs on the serving machine (sma-<run>-quality or -speed)."""
        return name.endswith(("-quality", "-speed"))

    # The build and the model files.

    def bin(self, name: str) -> str:
        return f"{self.cfg.remote.root}/{self.cfg.remote.llamacpp_build}/build/bin/{name}"

    def build(self) -> dict[str, Any]:
        """The pinned build's record (services/v100-serving/setup.sh writes it)."""
        text = self.runner.read_text(f"{self.cfg.remote.llamacpp_build}/build.json")
        if text is None:
            raise SmaError(
                f"The serving machine has no {self.cfg.remote.llamacpp_build}/build.json under {self.cfg.remote.root}. "
                "Build llama.cpp there with services/v100-serving/setup.sh."
            )
        return json.loads(text)

    def ensure_image(self, image: str | None = None) -> str | None:
        """For llama.cpp's server image, the serving machine's build: its commit stands in for a digest. Any
        other image (an eval-factory suite, the speed client) is pulled here, as without remote serving."""
        server = self.cfg.engine.llamacpp_image.split("@", 1)[0]
        if image is None or image.split("@", 1)[0] == server:
            return str(self.build()["commit"])
        return self.local.ensure_image(image)

    def ensure_model(self, manifest: dict[str, Any]) -> str:
        """Download the run's GGUF files on the serving machine from its pinned revision, once, and check
        each against this machine's sha256. Returns the folder that holds them there."""
        repo, revision = str(manifest["model_repo"]), str(manifest.get("revision_sha") or "")
        files = list((manifest.get("gguf") or {}).get("files") or [])
        hashes = manifest.get("file_hashes") or {}
        if not revision or not files:
            raise SmaError("The run has no pinned revision or GGUF file list, so SMA cannot copy its model to the serving machine.")
        folder = f"{MODELS}/{model_slug(repo)}@{revision[:12]}"
        for name in files:
            want = str(hashes.get(name) or "").removeprefix("sha256:")
            if not want:
                raise SmaError(f"{name} has no sha256 in the manifest, so SMA cannot check its copy on the serving machine.")
            path, url = f"{folder}/{name}", f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            quoted, marker = shlex.quote(path), shlex.quote(f"{path}.sha256")
            script = (
                f"mkdir -p {shlex.quote(folder)} && "
                f"if [ -f {quoted} ] && [ \"$(cat {marker} 2>/dev/null)\" = {want} ]; then echo present; exit 0; fi; "
                f"curl -fsSL --retry 3 -o {quoted}.part {shlex.quote(url)} && "
                f"got=$(sha256sum {quoted}.part | cut -d' ' -f1) && "
                f"if [ \"$got\" != {want} ]; then rm -f {quoted}.part; echo \"sha256 $got\"; exit 4; fi && "
                f"mv {quoted}.part {quoted} && echo {want} > {marker} && echo downloaded"
            )
            log_event("remote_model", file=name, repo=repo)
            out = self.runner.shell(script, what=f"copy {name} to the serving machine", timeout_s=float(self.cfg.hf.download_timeout_s))
            log_event("remote_model_ready", file=name, state=out.strip()[-40:])
        return f"{self.cfg.remote.root}/{folder}"

    # Servers.

    def _dir(self, name: str) -> str:
        return f"{SERVERS}/{name}"

    def start(self, spec: Any) -> None:
        model_dir = getattr(spec, "remote_model_dir", None)
        if not model_dir:
            raise SmaError("The server's model folder on the serving machine is not known. ensure_model runs first.")
        args = remote_serve_args(spec.serve_argv, model_dir=model_dir, port=self.cfg.remote.remote_port)
        env = {"CUDA_VISIBLE_DEVICES": self.cfg.remote.gpus, **spec.env}
        folder = self._dir(spec.name)
        # sh writes its own pid, then becomes env, then llama-server: llama.pid is the server's pid.
        exec_line = " ".join(["exec", "env", *(f"{key}={shlex.quote(value)}" for key, value in env.items()),
                              shlex.quote(self.bin("llama-server")), *(shlex.quote(arg) for arg in args)])
        command = f"sh -c {shlex.quote(f'echo $$ > {folder}/llama.pid; {exec_line}')}"
        self.commands.append(["remote", "start", spec.name, *args])
        log_event("remote_start", name=spec.name, mode=spec.mode, gpus=self.cfg.remote.gpus)
        # The folder names the server's GPUs, so a run on one slot can tell another slot's server from a
        # leftover on its own GPU (servers_on).
        self.runner.shell(
            f"rm -rf {shlex.quote(folder)} && mkdir -p {shlex.quote(folder)} && echo {shlex.quote(self.cfg.remote.gpus)} > {shlex.quote(folder)}/gpus",
            what="clear the server folder",
        )
        self.runner.start(folder, "server", command)
        self._forward(spec.name, int(spec.port))
        code, alive = self.runner.status(folder, "server")
        if code is not None and not alive:
            text = self.logs(spec.name)
            self.stop(spec.name)
            raise SmaError(f"{spec.name} exited during start on the serving machine. Log: {text[-800:]}")

    def _forward(self, name: str, local_port: int) -> None:
        for other in list(self.forwards):
            self.forwards.pop(other).stop(timeout_s=0.1)
        log = Path("~/.local/share/sma/ssh").expanduser() / f"forward-{name}.log"
        self.forwards[name] = self.runner.forward(local_port, self.cfg.remote.remote_port, log)

    def _signal(self, name: str, sig: str, wait_s: float) -> None:
        folder = shlex.quote(self._dir(name))
        script = (
            f"if [ -f {folder}/llama.pid ]; then kill -{sig} $(cat {folder}/llama.pid) 2>/dev/null; "
            f"for i in $(seq 1 {max(1, int(wait_s))}); do [ -f {folder}/server.exit ] && break; sleep 1; done; "
            f"kill -KILL $(cat {folder}/llama.pid) 2>/dev/null; fi; true"
        )
        self.runner.shell(script, what=f"stop {name} on the serving machine", timeout_s=wait_s + 60)

    def stop(self, name: str) -> None:
        if not self.served(name):
            self.local.stop(name)
            return
        self._signal(name, "TERM", float(self.cfg.engine.server_stop_timeout_s))
        following = self.forwards.pop(name, None)
        if following is not None:
            following.stop(timeout_s=0.1)

    def kill(self, name: str) -> None:
        if not self.served(name):
            self.local.kill(name)
            return
        self._signal(name, "KILL", 5)

    def logs(self, name: str) -> str:
        if not self.served(name):
            return self.local.logs(name)
        text = self.runner.shell(
            f"tail -c {_LOG_TAIL_BYTES} {shlex.quote(self._dir(name))}/server.log 2>/dev/null; true",
            what=f"read {name}'s log on the serving machine",
        )
        return scrub_text(text)

    def follow_logs(self, name: str, path: Path) -> Following:
        """Copy the server's log here while it runs. tail ends when the follower is stopped."""
        argv = self.runner.command(f"tail -n +1 -F {shlex.quote(self._dir(name))}/server.log 2>/dev/null")
        return follow_to_file(argv, path, what=f"follow {name}'s log")

    def exit_code(self, name: str) -> int | None:
        if not self.served(name):
            return self.local.exit_code(name)
        code, _alive = self.runner.status(self._dir(name), "server")
        return code

    def is_running(self, name: str) -> bool:
        if not self.served(name):
            return self.local.is_running(name)
        folder = shlex.quote(self._dir(name))
        out = self.runner.shell(
            f"if [ -f {folder}/llama.pid ] && kill -0 $(cat {folder}/llama.pid) 2>/dev/null; then echo alive; else echo gone; fi",
            what=f"check {name} on the serving machine", timeout_s=60,
        )
        return out.strip() == "alive"

    def started_at(self, name: str) -> datetime | None:
        out = self.runner.shell(f"stat -c %Y {shlex.quote(self._dir(name))}/llama.pid 2>/dev/null; true", what="read a server's start", timeout_s=60)
        return datetime.fromtimestamp(int(out.strip()), timezone.utc) if out.strip().isdigit() else None

    def list_sma(self) -> list[str]:
        """The servers alive on the serving machine, and SMA's containers here, by name."""
        out = self.runner.shell(
            f"for d in {SERVERS}/*/; do [ -f \"$d/llama.pid\" ] && kill -0 $(cat \"$d/llama.pid\") 2>/dev/null && basename \"$d\"; done; true",
            what="list servers on the serving machine", timeout_s=60,
        )
        return [line.strip() for line in out.splitlines() if line.strip()] + list(self.local.list_sma())

    def servers_on(self) -> dict[str, str]:
        """The servers alive on the serving machine, by name, with the GPUs each runs on ("" when not recorded)."""
        out = self.runner.shell(
            f"for d in {SERVERS}/*/; do [ -f \"$d/llama.pid\" ] && kill -0 $(cat \"$d/llama.pid\") 2>/dev/null "
            "&& echo \"$(basename \"$d\") $(cat \"$d/gpus\" 2>/dev/null)\"; done; true",
            what="list servers and their GPUs on the serving machine", timeout_s=60,
        )
        found: dict[str, str] = {}
        for line in out.splitlines():
            name, _, gpus = line.strip().partition(" ")
            if name:
                found[name] = gpus.strip()
        return found

    def gguf_architectures(self, image: str) -> set[str] | None:
        """Read from this machine's llama.cpp image: the same release as the serving machine's build."""
        return self.local.gguf_architectures(image)

    # llama-perplexity, telemetry.

    def perplexity_command(self, args: list[str]) -> list[str]:
        """The argv that runs llama-perplexity on the serving machine with args (paths already remote)."""
        env = f"CUDA_VISIBLE_DEVICES={shlex.quote(self.cfg.remote.gpus)}"
        args = [*args, *thread_args(self.cfg)]
        return self.runner.command(" ".join([env, shlex.quote(self.bin("llama-perplexity")), *(shlex.quote(a) for a in args)]))

    def telemetry_prefix(self) -> list[str]:
        """Prefix for a local command to run on the serving machine instead (nvidia-smi), over the shared connection."""
        return self.runner.prefix()

    def facts(self) -> dict[str, Any]:
        """The serving hardware, by kind and chip, never by name: GPU model, driver, the GPUs this worker uses."""
        out = self.runner.shell(
            f"nvidia-smi -i {shlex.quote(self.cfg.remote.gpus)} --query-gpu=name,memory.total,driver_version --format=csv,noheader",
            what="read the serving GPUs", timeout_s=60,
        )
        rows = [[part.strip() for part in line.split(",")] for line in out.splitlines() if line.strip()]
        return {
            "hardware": self.cfg.remote.hardware,
            # The slot (sma worker --slots) is where this run's server ran; runs on any slot are one subject.
            "slot": self.cfg.remote.slot,
            "gpus": self.cfg.remote.gpus,
            "gpu_names": [row[0] for row in rows],
            "gpu_memory": [row[1] for row in rows if len(row) > 1],
            "driver": rows[0][2] if rows and len(rows[0]) > 2 else None,
            "build": self.build(),
        }
