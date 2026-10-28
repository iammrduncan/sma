"""SWE-bench Verified-mini on the x86_64 runner: mini-swe-agent writes the patches, the harness grades them.

SWE-bench's task containers are x86_64, so they run on a separate Docker host (runs/cpu/baseline/
sweverified/README.md). SMA reaches it over SSH with the names in the root .env. While the agents run, an
SSH reverse forward puts the model server's port on the runner's loopback. The agent and the harness run
there detached, so a dropped SSH session does not stop them; SMA polls for their exit files.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

from sma.errors import SmaError
from sma.proc import Following, follow_to_file, require_ok, run_command
from sma.qualityout import wilson_interval

HOST_ENV, USER_ENV, KEY_ENV = "SWE_RUNNER_HOST", "SWE_RUNNER_USER", "SWE_RUNNER_SSH_KEY"
SUITE = "swebench_verified_mini"
SCORER = "SWE-bench harness 5.0.2 on mini-swe-agent 2.4.6 patches (swebench.yaml, bash tool calls)"

# mini-swe-agent's exit statuses that are the model's own result. Any other status is an exception's
# class name (the model server unreachable, a container that did not start): that task got no answer.
ANSWERED = frozenset(
    {"Submitted", "LimitsExceeded", "TimeExceeded", "FormatError", "RepeatedFormatError", "ContextWindowExceededError"}
)


def run_rel(run_id: str) -> str:
    """The run's folder on the runner, relative to swebench.remote_dir."""
    return f"runs/{run_id}"


def agent_overlay(*, served_name: str, port: int, step_limit: int, enable_thinking: bool | None) -> dict[str, Any]:
    """SMA's settings on top of mini-swe-agent's swebench.yaml: the model, its server, the step limit.

    No temperature or other sampling field is sent, so the server applies the model's own
    generation_config. enable_thinking restores the chat template's default (sma.thinking).
    """
    kwargs: dict[str, Any] = {"api_base": f"http://127.0.0.1:{port}/v1"}
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return {
        "agent": {"step_limit": step_limit},
        "model": {"model_name": f"hosted_vllm/{served_name}", "model_kwargs": kwargs},
    }


def agent_command(cfg: Any, run_id: str) -> str:
    sw = cfg.swebench
    rel = run_rel(run_id)
    return (
        f".venv/bin/mini-extra swebench --subset {shlex.quote(sw.subset)} --split {shlex.quote(sw.split)} "
        f"--output {rel} --workers {int(sw.workers)} -c {shlex.quote(sw.agent_config)} -c {rel}/sma-agent.yaml"
    )


def grade_command(cfg: Any, run_id: str) -> str:
    sw = cfg.swebench
    rel = run_rel(run_id)
    return (
        f"cd eval && ../.venv/bin/python -m swebench.harness.run_evaluation "
        f"--dataset_name ../{shlex.quote(sw.dataset_file)} --split {shlex.quote(sw.split)} "
        f"--predictions_path ../{rel}/preds.json --run_id {shlex.quote(run_id)} "
        f"--max_workers {int(sw.grade_workers)} --timeout {int(sw.grade_timeout_s)} "
        f"&& cp -- *.{shlex.quote(run_id)}.json ../{rel}/report.json"
    )


# Drops the tasks that got no answer from preds.json, so the next launch runs them again.
_PRUNE = """
import glob, json, os, sys
answered = set(sys.argv[1].split(","))
path = "preds.json"
if os.path.exists(path):
    preds = json.load(open(path))
    for traj in glob.glob("*/*.traj.json"):
        status = (json.load(open(traj)).get("info") or {}).get("exit_status")
        task = os.path.basename(os.path.dirname(traj))
        if status not in answered and task in preds:
            del preds[task]
            os.remove(traj)
    json.dump(preds, open(path, "w"), indent=2)
"""

# One line per task: its exit status and how many model calls it made. The trajectories stay on the runner.
_STATUSES = """
import glob, json, os
for traj in sorted(glob.glob("*/*.traj.json")):
    info = json.load(open(traj)).get("info") or {}
    stats = info.get("model_stats") or {}
    print(json.dumps({"instance_id": os.path.basename(os.path.dirname(traj)), "exit_status": info.get("exit_status"),
                      "api_calls": stats.get("api_calls")}))
"""

_TRAJECTORIES = """
import glob, json, sys
for traj in sorted(glob.glob("*/*.traj.json")):
    sys.stdout.write(json.dumps(json.load(open(traj))) + "\\n")
"""


class SshRunner:
    """The SWE-bench runner over SSH. Paths are relative to remote_dir in the SSH user's home."""

    def __init__(self, *, host: str, user: str, key: str, remote_dir: str) -> None:
        self.target = f"{user}@{host}"
        self.key = str(Path(key).expanduser())
        self.remote_dir = remote_dir

    def _ssh(self) -> list[str]:
        return [
            "ssh", "-i", self.key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
        ]  # fmt: skip

    def _run(self, script: str, *, what: str, timeout_s: float = 300.0, ok_codes: tuple[int, ...] = (0,)) -> Any:
        argv = [*self._ssh(), self.target, f"cd {shlex.quote(self.remote_dir)} && {script}"]
        result = run_command(argv, timeout_s=timeout_s, what=what)
        if result.returncode not in ok_codes:
            require_ok(result, what=what)
        return result

    def check(self, cfg: Any) -> None:
        sw = cfg.swebench
        script = (
            f"test -x .venv/bin/mini-extra && test -f {shlex.quote(sw.dataset_file)} && test -d {shlex.quote(sw.subset)} "
            "&& test -d eval && docker info --format '{{.ServerVersion}}' >/dev/null"
        )
        self._run(script, what="SWE-bench runner check (venv, dataset, eval folder, Docker)", timeout_s=60)

    def write_text(self, rel: str, text: str) -> None:
        if "\nSMA_EOF\n" in f"\n{text}\n":
            raise SmaError("The text to write on the SWE-bench runner contains SMA's heredoc marker.")
        folder = shlex.quote(str(Path(rel).parent))
        self._run(f"mkdir -p {folder} && cat > {shlex.quote(rel)} <<'SMA_EOF'\n{text}\nSMA_EOF", what=f"write {rel} on the SWE-bench runner")

    def start(self, rel_dir: str, name: str, command: str) -> None:
        """Start command detached from remote_dir; it writes <name>.log and, when it ends, <name>.exit."""
        log, done, pid = (shlex.quote(f"{rel_dir}/{name}.{ext}") for ext in ("log", "exit", "pid"))
        inner = f"{command} > {log} 2>&1; echo $? > {done}"
        script = f"mkdir -p {shlex.quote(rel_dir)} && rm -f {done} && (nohup sh -c {shlex.quote(inner)} > /dev/null 2>&1 & echo $! > {pid})"
        self._run(script, what=f"start {name} on the SWE-bench runner")

    def status(self, rel_dir: str, name: str) -> tuple[int | None, bool]:
        """(exit code once it has ended, whether its process is alive)."""
        done, pid = (shlex.quote(f"{rel_dir}/{name}.{ext}") for ext in ("exit", "pid"))
        script = (
            f"if [ -f {done} ]; then echo exit $(cat {done}); "
            f"elif [ -f {pid} ] && kill -0 $(cat {pid}) 2>/dev/null; then echo alive; else echo none; fi"
        )
        words = self._run(script, what=f"check {name} on the SWE-bench runner", timeout_s=60).stdout.split()
        if words[:1] == ["exit"]:
            return int(words[1]) if len(words) > 1 and words[1].lstrip("-").isdigit() else -1, False
        return None, words[:1] == ["alive"]

    def stop(self, run_id: str) -> None:
        """End every process and container of this run on the runner."""
        # "[r]uns/<id>" matches the agent's and the harness's command lines but not this shell's own.
        pattern = shlex.quote(f"[r]uns/{run_id}")
        token = shlex.quote(run_id)
        self._run(
            f"pkill -TERM -f -- {pattern}; docker ps -q --filter name={token} | xargs -r docker rm -f >/dev/null; true",
            what="stop SWE-bench processes on the runner",
            timeout_s=120,
        )

    def python(self, rel_dir: str, code: str, *args: str, what: str, timeout_s: float = 300.0) -> str:
        quoted = " ".join(shlex.quote(arg) for arg in args)
        script = f"cd {shlex.quote(rel_dir)} && python3 -c {shlex.quote(code)} {quoted}"
        return self._run(script, what=what, timeout_s=timeout_s).stdout

    def read_text(self, rel: str) -> str | None:
        result = self._run(f"if [ -f {shlex.quote(rel)} ]; then cat {shlex.quote(rel)}; else exit 3; fi", what=f"read {rel} on the SWE-bench runner", ok_codes=(0, 3))
        return None if result.returncode == 3 else result.stdout

    def tunnel(self, port: int, log_path: Path) -> Following:
        """Put the model server's port on the runner's loopback for as long as the agents run."""
        argv = [*self._ssh(), "-o", "ExitOnForwardFailure=yes", "-N", "-R", f"127.0.0.1:{port}:127.0.0.1:{port}", self.target]
        return follow_to_file(argv, log_path, what="SWE-bench tunnel")


def runner_from_env(cfg: Any) -> SshRunner | None:
    names = (HOST_ENV, USER_ENV, KEY_ENV)
    if not all(os.environ.get(name) for name in names):
        return None
    return SshRunner(host=os.environ[HOST_ENV], user=os.environ[USER_ENV], key=os.environ[KEY_ENV], remote_dir=cfg.swebench.remote_dir)


def prune_unanswered(runner: Any, run_id: str) -> None:
    runner.python(run_rel(run_id), _PRUNE, ",".join(sorted(ANSWERED)), what="drop unanswered SWE-bench tasks")


def task_statuses(runner: Any, run_id: str) -> list[dict[str, Any]]:
    out = runner.python(run_rel(run_id), _STATUSES, what="read SWE-bench exit statuses")
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def trajectories(runner: Any, run_id: str) -> list[str]:
    out = runner.python(run_rel(run_id), _TRAJECTORIES, what="read SWE-bench trajectories", timeout_s=900)
    return [line for line in out.splitlines() if line.strip()]


def unanswered(statuses: list[dict[str, Any]], total: int) -> dict[str, int]:
    """Exit statuses that are not the model's own result, with counts. Missing tasks count as 'no trajectory'."""
    counts: dict[str, int] = {}
    for row in statuses:
        status = str(row.get("exit_status"))
        if status not in ANSWERED:
            counts[status] = counts.get(status, 0) + 1
    missing = total - len(statuses)
    if missing > 0:
        counts["no trajectory"] = missing
    return counts


def dataset_size(cfg: Any, runner: Any) -> int:
    text = runner.read_text(cfg.swebench.dataset_file)
    if text is None:
        raise SmaError(f"The SWE-bench runner has no {cfg.swebench.dataset_file}.")
    return sum(1 for line in text.splitlines() if line.strip())


def score(report: dict[str, Any] | None, statuses: list[dict[str, Any]], total: int) -> dict[str, Any]:
    """The suite record: resolved over every task, with a Wilson interval. Null when a task got no answer."""
    missing = unanswered(statuses, total)
    exit_counts: dict[str, int] = {}
    for row in statuses:
        exit_counts[str(row.get("exit_status"))] = exit_counts.get(str(row.get("exit_status")), 0) + 1
    resolved = len((report or {}).get("resolved_ids") or [])
    payload: dict[str, Any] = {
        "suite": SUITE,
        "scorer": SCORER,
        "primary_metric": "resolved",
        "primary_score": None,
        "ci_lower": None,
        "ci_upper": None,
        "n": total,
        "limit": None,
        "resolved": resolved if report is not None else None,
        "exit_statuses": exit_counts,
        "unanswered": missing,
        "harness": {key: (report or {}).get(key) for key in ("total_instances", "submitted_instances", "completed_instances",
                                                            "resolved_instances", "unresolved_instances", "empty_patch_instances",
                                                            "error_instances")},
        "error": None,
    }
    if missing:
        payload["error"] = (
            f"{sum(missing.values())} of {total} tasks got no answer from the model ({', '.join(f'{k} {v}' for k, v in sorted(missing.items()))}). "
            "The score is null. Resume the run to retry those tasks."
        )
    elif report is None:
        payload["error"] = "The harness wrote no report. The score is null."
    else:
        lower, upper = wilson_interval(resolved, total)
        payload.update(primary_score=resolved / total if total else None, ci_lower=lower, ci_upper=upper)
    return payload


def overlay_yaml(overlay: dict[str, Any]) -> str:
    return yaml.safe_dump(overlay, sort_keys=False)
