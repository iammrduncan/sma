"""What each machine SMA uses is, for `sma watch`: this Spark (the model), the KernelBench service, the
x86_64 runner (SWE-bench and BrowseComp-Plus's search), and the judge. Kinds and chips, never names.

Each is asked once with a short timeout. A machine that is not configured in .env says so; one that
does not answer is down. Nothing is filled in that the machine did not report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from sma.config import Config
from sma.errors import SmaError
from sma.httpapi import request_json
from sma.proc import run_command

# Read-only. One "key=value" line per fact; cpu may repeat (the GB10 has two core types).
FACTS_SCRIPT = (
    'echo "arch=$(uname -m)"; '
    'echo "product=$(cat /sys/class/dmi/id/product_name 2>/dev/null)"; '
    'echo "vendor=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null)"; '
    "lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | sed 's/^/cpu=/'; "
    'echo "cores=$(nproc)"; '
    "awk '/^MemTotal:/ {print \"mem_kib=\" $2}' /proc/meminfo; "
    "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/gpu=/'"
)
_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Machine:
    """One machine's role, whether it answered (None: not configured), and one line of what it is: kind
    and chips first, then what else it reports. Empty: it reported nothing."""

    role: str
    up: bool | None
    summary: str = ""


def parse_facts(text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {"cpu": [], "gpu": []}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        value = value.strip()
        if not value:
            continue
        if key in ("cpu", "gpu"):
            if value not in facts[key]:
                facts[key].append(value)
        elif key in ("arch", "product", "vendor", "cores", "mem_kib"):
            facts[key] = value
    return facts


def describe_host(role: str, facts: dict[str, Any]) -> Machine:
    """Kind, GPU, CPU cores, memory: "DGX Spark · GB10 GPU · 20× Cortex-X925/A725 · 121.7 GiB"."""
    product = str(facts.get("product") or "").replace("_", " ").strip()
    vendor = str(facts.get("vendor") or "").strip()
    virtual = vendor == "QEMU" or product.startswith("Standard PC")
    kind = "QEMU VM" if virtual else _short(product) or vendor
    gpus = ", ".join(_short(name) for name in facts.get("gpu") or [])
    arch = str(facts.get("arch") or "")
    cpu = f"{arch} vCPU" if virtual else _cpus(facts.get("cpu") or []) or arch
    parts = [kind, f"{gpus} GPU" if gpus else "no GPU", f"{facts['cores']}× {cpu}" if facts.get("cores") else cpu]
    if str(facts.get("mem_kib") or "").isdigit():
        parts.append(f"{int(facts['mem_kib']) / 2**20:.1f} GiB")
    return Machine(role, True, " · ".join(part for part in parts if part))


def _short(name: str) -> str:
    return name.strip().removeprefix("NVIDIA ").strip()


def _cpus(names: list[str]) -> str:
    """Core types of one family as one name: Cortex-X925 and Cortex-A725 as Cortex-X925/A725."""
    families = {name.split("-", 1)[0] for name in names}
    if len(names) > 1 and len(families) == 1 and all("-" in name for name in names):
        return families.pop() + "-" + "/".join(name.split("-", 1)[1] for name in names)
    return " + ".join(names)


def local_machine() -> Machine:
    result = run_command(["sh", "-c", FACTS_SCRIPT], timeout_s=_TIMEOUT_S, what="machine facts")
    return describe_host("model", parse_facts(result.stdout)) if result.ok else Machine("model", True)


def kernelbench_machine() -> Machine:
    from sma.kbservice import URL_ENV

    url = os.environ.get(URL_ENV)
    if not url:
        return Machine("kernel", None)
    try:
        env = request_json("GET", url.rstrip("/") + "/v1/environment", timeout_s=_TIMEOUT_S, attempts=1)
    except SmaError:
        return Machine("kernel", False)
    env = env if isinstance(env, dict) else {}
    gpu = env.get("gpu") if isinstance(env.get("gpu"), dict) else {}
    chip = f"{_short(gpu['name'])} GPU" if gpu.get("name") else ""
    if chip and gpu.get("compute_capability"):
        chip += f" (sm {gpu['compute_capability']})"
    parts = [
        chip,
        f"CUDA {env['cuda_runtime']}" if env.get("cuda_runtime") else "",
        f"torch {str(env['torch']).split('+')[0]}" if env.get("torch") else "",
        # The service's API names its GPU and software, not the machine.
        "type not reported",
    ]
    return Machine("kernel", True, " · ".join(part for part in parts if part))


def runner_machine(cfg: Config) -> Machine:
    from sma.swebench import runner_from_env

    runner = runner_from_env(cfg, remote_dir=".", label="x86_64 runner")
    if runner is None:
        return Machine("swe", None)
    try:
        return describe_host("swe", parse_facts(runner.machine_facts(FACTS_SCRIPT, timeout_s=_TIMEOUT_S + 10)))
    except SmaError:
        return Machine("swe", False)


def judge_machine() -> Machine:
    from sma.judge import KEY_ENV, URL_ENV

    url, key = os.environ.get(URL_ENV), os.environ.get(KEY_ENV)
    if not url or not key:
        return Machine("judge", None)
    base = url.rstrip("/")
    base = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        reply = request_json(
            "GET", base + "/v1/models", timeout_s=_TIMEOUT_S, attempts=1, headers={"Authorization": f"Bearer {key}"}
        )
    except SmaError:
        return Machine("judge", False)
    models = [str(item.get("id")) for item in (reply.get("data") or []) if isinstance(item, dict)] if isinstance(reply, dict) else []
    serves = f"serves {', '.join(models)} · " if models else ""
    return Machine("judge", True, serves + "hardware not reported")


def gather(cfg: Config) -> list[Machine]:
    return [local_machine(), kernelbench_machine(), runner_machine(cfg), judge_machine()]
