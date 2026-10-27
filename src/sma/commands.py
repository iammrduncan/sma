"""The exact argv SMA prints in a dry-run and executes for real."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sma.config import Config, QualitySuite
from sma.errors import SmaError


def image_ref(cfg: Config, digest: str | None = None, *, engine: str | None = None) -> str:
    from sma.engines import configured_digest, image_for, image_ref as resolved

    name = engine or cfg.engine.name
    image = image_for(cfg, name)
    chosen = digest if digest is not None else configured_digest(cfg, name)
    return resolved(image, chosen)


def container_name(cfg: Config, run_id: str, mode: str) -> str:
    # run ids are UTC stamps plus hex, which docker accepts.
    return f"{cfg.engine.container_prefix}-{run_id}-{mode}"


def moe_backend_chain(cfg: Config) -> list[str]:
    chain = [cfg.engine.moe_backend, *cfg.engine.moe_backend_fallbacks]
    seen: list[str] = []
    for name in chain:
        if name and name not in seen:
            seen.append(name)
    return seen


def serve_argv(
    cfg: Config,
    *,
    mode: str,
    served_name: str,
    is_moe: bool,
    quant_flag: str,
    moe_backend: str | None,
    batch_invariant: bool,
    engine: str = "vllm",
    tool_parser: str | None = None,
) -> list[str]:
    """Server argv. vLLM is the default. SGLang is opt-in."""
    from sma.backends import get_backend

    return get_backend(engine).serve_argv(
        cfg,
        mode=mode,
        served_name=served_name,
        is_moe=is_moe,
        quant_flag=quant_flag,
        moe_backend=moe_backend,
        batch_invariant=batch_invariant,
        tool_parser=tool_parser,
    )


def serve_env(cfg: Config, *, mode: str, batch_invariant: bool, engine: str = "vllm") -> dict[str, str]:
    from sma.backends import get_backend

    return get_backend(engine).serve_env(cfg, mode=mode, batch_invariant=batch_invariant)


def docker_argv(
    cfg: Config,
    *,
    name: str,
    model_dir: Path,
    run_dir: Path,
    serve: list[str],
    env: dict[str, str],
    image: str | None = None,
) -> list[str]:
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        cfg.engine.gpus,
        "--ipc=host",
        "--shm-size",
        cfg.engine.shm_size,
        "-p",
        f"{cfg.engine.bind_host}:{cfg.engine.port}:8000",
        "-v",
        f"{model_dir}:/model:ro",
        "-v",
        f"{run_dir}:/sma-run:rw",
    ]
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    command.append(image or image_ref(cfg))
    command.extend(serve)
    return command


def docker_pull_argv(cfg: Config, *, engine: str | None = None) -> list[str]:
    return ["docker", "pull", image_ref(cfg, engine=engine)]


def docker_inspect_argv(cfg: Config, *, engine: str | None = None) -> list[str]:
    return [
        "docker",
        "image",
        "inspect",
        image_ref(cfg, engine=engine),
        "--format",
        "{{json .RepoDigests}}",
    ]


def docker_rm_argv(name: str) -> list[str]:
    return ["docker", "rm", "-f", name]


def docker_stop_argv(name: str, timeout_s: int) -> list[str]:
    return ["docker", "stop", "-t", str(timeout_s), name]


def docker_logs_argv(name: str, *, follow: bool = False) -> list[str]:
    return ["docker", "logs", "--follow", name] if follow else ["docker", "logs", name]


@dataclass(frozen=True)
class BenchCell:
    name: str
    shape: str
    input_len: int
    output_len: int
    concurrency: int
    repeat: int


def bench_cells(cfg: Config) -> list[BenchCell]:
    cells: list[BenchCell] = []
    repeats = cfg.speed.repeats
    for repeat in range(1, repeats + 1):
        cells.append(
            BenchCell(
                name=f"prefill_c1_r{repeat}",
                shape="prefill",
                input_len=cfg.speed.prefill.input_len,
                output_len=cfg.speed.prefill.output_len,
                concurrency=1,
                repeat=repeat,
            )
        )
        cells.append(
            BenchCell(
                name=f"decode_c1_r{repeat}",
                shape="decode",
                input_len=cfg.speed.decode.input_len,
                output_len=cfg.speed.decode.output_len,
                concurrency=1,
                repeat=repeat,
            )
        )
        for concurrency in cfg.speed.sweep_concurrency:
            cells.append(
                BenchCell(
                    name=f"sweep_c{concurrency}_r{repeat}",
                    shape="sweep",
                    input_len=cfg.speed.sweep.input_len,
                    output_len=cfg.speed.sweep.output_len,
                    concurrency=concurrency,
                    repeat=repeat,
                )
            )
    return cells


def bench_argv(
    cfg: Config,
    *,
    container: str,
    served_name: str,
    cell: BenchCell,
    engine: str = "vllm",
    result_dir: str | None = None,
) -> list[str]:
    from sma.backends import get_backend

    directory = result_dir or f"/sma-run/speed/cells/{cell.name}/attempt-1"
    return get_backend(engine).bench_argv(
        cfg,
        container=container,
        served_name=served_name,
        cell=cell,
        result_dir=directory,
    )


NEL_CONFIG = "nel.yaml"
_PACKAGED_BENCHES = Path(__file__).resolve().parent / "nel_benchmarks"


def packaged_bench(bench: str | None) -> Path | None:
    """The file behind sma:<name>, or None for any other benchmark name."""
    if not bench or not bench.startswith("sma:"):
        return None
    path = _PACKAGED_BENCHES / f"{bench[len('sma:'):]}.py"
    if not path.is_file():
        known = sorted(item.stem for item in _PACKAGED_BENCHES.glob("*.py"))
        raise SmaError(f"{bench} is not a benchmark SMA ships. Known: {', '.join('sma:' + k for k in known)}")
    return path


def nel_bench_ref(suite: QualitySuite) -> str:
    """What nel is told to run. A packaged file is copied next to nel.yaml, so the path is relative."""
    bench = suite.bench or suite.name
    packaged = packaged_bench(bench)
    return packaged.name if packaged else bench


def nel_config(cfg: Config, *, served_name: str, base_url: str, suite: QualitySuite) -> dict[str, Any]:
    """The NeMo Evaluator config for one suite, against the running quality server."""
    proxy: dict[str, Any] = {
        # Moves a leading <think> block out of the answer before scoring, for models that reason anyway.
        "interceptors": [{"name": "reasoning"}],
        "request_timeout": cfg.quality.request_timeout_s,
        "max_retries": max(cfg.retries.attempts - 1, 0),
    }
    if cfg.quality.chat_template_kwargs:
        # Merged into every request body. vLLM and SGLang both read it.
        proxy["extra_body"] = {"chat_template_kwargs": cfg.quality.chat_template_kwargs}
    return {
        "services": {
            "sma": {
                "type": "api",
                "url": base_url.rstrip("/") + "/v1/chat/completions",
                "protocol": "chat_completions",
                "model": served_name,
                "generation": {"temperature": 0.0, "seed": cfg.seed, "max_tokens": cfg.quality.max_tokens},
                "proxy": proxy,
            }
        },
        "benchmarks": [
            {
                "name": nel_bench_ref(suite),
                "max_problems": suite.limit,
                "max_concurrent": cfg.quality.concurrency,
                # Picks the same questions for every model and quant.
                "shuffle_seed": cfg.seed,
                "solver": {"type": "simple", "service": "sma"},
            }
        ],
    }


def nel_argv(cfg: Config, *, output_name: str) -> list[str]:
    """Run from the suite folder, which holds nel.yaml. Results go to <suite>/<output_name>."""
    return [cfg.quality.executable, "eval", "run", NEL_CONFIG, "-o", output_name]


EVAL_FACTORY_CONFIG = "run_config.yaml"


def eval_factory_config(cfg: Config, *, served_name: str, base_url: str, suite: QualitySuite) -> dict[str, Any]:
    """The run config an NVIDIA eval-factory container reads. Its harness posts to the URL as given."""
    return {
        "config": {
            "type": suite.task,
            "output_dir": "/results",
            "params": {"parallelism": cfg.quality.concurrency},
        },
        "target": {
            "api_endpoint": {
                "url": base_url.rstrip("/") + "/v1/chat/completions",
                "model_id": served_name,
                "type": "chat",
            }
        },
    }


def eval_factory_argv(*, name: str, image: str, attempt_dir: Path, config_path: Path, uid: int, gid: int) -> list[str]:
    """Run the container's harness against the quality server.

    Host networking, so 127.0.0.1 is the quality server's published port. The host user owns the
    output, so a later run can remove it. Telemetry is off: the harness would otherwise report the
    model name and task to NVIDIA.
    """
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--user",
        f"{uid}:{gid}",
        "-e",
        "HOME=/tmp",
        "-e",
        "NEMO_EVALUATOR_TELEMETRY_LEVEL=0",
        "-v",
        f"{attempt_dir}:/results",
        "-v",
        f"{config_path}:/config/{EVAL_FACTORY_CONFIG}:ro",
        image,
        "sh",
        "-c",
        "exec $(command -v nemo-evaluator >/dev/null 2>&1 && echo nemo-evaluator || echo eval-factory) "
        f"run_eval --run_config /config/{EVAL_FACTORY_CONFIG}",
    ]


def nvidia_smi_query(gpu_index: int) -> list[str]:
    fields = ",".join(
        [
            "clocks.sm",
            "temperature.gpu",
            "power.draw",
            "clocks_event_reasons.hw_thermal_slowdown",
            "clocks_event_reasons.sw_thermal_slowdown",
            "clocks_event_reasons.hw_power_brake_slowdown",
            "clocks_event_reasons.sw_power_cap",
        ]
    )
    return [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]


def nvidia_smi_query_legacy(gpu_index: int) -> list[str]:
    fields = ",".join(
        [
            "clocks.sm",
            "temperature.gpu",
            "power.draw",
            "clocks_throttle_reasons.hw_thermal_slowdown",
            "clocks_throttle_reasons.sw_thermal_slowdown",
            "clocks_throttle_reasons.hw_power_brake_slowdown",
            "clocks_throttle_reasons.sw_power_cap",
        ]
    )
    return [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
