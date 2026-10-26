"""The exact argv SMA prints in a dry-run and executes for real."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sma.config import Config


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


def docker_logs_argv(name: str) -> list[str]:
    return ["docker", "logs", name]


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
) -> list[str]:
    from sma.backends import get_backend

    return get_backend(engine).bench_argv(
        cfg, container=container, served_name=served_name, cell=cell
    )


def lm_eval_argv(
    cfg: Config,
    *,
    served_name: str,
    base_url: str,
    suite_name: str,
    tasks: str | None,
    limit: int | None,
    output_dir: Path,
) -> list[str]:
    model_args = (
        f"model={served_name},base_url={base_url}/v1,"
        f"num_concurrent={cfg.quality.concurrency},tokenized_requests=False"
    )
    command = [
        cfg.quality.executable,
        "--model",
        cfg.quality.model_backend,
        "--model_args",
        model_args,
        "--tasks",
        tasks or suite_name,
        "--output_path",
        str(output_dir),
        "--log_samples",
        "--seed",
        str(cfg.seed),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


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
