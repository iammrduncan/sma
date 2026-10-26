"""The exact argv SMA prints in a dry-run and executes for real."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sma.config import Config


def image_ref(cfg: Config) -> str:
    image = cfg.engine.image
    digest = cfg.engine.image_digest
    if digest:
        base = image.split("@", 1)[0]
        if not digest.startswith("sha256:"):
            digest = "sha256:" + digest
        return f"{base}@{digest}"
    return image


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
) -> list[str]:
    """vLLM serve argv. NGC images use a pass-through entrypoint, so `vllm` is included."""
    quality = mode == "quality"
    flags = [
        "vllm",
        "serve",
        "/model",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        served_name,
        "--seed",
        str(cfg.seed),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        f"{cfg.engine.gpu_memory_utilization:.4g}",
        "--max-model-len",
        str(cfg.engine.max_model_len),
        "--max-num-seqs",
        str(cfg.engine.max_num_seqs),
        "--max-logprobs",
        str(cfg.logprobs.top_k),
        "--logprobs-mode",
        "raw_logprobs",
    ]
    if quality and cfg.engine.no_flashinfer_autotune_quality:
        flags.append("--no-enable-flashinfer-autotune")
    if quality and batch_invariant and cfg.engine.no_async_scheduling_with_batch_invariant:
        flags.append("--no-async-scheduling")
    if is_moe and quant_flag in {"nvfp4", "mxfp4"} and moe_backend:
        # Explicit backend. Auto-selection on sm_121 can land on Marlin.
        flags.extend(["--moe-backend", moe_backend])
    if (not is_moe) and quant_flag == "nvfp4" and cfg.engine.linear_backend:
        flags.extend(["--linear-backend", cfg.engine.linear_backend])
    if cfg.engine.trust_remote_code:
        flags.append("--trust-remote-code")
    flags.extend(cfg.engine.extra_args)
    return flags


def serve_env(*, mode: str, batch_invariant: bool) -> dict[str, str]:
    if mode == "quality" and batch_invariant:
        return {"VLLM_BATCH_INVARIANT": "1"}
    return {}


def docker_argv(
    cfg: Config,
    *,
    name: str,
    model_dir: Path,
    run_dir: Path,
    serve: list[str],
    env: dict[str, str],
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
    command.append(image_ref(cfg))
    command.extend(serve)
    return command


def docker_pull_argv(cfg: Config) -> list[str]:
    return ["docker", "pull", image_ref(cfg)]


def docker_inspect_argv(cfg: Config) -> list[str]:
    return [
        "docker",
        "image",
        "inspect",
        image_ref(cfg),
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
) -> list[str]:
    result_dir = f"/sma-run/speed/cells/{cell.name}"
    return [
        "docker",
        "exec",
        container,
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        "http://127.0.0.1:8000",
        "--endpoint",
        "/v1/completions",
        "--model",
        served_name,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(cell.input_len),
        "--random-output-len",
        str(cell.output_len),
        "--num-prompts",
        str(cfg.speed.num_prompts),
        "--max-concurrency",
        str(cell.concurrency),
        "--request-rate",
        "inf",
        "--percentile-metrics",
        cfg.speed.percentile_metrics,
        "--metric-percentiles",
        cfg.speed.metric_percentiles,
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        "bench.json",
        "--seed",
        str(cfg.seed),
    ]


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
