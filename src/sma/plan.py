"""Dry-run plan. The commands are the same ones the real run executes."""

from __future__ import annotations

import shlex
from pathlib import Path

from sma.commands import (
    bench_argv,
    bench_cells,
    container_name,
    docker_argv,
    docker_inspect_argv,
    docker_pull_argv,
    docker_rm_argv,
    lm_eval_argv,
    moe_backend_chain,
    nvidia_smi_query,
    serve_argv,
    serve_env,
)
from sma.config import Config


def render_plan(
    cfg: Config,
    *,
    repo: str,
    quant_flag: str,
    revision: str | None,
    assume_moe: bool,
) -> str:
    run_id = "<run-id>"
    run_dir = Path("<run-dir>")
    model_dir = Path("<snapshot>")
    served = repo
    is_moe = assume_moe
    batch_invariant = cfg.engine.batch_invariant
    backend = cfg.engine.moe_backend if is_moe and quant_flag in {"nvfp4", "mxfp4"} else None
    quality_name = container_name(cfg, run_id, "quality")
    speed_name = container_name(cfg, run_id, "speed")
    quality_serve = serve_argv(
        cfg,
        mode="quality",
        served_name=served,
        is_moe=is_moe,
        quant_flag=quant_flag,
        moe_backend=backend,
        batch_invariant=batch_invariant,
    )
    speed_serve = serve_argv(
        cfg,
        mode="speed",
        served_name=served,
        is_moe=is_moe,
        quant_flag=quant_flag,
        moe_backend=backend,
        batch_invariant=False,
    )
    quality_env = serve_env(mode="quality", batch_invariant=batch_invariant)
    speed_env = serve_env(mode="speed", batch_invariant=False)
    lines = [
        "SMA dry-run",
        f"model: {repo}",
        f"quant: {quant_flag}",
        f"revision: {revision or '<default branch, pinned to the resolved commit SHA>'}",
        f"host_alias: {cfg.host_alias}",
        f"engine: vllm",
        f"image: {cfg.engine.image}",
        f"image_digest: {cfg.engine.image_digest or '<recorded at docker inspect; not invented>'}",
        "tensor_parallel_size: 1",
        "quantize: no",
        "",
        "1 fetch",
        "  Resolve the repo to a commit SHA with huggingface_hub, then snapshot_download that SHA (resumable).",
        "  Hash weight files, read safetensors shapes for total and active params, and read config.json / hf_quant_config.json.",
        f"  Refuse unless the checkpoint matches --{quant_flag}. SMA does not quantize.",
        "",
        "2 image",
        "  " + shlex.join(docker_pull_argv(cfg)),
        "  " + shlex.join(docker_inspect_argv(cfg)),
        "",
        "3 serve quality (batch invariance on when the architecture allows it)",
        "quality-server: "
        + shlex.join(
            docker_argv(
                cfg,
                name=quality_name,
                model_dir=model_dir,
                run_dir=run_dir,
                serve=quality_serve,
                env=quality_env,
            )
        ),
        f"  env: {quality_env or '{}'}",
        "  Health check: GET /health and GET /v1/models, with a deadline.",
        "",
        "4 determinism",
        "  Same prompts at batch sizes "
        + ", ".join(str(size) for size in cfg.determinism.batch_sizes)
        + ". One completions request per batch, temperature 0. Record pass or fail. Do not abort.",
        "",
        "5 logprobs",
        f"  Corpus {cfg.logprobs.corpus_version}, {cfg.logprobs.num_windows} windows of "
        f"{cfg.logprobs.window_tokens} token ids, no chat template.",
        f"  POST /v1/completions prompt=<token ids> prompt_logprobs={cfg.logprobs.top_k} "
        "max_tokens=1 temperature=0.",
        "  Store top-k, a tail bucket, and the true next-token logprob. Perplexity = exp(-mean logprob).",
        "",
        "6 quality",
        f"  Packed concurrency {cfg.quality.concurrency} (recorded). Scorer label: lm-eval-harness, not an arena pinned scorer.",
    ]
    base = f"http://{cfg.engine.bind_host}:{cfg.engine.port}"
    for suite in cfg.quality.suites:
        lines.append(
            "  "
            + shlex.join(
                lm_eval_argv(
                    cfg,
                    served_name=served,
                    base_url=base,
                    suite_name=suite.name,
                    tasks=suite.tasks,
                    limit=suite.limit,
                    output_dir=Path(f"<run-dir>/quality/{suite.name}"),
                )
            )
        )
    lines.extend(
        [
            "",
            "7 restart speed (batch invariance off, autotune left at the image default)",
            "speed-server: "
            + shlex.join(
                docker_argv(
                    cfg,
                    name=speed_name,
                    model_dir=model_dir,
                    run_dir=run_dir,
                    serve=speed_serve,
                    env=speed_env,
                )
            ),
            f"  env: {speed_env or '{}'}",
            f"  Warm-up requests: {cfg.speed.warmup_requests}. Soak seconds: {cfg.speed.warmup_soak_s}.",
            "",
            "8 speed (solo)",
        ]
    )
    for cell in bench_cells(cfg):
        lines.append(
            "  "
            + shlex.join(
                bench_argv(cfg, container=speed_name, served_name=served, cell=cell)
            )
        )
    lines.extend(
        [
            "  Derived: prefill tok/s = input_len / TTFT at concurrency 1; decode tok/s = 1 / TPOT at concurrency 1.",
            "  Scrape GET /metrics before and after each cell and store the counter diff.",
            "  A thermal or power throttle marks that cell invalid and retries once after a cool-down.",
            "",
            "9 telemetry",
            "  " + shlex.join(nvidia_smi_query(cfg.doctor.gpu_index)),
            f"  Sampled every {cfg.telemetry.interval_s}s for the whole run (SM clock, temperature, power, throttle reasons).",
            "",
            "10 teardown (always: success, failure, SIGINT, SIGTERM)",
            "  " + shlex.join(docker_rm_argv(quality_name)),
            "  " + shlex.join(docker_rm_argv(speed_name)),
            "  Drop the page cache if /proc/sys/vm/drop_caches is writable, then idle until cool.",
            "",
            _moe_note(cfg, quant_flag, assume_moe),
        ]
    )
    return "\n".join(lines) + "\n"


def _moe_note(cfg: Config, quant_flag: str, assume_moe: bool) -> str:
    chain = ", ".join(moe_backend_chain(cfg))
    if quant_flag == "bf16":
        return (
            "moe: --moe-backend is not passed for BF16. "
            f"For an NVFP4 or MXFP4 MoE checkpoint the chain is {chain}. "
            "Auto-selection is never used for those checkpoints."
        )
    if assume_moe:
        return (
            f"moe: --assume-moe is set, so the server command includes --moe-backend {cfg.engine.moe_backend}. "
            f"If that backend fails health checks, SMA tries {chain} and records the one that started. "
            "A failed FP4 cell is recorded as failed, not skipped."
        )
    return (
        f"moe: not known until fetch. If the checkpoint is MoE, --moe-backend is set from [{chain}] "
        "and the choice is recorded. Dense NVFP4 also passes --linear-backend "
        f"{cfg.engine.linear_backend}. Re-run with --assume-moe to print the MoE command."
    )
