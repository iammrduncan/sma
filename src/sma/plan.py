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
    image_ref,
    docker_rm_argv,
    NEL_CONFIG,
    moe_backend_chain,
    nel_argv,
    nvidia_smi_query,
    serve_argv,
    serve_env,
)
from sma.backends import BenchMounts, get_backend
from sma.commands import perplexity_argv
from sma.config import Config
from sma.engines import batch_invariance_replaces_kernels, configured_digest, image_for
from sma.kernelbench import COMMIT as KB_COMMIT
from sma.swebench import agent_command, grade_command
from sma.runplan import describe, serve_plan


def render_plan(
    cfg: Config,
    *,
    repo: str,
    quant_flag: str,
    revision: str | None,
    assume_moe: bool,
    engine: str = "vllm",
    tool_parser: str | None = None,
    gguf: str | None = None,
    selected: str = "default",
    memory_plan: dict | None = None,
    memory_note: str | None = None,
    model_file: str | None = None,
) -> str:
    run_id = "<run-id>"
    plan = serve_plan(cfg, memory_plan, model_file=model_file)
    backend_info = get_backend(engine)
    run_dir = Path("<run-dir>")
    model_dir = Path("<snapshot>")
    served = repo
    is_moe = assume_moe
    real_kernels = batch_invariance_replaces_kernels(engine, quant_flag)
    batch_invariant = cfg.engine.batch_invariant and not real_kernels
    backend = None
    if engine == "vllm" and is_moe and quant_flag in {"nvfp4", "mxfp4", "mixed"}:
        backend = cfg.engine.moe_backend
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
        plan=plan,
        engine=engine,
        tool_parser=tool_parser or cfg.engine.tool_call_parser,
    )
    speed_serve = serve_argv(
        cfg,
        mode="speed",
        served_name=served,
        is_moe=is_moe,
        quant_flag=quant_flag,
        moe_backend=backend,
        batch_invariant=False,
        plan=plan,
        engine=engine,
    )
    quality_env = serve_env(cfg, mode="quality", batch_invariant=batch_invariant, engine=engine, is_moe=is_moe, quant_flag=quant_flag)
    speed_env = serve_env(cfg, mode="speed", batch_invariant=False, engine=engine, is_moe=is_moe, quant_flag=quant_flag)
    lines = [
        "SMA dry-run",
        f"model: {repo}",
        f"quant: {quant_flag}",
        f"revision: {revision or '<default branch, pinned to the resolved commit SHA>'}",
        f"host_alias: {cfg.host_alias}",
        f"engine: {engine} ({_why(engine, selected)})",
        f"image: {image_for(cfg, engine)}",
        f"image_digest: {configured_digest(cfg, engine) or '<recorded at docker inspect; not invented>'}",
        "tensor_parallel_size: 1",
        "quantize: no",
        f"kv cache: {backend_info.kv_cache_type} (always 8-bit)",
    ]
    if gguf:
        lines.insert(3, f"gguf: {gguf} (the file's own metadata names its quant type)")
    if memory_plan:
        lines.extend("memory plan: " + line for line in describe(memory_plan))
    else:
        lines.append(f"memory plan: computed at fetch from the checkpoint's config and file sizes. {memory_note or ''}".rstrip())
    lines.append(
        f"  Budget {cfg.memory.budget_gb:g} GB. Context starts at {cfg.memory.start_context:,} tokens, or the "
        "native maximum if smaller, never RoPE-stretched, and halves until one stream fits. "
        "A run that cannot fit is refused before the weight download."
    )
    lines.extend(
        [
        "",
        "1 fetch",
        "  Resolve the repo to a commit SHA with huggingface_hub, then snapshot_download that SHA (resumable) into <hf.cache_dir>/<model>/<sha>.",
        (
            f"  GGUF: pick the {gguf} file (or split set), read its header with ranged requests, plan memory, "
            "then download only those files. Hash them and read the quant type, tensor types, and KV layout from the metadata."
            if gguf
            else "  Hash weight files, read safetensors shapes for total and active params, and read config.json / hf_quant_config.json."
        ),
        f"  Refuse unless the checkpoint matches --{quant_flag}. SMA does not quantize or convert.",
        "",
        ]
    )
    lines.extend([
        "2 image",
        "  " + shlex.join(docker_pull_argv(cfg, engine=engine)),
        "  " + shlex.join(docker_inspect_argv(cfg, engine=engine)),
        "",
        "3 serve quality "
        + (
            f"(batch invariance off: a {quant_flag} checkpoint runs its real kernels, which vLLM's batch-invariant mode replaces)"
            if real_kernels
            else "(batch invariance on when the architecture allows it)"
        ),
        "quality-server: "
        + shlex.join(
            docker_argv(
                cfg,
                name=quality_name,
                model_dir=model_dir,
                run_dir=run_dir,
                serve=quality_serve,
                env=quality_env,
                image=image_ref(cfg, engine=engine),
                entrypoint=backend_info.entrypoint,
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
        *_logprob_lines(cfg, engine, run_dir, model_dir, plan),
        "",
        "6 quality",
        f"  NeMo Evaluator against http://{cfg.engine.bind_host}:{cfg.engine.port}/v1/chat/completions, "
        f"{cfg.quality.concurrency} requests in flight per "
        f"suite (recorded), temperature 0, seed {cfg.seed}, max_tokens {cfg.quality.max_tokens}, "
        f"chat_template_kwargs {cfg.quality.chat_template_kwargs}. Not an arena pinned scorer.",
        f"  Each suite writes <run-dir>/quality/<suite>/{NEL_CONFIG} and runs there: "
        + shlex.join(nel_argv(cfg, output_name="attempt-1")),
        ]
    )
    for suite in cfg.quality.suites:
        if suite.image is not None:
            tools = ", calls tools (needs --tool-parser)" if suite.tools else ""
            lines.append(
                f"  {suite.name}: {suite.image} task {suite.task}, all questions, run directly with host "
                f"networking and telemetry off{tools}"
            )
            continue
        size = "all questions" if suite.limit is None else f"{suite.limit} questions (seeded shuffle)"
        lines.append(f"  {suite.name}: {suite.bench or suite.name}, {size}")
    if cfg.kernelbench.enabled:
        lines += [
            "",
            f"6a kernelbench: KernelBench {KB_COMMIT[:8]} Level 1, 100 problems, its one-shot CUDA prompt as a chat, "
            f"{cfg.kernelbench.concurrency} in flight, up to {cfg.kernelbench.max_tokens} output tokens, with the model's own "
            "sampling and thinking mode. The kernels go to $KERNELBENCH_URL as one job; 8a collects the results.",
        ]
    if cfg.swebench.enabled:
        lines += [
            "",
            f"6b swebench: mini-swe-agent on the runner ($SWE_RUNNER_HOST over SSH), {cfg.swebench.workers} agents, "
            f"{cfg.swebench.step_limit} steps, the model's own sampling and thinking mode, with port {cfg.engine.port} "
            "forwarded to the runner's loopback. Needs --tool-parser.",
            "  agent: " + agent_command(cfg, run_id),
            "  grade: " + grade_command(cfg, run_id),
        ]
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
                    image=image_ref(cfg, engine=engine),
                    entrypoint=backend_info.entrypoint,
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
                bench_argv(
                    cfg,
                    container=speed_name,
                    served_name=served,
                    cell=cell,
                    engine=engine,
                    result_dir=f"/sma-run/speed/cells/{cell.name}/attempt-1",
                    mounts=(
                        BenchMounts(image_ref(cfg, engine="vllm"), str(model_dir), str(run_dir), f"/model/{plan.model_file}", 0, 0)
                        if engine == "llamacpp"
                        else None
                    ),
                )
            )
        )
    lines.extend(
        [
            "  Derived: prefill tok/s = input_len / TTFT at concurrency 1; decode tok/s = 1 / TPOT at concurrency 1.",
            (
                "  Scrape GET /metrics. Watched SGLang counters that are absent are null, with a reason. "
                "vLLM names are not copied in."
                if engine == "sglang"
                else "  Scrape GET /metrics before and after each cell. Metric names match with labels stripped."
            ),
            "  A real thermal or power throttle marks that cell invalid and retries once after a cool-down.",
            "  Missing telemetry, a failed scrape, an empty window, or [N/A] flags mark the cell unverified. "
            "Unverified cells are not retried and are left out of the headline tok/s.",
            (
                "  note: stock vLLM does not load mixed FP8/NVFP4. Fetch refuses --mixed on vllm. "
                "SMA does not rewrite the checkpoint."
                if engine == "vllm" and quant_flag == "mixed"
                else "  quant/engine pair is checked before the snapshot download."
            ),
            "",
            *(
                ["8a kernelbench_results: wait for the service's job, then score it with KernelBench's fast_p.", ""]
                if cfg.kernelbench.enabled
                else []
            ),
            "9 telemetry",
            "  " + shlex.join(nvidia_smi_query(cfg.doctor.gpu_index)),
            f"  Sampled every {cfg.telemetry.interval_s}s for the whole run (SM clock, temperature, power, throttle reasons).",
            "",
            "10 teardown (always: success, failure, SIGINT, SIGTERM)",
            "  " + shlex.join(docker_rm_argv(quality_name)),
            "  " + shlex.join(docker_rm_argv(speed_name)),
            *(
                ["  " + shlex.join(docker_rm_argv(container_name(cfg, run_id, name))) for name in ("logprobs", "bench")]
                if engine == "llamacpp"
                else []
            ),
            "  Drop the page cache if /proc/sys/vm/drop_caches is writable, then idle until cool.",
            "",
            _moe_note(cfg, quant_flag, assume_moe, engine),
        ]
    )
    return "\n".join(lines) + "\n"


def _why(engine: str, selected: str) -> str:
    return {
        "format": "GGUF runs on llama.cpp",
        "flag": "chosen with --engine",
        "preflight": "preflight found vLLM cannot serve this architecture",
    }.get(selected, "the default for safetensors; preflight may pick SGLang if vLLM cannot serve the architecture" if engine == "vllm" else "configured default")


def _logprob_lines(cfg: Config, engine: str, run_dir: Path, model_dir: Path, plan) -> list[str]:
    if engine == "llamacpp":
        argv = perplexity_argv(
            cfg,
            name=container_name(cfg, "<run-id>", "logprobs"),
            image=image_ref(cfg, engine=engine),
            model_dir=model_dir,
            run_dir=run_dir,
            model_file=str(plan.model_file),
            uid=0,
            gid=0,
            reference_base=None,
            write_base=True,
        )
        return [
            "  llama-server returns no prompt logprobs. The quality server stops and llama.cpp's llama-perplexity "
            f"scores {cfg.logprobs.llamacpp_chunks} chunks of {cfg.logprobs.window_tokens} tokens of {cfg.logprobs.corpus_version}:",
            "  " + shlex.join(argv),
            "  A full-precision GGUF writes its logits (--kl-divergence-base). With --reference, a quant reads the "
            "reference's logits and reports KL over the full vocabulary (--kl-divergence).",
        ]
    return [
        f"  Corpus {cfg.logprobs.corpus_version}, {cfg.logprobs.num_windows} windows of "
        f"{cfg.logprobs.window_tokens} token ids, no chat template.",
        (
            "  POST /generate input_ids=<token ids> return_logprob top_logprobs_num="
            f"{cfg.logprobs.top_k}. Input-token top-k. A missing field stays null."
            if engine == "sglang"
            else f"  POST /v1/completions prompt=<token ids> prompt_logprobs={cfg.logprobs.top_k} "
            "max_tokens=1 temperature=0."
        ),
        "  Store top-k, a tail bucket, and the true next-token logprob. Perplexity = exp(-mean logprob).",
    ]


def _moe_note(cfg: Config, quant_flag: str, assume_moe: bool, engine: str) -> str:
    chain = ", ".join(moe_backend_chain(cfg))
    if engine == "sglang":
        return (
            "moe: SGLang does not take vLLM's --moe-backend chain. "
            "moe_backend_quality and moe_backend_speed stay unset unless a log line names one. "
            "SMA does not invent a backend name. NVFP4 on SGLang is accepted for ModelOpt checkpoints; "
            "compressed-tensors NVFP4 is refused and the error says to use vLLM."
        )
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
        "and the choice is recorded. Dense NVFP4 sets VLLM_NVFP4_GEMM_BACKEND="
        f"{cfg.engine.linear_backend.replace('_', '-')} instead. Re-run with --assume-moe to print the MoE command."
    )
