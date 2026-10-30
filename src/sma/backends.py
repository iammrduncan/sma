"""Small backend interface. vLLM is the default, SGLang is opt-in, llama.cpp runs GGUF files.

Each engine owns its serve command, health check, logprobs, bench command,
metrics names, and deterministic-mode flags. A measurement the engine does
not return stays null; nothing is copied from another engine.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sma.config import Config
from sma.errors import SmaError


@dataclass(frozen=True)
class ServePlan:
    """What the memory plan and the checkpoint give a serve command.

    In a dry run before the plan exists, the numbers are placeholder strings such as "<context>".
    """

    context: int | str
    pool_bytes: int | str
    pool_tokens: int | str
    # llama.cpp: the GGUF file under /model (the first file of a split set). None for safetensors.
    model_file: str | None = None
    # llama.cpp: parallel slots. The unified KV pool is shared by all of them.
    slots: int = 8
    # The checkpoint's weights in bytes. SGLang's --mem-fraction-static is set from them and the pool.
    weights_bytes: int | None = None


@dataclass(frozen=True)
class BenchMounts:
    """A bench that runs in its own container (llama.cpp): the image and host folders it mounts."""

    image: str
    model_dir: str
    run_dir: str
    tokenizer: str
    uid: int
    gid: int
    # A tokenizer folder mounted at BENCH_TOKENIZER_DIR, when the GGUF's own cannot be read by the bench.
    tokenizer_dir: str | None = None


BENCH_TOKENIZER_DIR = "/bench-tokenizer"


class EngineBackend(Protocol):
    name: str
    # The engine's 8-bit KV cache type, in the memory plan's terms (sma.memplan.KV_TYPES).
    kv_cache_type: str
    # Set when the image's own entrypoint must be replaced (llama.cpp's "full" image).
    entrypoint: str | None

    def serve_argv(
        self,
        cfg: Config,
        *,
        mode: str,
        served_name: str,
        is_moe: bool,
        quant_flag: str,
        moe_backend: str | None,
        batch_invariant: bool,
        plan: ServePlan,
        tool_parser: str | None = None,
    ) -> list[str]: ...

    def serve_env(
        self, cfg: Config, *, mode: str, batch_invariant: bool, is_moe: bool = False, quant_flag: str = ""
    ) -> dict[str, str]: ...

    def health_paths(self) -> tuple[str, ...]: ...

    def bench_argv(
        self,
        cfg: Config,
        *,
        container: str,
        served_name: str,
        cell: Any,
        result_dir: str,
        mounts: BenchMounts | None = None,
    ) -> list[str]: ...

    def kv_report(self, log_text: str) -> dict[str, Any]: ...

    def metrics_path(self) -> str: ...

    def watched_metrics(self) -> tuple[str, ...]: ...

    def metrics_note(self) -> str: ...

    def serving_metrics(self) -> tuple[str, ...]: ...

    def serving_parts(self, change: dict[str, float], seconds: float) -> dict[str, tuple[float | None, float | None]]: ...

    def serving_rates(
        self, change: dict[str, float], mean: dict[str, float], seconds: float
    ) -> dict[str, float | None]: ...

    def logprob_positions(
        self,
        *,
        base_url: str,
        served_name: str,
        token_ids: list[int],
        top_k: int,
        seed: int,
        timeout_s: float,
    ) -> list[dict[str, Any]]: ...

    def deterministic(self, cfg: Config) -> dict[str, Any]: ...


# Mounted into the run directory. SGLang's random dataset downloads ShareGPT
# when this path is missing. See python/sglang/benchmark/datasets/random.py
# at tag v0.5.10.post1, sample_random_requests.
SGLANG_DATASET_CONTAINER_PATH = "/sma-run/speed/sglang-random-dataset.json"
SGLANG_DATASET_NAME = "sglang-random-dataset.json"
# The bench container must not fetch a tokenizer. vLLM uses
# tokenizer_id = args.tokenizer or model_id (vllm/benchmarks/serve.py).
# SGLang falls back to the repo id from /v1/models and
# AutoTokenizer.from_pretrained(..., trust_remote_code=True)
# (bench_serving.py around the tokenizer load, and benchmark/utils.py).
BENCH_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

# Names vLLM documents. Absent series stay null.
VLLM_WATCHED = (
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:request_time_per_output_token_seconds_sum",
    "vllm:request_time_per_output_token_seconds_count",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_sum",
)

# SGLang's own series. If a build does not emit one, the diff stores null.
SGLANG_WATCHED = (
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:cached_tokens_total",
    "sglang:num_requests_total",
    "sglang:time_to_first_token_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:inter_token_latency_seconds_sum",
    "sglang:inter_token_latency_seconds_count",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
)


SGLANG_PREFILL_SECONDS = 'sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward"}'


def docker_exec(container: str, command: list[str]) -> list[str]:
    """docker exec with Hub access turned off for the bench process."""
    argv = ["docker", "exec"]
    for key, value in BENCH_OFFLINE_ENV.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.append(container)
    argv.extend(command)
    return argv


def sglang_dataset_rows() -> int:
    meta_path = Path(__file__).resolve().parent / "data" / "sglang-random-dataset.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = meta.get("rows")
    if not isinstance(rows, int) or rows < 1:
        raise SmaError(f"{meta_path.name} does not record a positive row count.")
    return rows


def assert_sglang_dataset_covers(num_prompts: int) -> None:
    """Refuse rather than silently using fewer prompts than configured."""
    rows = sglang_dataset_rows()
    if num_prompts > rows:
        raise SmaError(
            f"speed.num_prompts is {num_prompts}, but the pinned SGLang dataset "
            f"{SGLANG_DATASET_NAME} has {rows} rows. SMA will not cap the benchmark. "
            "Lower speed.num_prompts, or enlarge that file and its sha256."
        )


class VllmBackend:
    name = "vllm"
    kv_cache_type = "fp8"
    entrypoint = None

    def serve_argv(
        self,
        cfg: Config,
        *,
        mode: str,
        served_name: str,
        is_moe: bool,
        quant_flag: str,
        moe_backend: str | None,
        batch_invariant: bool,
        plan: ServePlan,
        tool_parser: str | None = None,
    ) -> list[str]:
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
            # The memory plan sizes the pool and the context. The KV cache is always 8-bit.
            "--kv-cache-memory-bytes",
            str(plan.pool_bytes),
            "--kv-cache-dtype",
            "fp8",
            "--max-model-len",
            str(plan.context),
            "--max-num-seqs",
            str(cfg.engine.max_num_seqs),
            "--max-logprobs",
            str(cfg.logprobs.top_k),
            "--logprobs-mode",
            "raw_logprobs",
        ]
        if quality and cfg.engine.no_flashinfer_autotune_quality:
            flags.append("--no-enable-flashinfer-autotune")
        if quality:
            # The determinism probe shares this server. A prefix-cache hit
            # would hide batch-size differences.
            flags.append("--no-enable-prefix-caching")
        if quality and batch_invariant and cfg.engine.no_async_scheduling_with_batch_invariant:
            flags.append("--no-async-scheduling")
        if is_moe and quant_flag in {"nvfp4", "mxfp4", "mixed"} and moe_backend:
            flags.extend(["--moe-backend", moe_backend])
        if quality and cfg.quality.chat_template_kwargs:
            # The default for every quality client, including harnesses that cannot add it to a
            # request (BFCL). A request's own chat_template_kwargs still wins.
            flags.extend(
                ["--default-chat-template-kwargs", json.dumps(cfg.quality.chat_template_kwargs, sort_keys=True)]
            )
        if quality and tool_parser:
            flags.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])
        if cfg.engine.reasoning_parser:
            flags.extend(["--reasoning-parser", cfg.engine.reasoning_parser])
        if cfg.engine.trust_remote_code:
            flags.append("--trust-remote-code")
        flags.extend(cfg.engine.extra_args)
        return flags

    def serve_env(
        self, cfg: Config, *, mode: str, batch_invariant: bool, is_moe: bool = False, quant_flag: str = ""
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        if mode == "quality" and batch_invariant:
            env["VLLM_BATCH_INVARIANT"] = "1"
        if (not is_moe) and quant_flag == "nvfp4" and cfg.engine.linear_backend:
            # vLLM 0.21 has no --linear-backend flag ("unrecognized arguments" on 2026-10-27). The dense
            # NVFP4 GEMM is chosen by this variable, spelled with hyphens (vllm/envs.py).
            env["VLLM_NVFP4_GEMM_BACKEND"] = cfg.engine.linear_backend.replace("_", "-")
        return env

    def health_paths(self) -> tuple[str, ...]:
        return ("/health", "/v1/models")

    def bench_argv(
        self,
        cfg: Config,
        *,
        container: str,
        served_name: str,
        cell: Any,
        result_dir: str,
        mounts: BenchMounts | None = None,
    ) -> list[str]:
        del mounts
        # --model is the served API name. --tokenizer /model is the pinned snapshot.
        # Without --tokenizer, vLLM sets tokenizer_id from the repo id and pulls main.
        return docker_exec(container, vllm_bench_command(cfg, served_name=served_name, cell=cell, result_dir=result_dir,
                                                         base_url="http://127.0.0.1:8000", tokenizer="/model"))

    def kv_report(self, log_text: str) -> dict[str, Any]:
        # A checkpoint can declare FP8 KV scales that are all 1.0 (nvidia/Qwen3-8B-NVFP4 does); vLLM says so.
        return _kv_report(
            log_text,
            type_line=r"Using (\S+) data type to store kv cache",
            tokens=_VLLM_KV_TOKENS,
            scale_one=r"Using KV cache scaling factor 1\.0",
        )

    def metrics_path(self) -> str:
        return "/metrics"

    def watched_metrics(self) -> tuple[str, ...]:
        return VLLM_WATCHED

    def metrics_note(self) -> str:
        return (
            "Watched names are vLLM counters, matched with labels stripped. "
            "A name absent from /metrics is null. Nothing is filled in."
        )

    def serving_metrics(self) -> tuple[str, ...]:
        return VLLM_WATCHED + (
            "vllm:request_prefill_time_seconds_sum",
            "vllm:request_decode_time_seconds_sum",
            "vllm:num_requests_running",
        )

    def serving_parts(self, change: dict[str, float], seconds: float) -> dict[str, tuple[float | None, float | None]]:
        return {
            "prefill_tok_s": (change.get("vllm:prompt_tokens_total"), change.get("vllm:request_prefill_time_seconds_sum")),
            "decode_tok_s": (change.get("vllm:generation_tokens_total"), change.get("vllm:request_decode_time_seconds_sum")),
            "itl_ms": (change.get("vllm:inter_token_latency_seconds_sum"), change.get("vllm:inter_token_latency_seconds_count")),
            "generated_tok_s_all": (change.get("vllm:generation_tokens_total"), seconds),
        }

    def serving_rates(self, change: dict[str, float], mean: dict[str, float], seconds: float) -> dict[str, float | None]:
        return {
            **rates_from_parts([self.serving_parts(change, seconds)]),
            "ttft_ms": _ratio(change.get("vllm:time_to_first_token_seconds_sum"), change.get("vllm:time_to_first_token_seconds_count"), 1000),
            "prompt_tok_s_all": _ratio(change.get("vllm:prompt_tokens_total"), seconds),
            "in_flight": mean.get("vllm:num_requests_running"),
        }

    def logprob_positions(
        self,
        *,
        base_url: str,
        served_name: str,
        token_ids: list[int],
        top_k: int,
        seed: int,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        from sma.probe import request_window_logprobs

        return request_window_logprobs(
            base_url=base_url,
            served_name=served_name,
            token_ids=token_ids,
            top_k=top_k,
            seed=seed,
            timeout_s=timeout_s,
        )

    def deterministic(self, cfg: Config) -> dict[str, Any]:
        return {
            "prefix_cache": False,
            "attention_backend": None,
            "determinism_backend": "batch_invariant" if cfg.engine.batch_invariant else None,
            "radix_cache": None,
        }


def vllm_bench_command(cfg: Config, *, served_name: str, cell: Any, result_dir: str, base_url: str, tokenizer: str) -> list[str]:
    """vLLM's bench client. It drives vLLM and, from its own container, llama-server."""
    return [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "openai",
            "--base-url",
            base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            served_name,
            "--tokenizer",
            tokenizer,
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


class SglangBackend:
    name = "sglang"
    kv_cache_type = "fp8"
    entrypoint = None

    def serve_argv(
        self,
        cfg: Config,
        *,
        mode: str,
        served_name: str,
        is_moe: bool,
        quant_flag: str,
        moe_backend: str | None,
        batch_invariant: bool,
        plan: ServePlan,
        tool_parser: str | None = None,
    ) -> list[str]:
        del is_moe, quant_flag, moe_backend, batch_invariant
        backend = cfg.engine.sglang_attention_backend
        flags = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "/model",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--served-model-name",
            served_name,
            "--tp-size",
            "1",
            "--mem-fraction-static",
            f"{sglang_mem_fraction(cfg, plan):.4g}",
            # The memory plan sizes the pool (in tokens) and the context. The KV cache is always 8-bit.
            "--context-length",
            str(plan.context),
            "--max-total-tokens",
            str(plan.pool_tokens),
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--attention-backend",
            backend,
            "--max-running-requests",
            str(cfg.engine.max_num_seqs),
            "--random-seed",
            str(cfg.seed),
            "--enable-metrics",
        ]
        if mode == "quality":
            # FlashInfer deterministic serving disables the radix cache.
            if backend == "flashinfer" or cfg.engine.sglang_determinism_backend == "flashinfer":
                flags.append("--disable-radix-cache")
            flags.append("--enable-deterministic-inference")
            if tool_parser:
                flags.extend(["--tool-call-parser", tool_parser])
        if cfg.engine.reasoning_parser:
            flags.extend(["--reasoning-parser", cfg.engine.reasoning_parser])
        if cfg.engine.trust_remote_code:
            flags.append("--trust-remote-code")
        flags.extend(cfg.engine.extra_args)
        return flags

    def serve_env(
        self, cfg: Config, *, mode: str, batch_invariant: bool, is_moe: bool = False, quant_flag: str = ""
    ) -> dict[str, str]:
        del batch_invariant, is_moe, quant_flag
        env: dict[str, str] = {}
        if mode == "quality":
            env["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
        if cfg.engine.sglang_disable_deep_gemm:
            env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"
        return env

    def health_paths(self) -> tuple[str, ...]:
        return ("/health", "/v1/models")

    def bench_argv(
        self,
        cfg: Config,
        *,
        container: str,
        served_name: str,
        cell: Any,
        result_dir: str,
        mounts: BenchMounts | None = None,
    ) -> list[str]:
        del served_name, mounts
        assert_sglang_dataset_covers(cfg.speed.num_prompts)
        # Key names checked against sglang tag v0.5.10.post1,
        # python/sglang/bench_serving.py (result dict around the JSONL write)
        # and python/sglang/benchmark/datasets/random.py (ShareGPT download).
        # That result dict has median/p99 TTFT and TPOT, median/p95/p99 ITL,
        # and median/p90/p99 e2e latency. It has no p90 TTFT, TPOT, or ITL.
        # --output-details adds per-request ttfts and itls (seconds) so SMA
        # can compute the missing percentiles. The file is appended JSONL.
        # --model and --tokenizer are the mounted snapshot, not the repo id
        # from /v1/models. HF_HUB_OFFLINE makes a Hub fetch fail instead of
        # drifting to main or executing remote code from the Hub.
        return docker_exec(
            container,
            [
            "python3",
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model",
            "/model",
            "--tokenizer",
            "/model",
            "--dataset-name",
            "random",
            "--dataset-path",
            SGLANG_DATASET_CONTAINER_PATH,
            "--tokenize-prompt",
            "--num-prompts",
            str(cfg.speed.num_prompts),
            "--random-input-len",
            str(cell.input_len),
            "--random-output-len",
            str(cell.output_len),
            "--random-range-ratio",
            "1.0",
            "--seed",
            str(cfg.seed),
            "--max-concurrency",
            str(cell.concurrency),
            "--output-details",
            "--output-file",
            f"{result_dir}/bench.json",
            ],
        )

    def kv_report(self, log_text: str) -> dict[str, Any]:
        return _kv_report(
            log_text,
            type_line=r"Using KV cache dtype: (\S+)",
            tokens=_SGLANG_KV_TOKENS,
            scale_one=r"Defaulting to scaling factors of 1\.0",
        )

    def metrics_path(self) -> str:
        return "/metrics"

    def watched_metrics(self) -> tuple[str, ...]:
        return SGLANG_WATCHED

    def metrics_note(self) -> str:
        return (
            "Watched names are SGLang counters. A name absent from /metrics is null. "
            "vLLM counter names are not copied in to fill the gap. Other series, when "
            "the scrape has them, stay under counters."
        )

    def serving_metrics(self) -> tuple[str, ...]:
        # Each request's time in prefill, which SGLang builds after v0.5.10 report per stage (dev-lfm3 does).
        return SGLANG_WATCHED + ("sglang:num_running_reqs", SGLANG_PREFILL_SECONDS)

    def serving_parts(self, change: dict[str, float], seconds: float) -> dict[str, tuple[float | None, float | None]]:
        intervals, spent = change.get("sglang:inter_token_latency_seconds_count"), change.get("sglang:inter_token_latency_seconds_sum")
        return {
            # Prompt tokens over the time requests spent in prefill, as for vLLM. SGLang v0.5.10 reports
            # no prefill time, so its prefill rate is not measured.
            "prefill_tok_s": (change.get("sglang:prompt_tokens_total"), change.get(SGLANG_PREFILL_SECONDS)),
            # Per stream: the tokens between intervals over the time they took.
            "decode_tok_s": (intervals, spent),
            "itl_ms": (spent, intervals),
            "generated_tok_s_all": (change.get("sglang:generation_tokens_total"), seconds),
        }

    def serving_rates(self, change: dict[str, float], mean: dict[str, float], seconds: float) -> dict[str, float | None]:
        return {
            **rates_from_parts([self.serving_parts(change, seconds)]),
            "ttft_ms": _ratio(change.get("sglang:time_to_first_token_seconds_sum"), change.get("sglang:time_to_first_token_seconds_count"), 1000),
            "prompt_tok_s_all": _ratio(change.get("sglang:prompt_tokens_total"), seconds),
            "in_flight": mean.get("sglang:num_running_reqs"),
        }

    def logprob_positions(
        self,
        *,
        base_url: str,
        served_name: str,
        token_ids: list[int],
        top_k: int,
        seed: int,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        del served_name, seed
        from sma.httpapi import request_json
        from sma.probe import parse_sglang_input_logprobs

        body = request_json(
            "POST",
            base_url.rstrip("/") + "/generate",
            timeout_s=timeout_s,
            payload={
                "input_ids": token_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": top_k,
            },
        )
        if not isinstance(body, dict):
            raise SmaError("SGLang /generate did not return a JSON object. Logprobs were not invented.")
        return parse_sglang_input_logprobs(body, token_ids, top_k)

    def deterministic(self, cfg: Config) -> dict[str, Any]:
        attention = cfg.engine.sglang_attention_backend
        chosen = cfg.engine.sglang_determinism_backend
        # On SM12x attention is FlashInfer, and that determinism path turns the radix cache off.
        radix = False if attention == "flashinfer" or chosen == "flashinfer" else None
        return {
            "prefix_cache": False,
            "attention_backend": attention,
            "determinism_backend": chosen,
            "radix_cache": radix,
        }


class LlamaCppBackend:
    """llama-server for GGUF files. Logprobs come from llama-perplexity instead (see pipeline)."""

    name = "llamacpp"
    kv_cache_type = "q8_0"
    entrypoint = "/app/llama-server"

    def serve_argv(
        self,
        cfg: Config,
        *,
        mode: str,
        served_name: str,
        is_moe: bool,
        quant_flag: str,
        moe_backend: str | None,
        batch_invariant: bool,
        plan: ServePlan,
        tool_parser: str | None = None,
    ) -> list[str]:
        del is_moe, quant_flag, moe_backend, batch_invariant, tool_parser
        if not plan.model_file:
            raise SmaError("llama.cpp needs the GGUF file to load. The fetch step records it; re-run from fetch.")
        flags = [
            "-m",
            f"/model/{plan.model_file}",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--alias",
            served_name,
            "--seed",
            str(cfg.seed),
            "-ngl",
            "all",
            # Q8_0 is llama.cpp's 8-bit KV cache. A quantized V cache needs flash attention.
            "-fa",
            "on",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            # One unified KV pool of pool_tokens, shared by the slots, each capped at the context.
            "-c",
            str(plan.pool_tokens),
            "-np",
            str(plan.slots),
            "--kv-unified",
            "--kv-unified-per-slot",
            str(plan.context),
            # The prompt cache in host RAM, set explicitly because the memory plan counts it.
            "--cache-ram",
            str(cfg.engine.llamacpp_cache_ram_mib),
            "--metrics",
            "--no-webui",
            # Tool calls use the model's own chat template.
            "--jinja",
            # Level 4 prints the KV cache line ("K (q8_0) ... V (q8_0)") that confirms the cache type.
            "-lv",
            "4",
        ]
        if mode == "quality" and cfg.quality.chat_template_kwargs:
            flags.extend(["--chat-template-kwargs", json.dumps(cfg.quality.chat_template_kwargs, sort_keys=True)])
        flags.extend(cfg.engine.extra_args)
        return flags

    def serve_env(
        self, cfg: Config, *, mode: str, batch_invariant: bool, is_moe: bool = False, quant_flag: str = ""
    ) -> dict[str, str]:
        del cfg, mode, batch_invariant, is_moe, quant_flag
        return {}

    def health_paths(self) -> tuple[str, ...]:
        return ("/health", "/v1/models")

    def bench_argv(
        self,
        cfg: Config,
        *,
        container: str,
        served_name: str,
        cell: Any,
        result_dir: str,
        mounts: BenchMounts | None = None,
    ) -> list[str]:
        """vLLM's bench client in its own container on the host network, as for vLLM, against llama-server."""
        if mounts is None:
            raise SmaError("The llama.cpp bench runs in the vLLM image and needs its mounts.")
        bench_container = container.rsplit("-", 1)[0] + "-bench"
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            bench_container,
            "--network",
            "host",
            "--user",
            f"{mounts.uid}:{mounts.gid}",
            "-e",
            "HOME=/tmp",
        ]
        for key, value in BENCH_OFFLINE_ENV.items():
            argv.extend(["-e", f"{key}={value}"])
        argv.extend(["-v", f"{mounts.model_dir}:/model:ro", "-v", f"{mounts.run_dir}:/sma-run:rw"])
        if mounts.tokenizer_dir:
            argv.extend(["-v", f"{mounts.tokenizer_dir}:{BENCH_TOKENIZER_DIR}:ro"])
        argv.append(mounts.image)
        argv.extend(
            vllm_bench_command(
                cfg,
                served_name=served_name,
                cell=cell,
                result_dir=result_dir,
                base_url=f"http://127.0.0.1:{cfg.engine.port}",
                tokenizer=mounts.tokenizer,
            )
        )
        return argv

    def kv_report(self, log_text: str) -> dict[str, Any]:
        report = _kv_report(log_text, type_line=r"llama_kv_cache: size = .*K \((\w+)\).*V \((\w+)\)", tokens=_LLAMACPP_KV_TOKENS)
        return report

    def metrics_path(self) -> str:
        return "/metrics"

    def watched_metrics(self) -> tuple[str, ...]:
        return LLAMACPP_WATCHED

    def metrics_note(self) -> str:
        return (
            "Watched names are llama-server counters (it runs with --metrics). A name absent from /metrics is "
            "null. vLLM counter names are not copied in."
        )

    def serving_metrics(self) -> tuple[str, ...]:
        return LLAMACPP_WATCHED + ("llamacpp:n_busy_slots_per_decode",)

    def serving_parts(self, change: dict[str, float], seconds: float) -> dict[str, tuple[float | None, float | None]]:
        # Prompt and generation seconds are summed over the slots, so these rates are per request.
        tokens, spent = change.get("llamacpp:tokens_predicted_total"), change.get("llamacpp:tokens_predicted_seconds_total")
        return {
            "prefill_tok_s": (change.get("llamacpp:prompt_tokens_total"), change.get("llamacpp:prompt_seconds_total")),
            "decode_tok_s": (tokens, spent),
            "itl_ms": (spent, tokens),
            "generated_tok_s_all": (tokens, seconds),
        }

    def serving_rates(self, change: dict[str, float], mean: dict[str, float], seconds: float) -> dict[str, float | None]:
        return {
            **rates_from_parts([self.serving_parts(change, seconds)]),
            # llama-server reports no time to first token and no request count to estimate it from.
            "ttft_ms": None,
            "prompt_tok_s_all": _ratio(change.get("llamacpp:prompt_tokens_total"), seconds),
            "in_flight": mean.get("llamacpp:requests_processing"),
        }

    def logprob_positions(
        self,
        *,
        base_url: str,
        served_name: str,
        token_ids: list[int],
        top_k: int,
        seed: int,
        timeout_s: float,
    ) -> list[dict[str, Any]]:
        raise SmaError(
            "llama-server returns no prompt logprobs. SMA scores llama.cpp runs with llama-perplexity instead."
        )

    def deterministic(self, cfg: Config) -> dict[str, Any]:
        del cfg
        # llama.cpp has no batch-invariant mode, and llama-server caches prompts by default.
        return {"prefix_cache": True, "attention_backend": "flash_attn", "determinism_backend": None, "radix_cache": None}


# The engine's own startup lines: the KV type it chose and the pool it allocated.
_VLLM_KV_TOKENS = r"GPU KV cache size: ([\d,]+) tokens"
_SGLANG_KV_TOKENS = r"KV Cache is allocated\. #tokens: ([\d,]+)"
_LLAMACPP_KV_TOKENS = r"llama_kv_cache: size = .*?\((\d+) cells"

LLAMACPP_WATCHED = (
    "llamacpp:prompt_tokens_total",
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:prompt_tokens_cached_total",
    "llamacpp:n_decode_total",
    "llamacpp:requests_processing",
    "llamacpp:requests_deferred",
)


def sglang_mem_fraction(cfg: Config, plan: ServePlan, memory: tuple[int, int] | None = None) -> float:
    """SGLang's --mem-fraction-static: the share of the device its weights and KV pool may hold.

    SGLang sizes its pool to leave total x (1 - fraction) free, and --max-total-tokens only caps it in tokens,
    so with 0.9 a model whose tokens cost more than planned filled the Spark (LFM3-1.7B, 2026-10-30). The
    fraction is what is already in use plus the plan's weights and pool, so the pool is the plan's bytes
    whatever SGLang thinks a token costs; never above engine.gpu_memory_utilization. memory is
    (available, total) from /proc/meminfo, the GB10's one pool; without it, or before a plan, that cap.
    """
    from sma.telemetry import meminfo

    cap = cfg.engine.gpu_memory_utilization
    if not isinstance(plan.pool_bytes, int) or plan.weights_bytes is None:
        return cap
    found = memory or meminfo()
    if found is None:
        return cap
    available, total = found
    return min(cap, (total - available + plan.weights_bytes + plan.pool_bytes) / total)


def rates_from_parts(parts: list[dict[str, tuple[float | None, float | None]]]) -> dict[str, float | None]:
    """Serving rates from each engine's (numerator, denominator) parts, summed first: tokens over seconds
    across every engine that served, so a window that spans two engines is one honest average. A part an
    engine does not report adds nothing (SGLang's prefill time). ITL is in milliseconds."""
    sums: dict[str, list[float]] = {}
    for one in parts:
        for key, (top, bottom) in one.items():
            entry = sums.setdefault(key, [0.0, 0.0])
            if top is not None and bottom is not None:
                entry[0] += top
                entry[1] += bottom
    return {key: _ratio(top, bottom, 1000 if key == "itl_ms" else 1) for key, (top, bottom) in sums.items()}


def _ratio(top: float | None, bottom: float | None, scale: float = 1.0) -> float | None:
    """top / bottom, or None when either is missing or bottom is not positive. Never a filled-in zero."""
    if top is None or bottom is None or bottom <= 0:
        return None
    return scale * top / bottom


def _kv_report(log_text: str, *, type_line: str, tokens: str, scale_one: str | None = None) -> dict[str, Any]:
    """The KV cache type line, the pool size in tokens, and whether the engine says it used a KV
    scale of 1.0, from an engine log. None where absent."""
    line = None
    kv_type = None
    for raw in log_text.splitlines():
        found = re.search(type_line, raw)
        if found:
            line = raw.strip()[-300:]
            kv_type = ",".join(group for group in found.groups() if group)
    counts = re.findall(tokens, log_text)
    pool_tokens = int(counts[-1].replace(",", "")) if counts else None
    scale = None if scale_one is None else bool(re.search(scale_one, log_text))
    return {"kv_type": kv_type, "kv_type_line": line, "pool_tokens": pool_tokens, "scale_1_0": scale}


_BACKENDS: dict[str, EngineBackend] = {
    "vllm": VllmBackend(),
    "sglang": SglangBackend(),
    "llamacpp": LlamaCppBackend(),
}


def get_backend(name: str) -> EngineBackend:
    backend = _BACKENDS.get(name)
    if backend is None:
        raise SmaError(f"Unknown engine {name!r}. SMA supports vllm (default), sglang, and llamacpp.")
    return backend
