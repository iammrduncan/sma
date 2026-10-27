"""Small backend interface. vLLM is the default. SGLang is opt-in.

Each engine owns its serve command, health check, logprobs, bench command,
metrics names, and deterministic-mode flags. A measurement the engine does
not return stays null; nothing is copied from the other engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from sma.config import Config
from sma.errors import SmaError


class EngineBackend(Protocol):
    name: str

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
        tool_parser: str | None = None,
    ) -> list[str]: ...

    def serve_env(self, cfg: Config, *, mode: str, batch_invariant: bool) -> dict[str, str]: ...

    def health_paths(self) -> tuple[str, ...]: ...

    def bench_argv(
        self, cfg: Config, *, container: str, served_name: str, cell: Any, result_dir: str
    ) -> list[str]: ...

    def metrics_path(self) -> str: ...

    def watched_metrics(self) -> tuple[str, ...]: ...

    def metrics_note(self) -> str: ...

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
            *(
                ["--kv-cache-memory-bytes", str(int(cfg.engine.kv_cache_gib * 2**30))]
                if cfg.engine.kv_cache_gib is not None
                else []
            ),
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
        if quality:
            # The determinism probe shares this server. A prefix-cache hit
            # would hide batch-size differences.
            flags.append("--no-enable-prefix-caching")
        if quality and batch_invariant and cfg.engine.no_async_scheduling_with_batch_invariant:
            flags.append("--no-async-scheduling")
        if is_moe and quant_flag in {"nvfp4", "mxfp4", "mixed"} and moe_backend:
            flags.extend(["--moe-backend", moe_backend])
        if (not is_moe) and quant_flag == "nvfp4" and cfg.engine.linear_backend:
            flags.extend(["--linear-backend", cfg.engine.linear_backend])
        if quality and cfg.quality.chat_template_kwargs:
            # The default for every quality client, including harnesses that cannot add it to a
            # request (BFCL). A request's own chat_template_kwargs still wins.
            flags.extend(
                ["--default-chat-template-kwargs", json.dumps(cfg.quality.chat_template_kwargs, sort_keys=True)]
            )
        if quality and tool_parser:
            flags.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])
        if cfg.engine.trust_remote_code:
            flags.append("--trust-remote-code")
        flags.extend(cfg.engine.extra_args)
        return flags

    def serve_env(self, cfg: Config, *, mode: str, batch_invariant: bool) -> dict[str, str]:
        del cfg
        if mode == "quality" and batch_invariant:
            return {"VLLM_BATCH_INVARIANT": "1"}
        return {}

    def health_paths(self) -> tuple[str, ...]:
        return ("/health", "/v1/models")

    def bench_argv(
        self, cfg: Config, *, container: str, served_name: str, cell: Any, result_dir: str
    ) -> list[str]:
        # --model is the served API name. --tokenizer /model is the pinned snapshot.
        # Without --tokenizer, vLLM sets tokenizer_id from the repo id and pulls main.
        return docker_exec(
            container,
            [
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
            "--tokenizer",
            "/model",
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
            ],
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


class SglangBackend:
    name = "sglang"

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
            f"{cfg.engine.gpu_memory_utilization:.4g}",
            "--context-length",
            str(cfg.engine.max_model_len),
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
        if cfg.engine.trust_remote_code:
            flags.append("--trust-remote-code")
        flags.extend(cfg.engine.extra_args)
        return flags

    def serve_env(self, cfg: Config, *, mode: str, batch_invariant: bool) -> dict[str, str]:
        del cfg, batch_invariant
        if mode == "quality":
            return {"SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1"}
        return {}

    def health_paths(self) -> tuple[str, ...]:
        return ("/health", "/v1/models")

    def bench_argv(
        self, cfg: Config, *, container: str, served_name: str, cell: Any, result_dir: str
    ) -> list[str]:
        del served_name
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


_BACKENDS: dict[str, EngineBackend] = {
    "vllm": VllmBackend(),
    "sglang": SglangBackend(),
}


def get_backend(name: str) -> EngineBackend:
    backend = _BACKENDS.get(name)
    if backend is None:
        raise SmaError(f"Unknown engine {name!r}. SMA supports vllm (default) and sglang.")
    return backend
