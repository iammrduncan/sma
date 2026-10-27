"""Load sma.yaml and reject settings that would change the engine contract."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from sma.errors import SmaError

# The Spark playbook's current NGC tag. The digest is recorded at pull time
# and is never invented here. Pin engine.image_digest in sma.yaml once known.
DEFAULT_IMAGE = "nvcr.io/nvidia/vllm:26.05.post1-py3"

_DEFAULT_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "def add(a, b):\n    return",
]


@dataclass
class HfConfig:
    cache_dir: str = "~/.cache/sma/hf"
    token_env: str = "HF_TOKEN"
    download_timeout_s: int = 3600


@dataclass
class EngineConfig:
    name: str = "vllm"
    image: str = DEFAULT_IMAGE
    image_digest: str | None = None
    # Versioned SGLang tag for DGX Spark (arm64, CUDA 13). Digest is recorded at pull.
    sglang_image: str = "lmsysorg/sglang:v0.5.10.post1-cu130"
    sglang_image_digest: str | None = None
    # SM12x requires FlashInfer attention. Deterministic quality mode disables the radix cache.
    sglang_attention_backend: str = "flashinfer"
    sglang_determinism_backend: str = "flashinfer"
    # DeepGEMM in the pinned SGLang image does not support the GB10 (SM121): deterministic serving
    # crashes at CUDA-graph capture with "Unsupported architecture". SMA turns it off.
    sglang_disable_deep_gemm: bool = True
    # llama.cpp runs GGUF files. The "full" image has llama-server and llama-perplexity, arm64 + CUDA 13,
    # built for SM 12.1. Versioned tag, digest recorded at pull.
    llamacpp_image: str = "ghcr.io/ggml-org/llama.cpp:full-cuda13-b11913"
    llamacpp_image_digest: str | None = None
    # llama-server's host-RAM prompt cache, in MiB (--cache-ram). 8192 is llama.cpp's own default; SMA
    # passes it explicitly and counts it in the memory plan. 0 turns it off.
    llamacpp_cache_ram_mib: int = 8192
    port: int = 8000
    bind_host: str = "127.0.0.1"
    gpus: str = "all"
    shm_size: str = "16g"
    # vLLM's startup free-memory check and SGLang's static memory share. The memory plan sizes the KV
    # pool itself (see memory below), so this no longer decides how much KV cache the engine takes.
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 64
    moe_backend: str = "flashinfer_b12x"
    moe_backend_fallbacks: list[str] = field(default_factory=lambda: ["marlin"])
    linear_backend: str = "flashinfer_b12x"
    batch_invariant: bool = True
    no_flashinfer_autotune_quality: bool = True
    no_async_scheduling_with_batch_invariant: bool = True
    trust_remote_code: bool = False
    # The engine's parser for the model's tool calls, for example hermes for Qwen3. It depends on
    # the model, so `--tool-parser` on the command line sets it per run. null serves no tools.
    tool_call_parser: str | None = None
    extra_args: list[str] = field(default_factory=list)
    server_stop_timeout_s: int = 30
    clock_cap_mhz: int | None = None
    apply_clock_cap: bool = False
    container_prefix: str = "sma"


@dataclass
class MemoryConfig:
    """The memory plan. See docs/serving-design.md."""

    # A run's planned footprint (weights, KV pool, engine overhead) stays under this many GB (1e9).
    # The Spark has 130.7 GB; the margin keeps earlyoom's 8% floor away.
    budget_gb: float = 100.0
    # Context per stream starts here, or at the model's native maximum if smaller, and halves to fit.
    start_context: int = 262_144
    # Run memory beyond weights and KV pool, in GB: the engine's activations, CUDA graphs, and server
    # processes, plus SMA's tools beside it (nel, the bench client). Measured per engine on the Spark
    # (docs/first-spark-run.md).
    overhead_gb: dict[str, float] = field(default_factory=lambda: {"vllm": 10.0, "sglang": 10.0, "llamacpp": 3.0})
    # Engine memory that grows with the KV pool, as a share of it. llama.cpp's buffers sized by its
    # context measured 0.9 GB plus 11.7% of the pool on a Spark (Qwen3-0.6B, -c 372,578 and 1,490,313).
    pool_overhead_fraction: dict[str, float] = field(default_factory=lambda: {"vllm": 0.0, "sglang": 0.0, "llamacpp": 0.12})
    # The least context per stream a quality request needs: prompt plus quality.max_tokens.
    quality_min_context: int = 8192


@dataclass
class TimeoutConfig:
    server_start_s: int = 900
    health_request_s: float = 5
    health_poll_s: float = 2
    command_s: int = 120
    bench_s: int = 7200
    eval_s: int = 14400
    docker_s: int = 300
    # Image pulls are separate from short docker inspect calls. A cold
    # arm64 CUDA image is large, so this timeout is intentionally long.
    image_pull_s: int = 3600
    metrics_s: float = 10
    logprob_request_s: float = 600


@dataclass
class RetryConfig:
    attempts: int = 3
    backoff_s: float = 2.0
    backoff_factor: float = 2.0


@dataclass
class DeterminismConfig:
    prompts: list[str] = field(default_factory=lambda: list(_DEFAULT_PROMPTS))
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 64])
    max_tokens: int = 32
    temperature: float = 0.0


@dataclass
class LogprobConfig:
    top_k: int = 64
    window_tokens: int = 2048
    num_windows: int = 48
    corpus_version: str = "sma-corpus-v3"
    # llama.cpp only. llama-server returns no prompt logprobs, so llama-perplexity scores this many
    # chunks of window_tokens and writes the reference's full-vocabulary logits for KL. That file is
    # about chunks * window_tokens / 2 * vocab * 2 bytes (2.5 GB for 8 chunks of a 152K vocabulary).
    llamacpp_chunks: int = 8


@dataclass
class QualitySuite:
    name: str
    # Questions to run, picked by a seeded shuffle of the whole dataset. None runs all of them.
    limit: int | None = None
    # The NeMo Evaluator benchmark: a built-in name, lm-eval://<task>, or sma:<name> for a file in
    # src/sma/nel_benchmarks/. None means the suite name.
    bench: str | None = None
    # An NVIDIA eval-factory container and its task, run directly instead of through nel, for
    # harnesses nel only runs in a temporary folder (BFCL). Both or neither.
    image: str | None = None
    task: str | None = None
    # The suite calls tools through the chat API, so the server needs a tool-call parser.
    tools: bool = False


@dataclass
class QualityConfig:
    enabled: bool = True
    # Requests in flight per suite (nel max_concurrent).
    concurrency: int = 8
    executable: str = "nel"
    max_tokens: int = 4096
    # One request to the model server. A retry follows retries.attempts.
    request_timeout_s: float = 900.0
    # Sent with every chat request. enable_thinking=False turns off the reasoning block in chat
    # templates that read it (Qwen3 and others). SMA scores greedily with an 8192-token context;
    # Qwen says not to decode thinking mode greedily and to allow 32,768 output tokens.
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    suites: list[QualitySuite] = field(
        default_factory=lambda: [
            QualitySuite("ifeval", 50, bench="lm-eval://ifeval"),
            # Both of nel 0.3.0's GSM8K graders mark answers wrongly. See nel_benchmarks/gsm8k.py.
            QualitySuite("gsm8k", 50, bench="sma:gsm8k"),
            QualitySuite("mmlu_pro", 700),
            QualitySuite("gpqa_diamond", 20, bench="gpqa"),
            # Native tool calls through the chat API: needs --tool-parser.
            QualitySuite("bfcl_v3", image="nvcr.io/nvidia/eval-factory/bfcl:26.03", task="bfclv3_ast", tools=True),
        ]
    )


@dataclass
class SpeedShape:
    name: str
    input_len: int
    output_len: int


@dataclass
class SpeedConfig:
    warmup_requests: int = 4
    warmup_max_tokens: int = 8
    warmup_soak_s: float = 30.0
    num_prompts: int = 32
    # Arena plan asks for 3 repeats. v0 defaults to 1; raise this in sma.yaml.
    repeats: int = 1
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,99"
    derive_stat: str = "p50"
    throttle_retries: int = 1
    prefill: SpeedShape = field(
        default_factory=lambda: SpeedShape("prefill", 2048, 64)
    )
    decode: SpeedShape = field(
        default_factory=lambda: SpeedShape("decode", 128, 512)
    )
    sweep: SpeedShape = field(
        default_factory=lambda: SpeedShape("sweep", 1024, 256)
    )
    sweep_concurrency: list[int] = field(default_factory=lambda: [1, 4, 8])


@dataclass
class TelemetryConfig:
    interval_s: float = 1.0
    idle_until_cool_c: float = 55.0
    idle_timeout_s: float = 600.0
    poll_s: float = 5.0
    drop_page_cache: bool = True
    cool_between_legs: bool = True
    # DGX Spark spec constants used only to label a bandwidth ceiling.
    # They are not measurements.
    spec_bandwidth_gbps: float = 273.0
    spec_usable_gb: float = 115.0
    # GB10 sits near a 140 W cap, so a power-cap flag can be steady-state rather than a slowdown.
    # The raw flag is always stored. It invalidates a speed cell only when this is true.
    sw_power_cap_invalidates: bool = False
    max_sample_gap_s: float = 2.5
    # A speed cell needs about this fraction of duration/interval samples,
    # and no gap larger than max_sample_gap_s, or it is unverified.
    min_sample_ratio: float = 0.8


@dataclass
class QueueConfig:
    dir: str = "~/.local/share/sma/queue"
    poll_s: float = 5.0
    stale_claim_s: float = 3600.0
    heartbeat_s: float = 30.0
    claim_grace_s: float = 30.0
    max_attempts: int = 1
    # After a failure the job waits, then sorts behind jobs that have not failed.
    retry_backoff_s: float = 30.0
    retry_backoff_factor: float = 2.0


@dataclass
class DoctorConfig:
    min_free_gb: float = 20.0
    gpu_index: int = 0


@dataclass
class Config:
    schema_version: int = 1
    host_alias: str = "gpu-host-1"
    runs_dir: str = "runs"
    seed: int = 0
    lock_path: str = "~/.local/share/sma/locks/gpu.lock"
    hf: HfConfig = field(default_factory=HfConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)
    logprobs: LogprobConfig = field(default_factory=LogprobConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    doctor: DoctorConfig = field(default_factory=DoctorConfig)

    def runs_path(self) -> Path:
        return Path(self.runs_dir).expanduser().resolve()

    def lock_file(self) -> Path:
        return Path(self.lock_path).expanduser()

    def queue_path(self) -> Path:
        return Path(self.queue.dir).expanduser()

    def cache_path(self) -> Path:
        return Path(self.hf.cache_dir).expanduser()

    def to_snapshot(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> Config:
    return Config()


_NESTED = {
    "hf": HfConfig,
    "engine": EngineConfig,
    "memory": MemoryConfig,
    "timeouts": TimeoutConfig,
    "retries": RetryConfig,
    "determinism": DeterminismConfig,
    "logprobs": LogprobConfig,
    "quality": QualityConfig,
    "speed": SpeedConfig,
    "telemetry": TelemetryConfig,
    "queue": QueueConfig,
    "doctor": DoctorConfig,
}


# Keys an older sma.yaml may still carry, and what replaced them.
_REPLACED = {
    "EngineConfig": {
        "max_model_len": "The memory plan sets the context per stream (memory.start_context, stepped down to fit). "
        "Remove it from sma.yaml.",
        "kv_cache_gib": "The memory plan sizes the KV pool from memory.budget_gb. Remove it from sma.yaml.",
    }
}


def _build(cls: type, raw: dict[str, Any] | None, path: str) -> Any:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SmaError(f"{path} must be a mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - known)
    replaced = [key for key in unknown if key in _REPLACED.get(cls.__name__, {})]
    if replaced:
        raise SmaError(f"{path}.{replaced[0]} is gone. {_REPLACED[cls.__name__][replaced[0]]}")
    if unknown:
        raise SmaError(f"Unknown config keys at {path}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in raw:
            continue
        value = raw[item.name]
        if item.name == "suites":
            kwargs[item.name] = _suites(value, f"{path}.suites")
        elif item.name in {"prefill", "decode", "sweep"}:
            kwargs[item.name] = _shape(value, f"{path}.{item.name}")
        elif item.name in _NESTED and item.name != path:
            kwargs[item.name] = _build(_NESTED[item.name], value, f"{path}.{item.name}")
        else:
            kwargs[item.name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise SmaError(f"Invalid config at {path}: {exc}") from exc


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _suites(value: Any, path: str) -> list[QualitySuite]:
    if not isinstance(value, list) or not value:
        raise SmaError(f"{path} must be a non-empty list")
    suites: list[QualitySuite] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or "name" not in item:
            raise SmaError(f"{path}[{index}] needs a name")
        unknown = set(item) - {"name", "limit", "bench", "image", "task", "tools"}
        if unknown:
            raise SmaError(f"{path}[{index}] has unknown keys: {sorted(unknown)}")
        limit = item.get("limit")
        suites.append(
            QualitySuite(
                name=str(item["name"]),
                limit=None if limit is None else int(limit),
                bench=_optional_str(item.get("bench")),
                image=_optional_str(item.get("image")),
                task=_optional_str(item.get("task")),
                tools=bool(item.get("tools", False)),
            )
        )
    return suites


def _shape(value: Any, path: str) -> SpeedShape:
    if not isinstance(value, dict):
        raise SmaError(f"{path} must be a mapping")
    unknown = set(value) - {"name", "input_len", "output_len"}
    if unknown:
        raise SmaError(f"{path} has unknown keys: {sorted(unknown)}")
    try:
        return SpeedShape(
            name=str(value["name"]),
            input_len=int(value["input_len"]),
            output_len=int(value["output_len"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SmaError(f"{path} needs name, input_len, output_len") from exc


def config_from_mapping(raw: dict[str, Any] | None) -> Config:
    return _build(Config, raw or {}, "config")


def load_config(path: Path | None = None) -> Config:
    chosen = path
    if chosen is None:
        for candidate in (Path("sma.yaml"), Path.home() / ".config" / "sma" / "sma.yaml"):
            if candidate.is_file():
                chosen = candidate
                break
    if chosen is None:
        cfg = default_config()
    else:
        try:
            loaded = yaml.safe_load(Path(chosen).read_text(encoding="utf-8"))
        except OSError as exc:
            raise SmaError(f"Could not read config {chosen}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise SmaError(f"Could not parse config {chosen}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise SmaError(f"Config {chosen} must be a mapping")
        cfg = config_from_mapping(loaded)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.engine.max_num_seqs < max(cfg.determinism.batch_sizes):
        raise SmaError(
            "engine.max_num_seqs must be at least the largest determinism batch "
            f"({max(cfg.determinism.batch_sizes)})"
        )
    if cfg.quality.concurrency > cfg.engine.max_num_seqs:
        raise SmaError("quality.concurrency cannot exceed engine.max_num_seqs")
    if cfg.quality.concurrency < 1:
        raise SmaError("quality.concurrency must be >= 1")
    if cfg.logprobs.top_k < 1:
        raise SmaError("logprobs.top_k must be >= 1")
    if cfg.logprobs.window_tokens < 2:
        raise SmaError("logprobs.window_tokens must be >= 2")
    if cfg.memory.quality_min_context < cfg.logprobs.window_tokens:
        raise SmaError(
            "memory.quality_min_context must be at least logprobs.window_tokens so a full window fits"
        )
    if cfg.memory.budget_gb <= 0:
        raise SmaError("memory.budget_gb must be a positive number of GB")
    if cfg.memory.start_context < 1024:
        raise SmaError("memory.start_context must be at least 1024 tokens")
    for name in ("vllm", "sglang", "llamacpp"):
        overhead = cfg.memory.overhead_gb.get(name) if isinstance(cfg.memory.overhead_gb, dict) else None
        if not isinstance(overhead, (int, float)) or isinstance(overhead, bool) or overhead < 0:
            raise SmaError(f"memory.overhead_gb.{name} must be a number of GB, 0 or more")
    for name in ("vllm", "sglang", "llamacpp"):
        share = cfg.memory.pool_overhead_fraction.get(name) if isinstance(cfg.memory.pool_overhead_fraction, dict) else None
        if not isinstance(share, (int, float)) or isinstance(share, bool) or not 0 <= share < 1:
            raise SmaError(f"memory.pool_overhead_fraction.{name} must be a share from 0 to below 1")
    if cfg.engine.llamacpp_cache_ram_mib < 0:
        raise SmaError("engine.llamacpp_cache_ram_mib must be 0 (off) or more")
    if cfg.logprobs.llamacpp_chunks < 1:
        raise SmaError("logprobs.llamacpp_chunks must be >= 1")
    if cfg.engine.gpu_memory_utilization <= 0 or cfg.engine.gpu_memory_utilization > 1:
        raise SmaError("engine.gpu_memory_utilization must be in (0, 1]")
    if not isinstance(cfg.quality.chat_template_kwargs, dict) or not all(
        isinstance(key, str) for key in cfg.quality.chat_template_kwargs
    ):
        raise SmaError("quality.chat_template_kwargs must be a mapping with string keys")
    if not 0 < cfg.quality.max_tokens < cfg.memory.quality_min_context:
        raise SmaError("quality.max_tokens must be positive and below memory.quality_min_context, so the prompt fits")
    if cfg.quality.request_timeout_s <= 0:
        raise SmaError("quality.request_timeout_s must be positive")
    names = [suite.name for suite in cfg.quality.suites]
    if len(set(names)) != len(names):
        raise SmaError("quality.suites names must be unique; each one is a folder under quality/")
    from sma.commands import packaged_bench

    for suite in cfg.quality.suites:
        if suite.limit is not None and suite.limit < 1:
            raise SmaError(f"quality suite {suite.name}: limit must be >= 1, or null for the whole dataset")
        if (suite.image is None) != (suite.task is None):
            raise SmaError(f"quality suite {suite.name}: image and task go together")
        if suite.image is not None and (suite.bench is not None or suite.limit is not None):
            raise SmaError(
                f"quality suite {suite.name}: an eval-factory container takes no bench or limit. "
                "Its limit keeps the first questions in file order, not a seeded sample, so SMA runs them all."
            )
        packaged_bench(suite.bench)  # an unknown sma:<name> fails here, before any download
    if cfg.engine.name not in {"vllm", "sglang"}:
        raise SmaError(
            "engine.name must be vllm or sglang: the default engine for safetensors checkpoints. "
            "llama.cpp is chosen by --gguf, not by this setting."
        )
    if cfg.engine.name == "sglang":
        # Before any pull or download. The speed step repeats this check.
        from sma.backends import assert_sglang_dataset_covers

        assert_sglang_dataset_covers(cfg.speed.num_prompts)
    if cfg.engine.sglang_attention_backend != "flashinfer":
        raise SmaError(
            "engine.sglang_attention_backend must be flashinfer. "
            "SGLang on SM12x requires FlashInfer attention."
        )
    if cfg.engine.sglang_determinism_backend not in {"flashinfer", "fa3", "triton"}:
        raise SmaError("engine.sglang_determinism_backend must be flashinfer, fa3, or triton")
    if cfg.speed.derive_stat not in {"p50", "p90", "p99"}:
        raise SmaError("speed.derive_stat must be p50, p90, or p99")
    if cfg.queue.max_attempts < 1:
        raise SmaError("queue.max_attempts must be >= 1")
    if cfg.queue.stale_claim_s <= cfg.queue.heartbeat_s + cfg.queue.claim_grace_s:
        raise SmaError(
            "queue.stale_claim_s must be longer than heartbeat_s plus claim_grace_s "
            "so a live job is not reclaimed between heartbeats."
        )
    if cfg.speed.repeats < 1:
        raise SmaError("speed.repeats must be >= 1")
    if 1 not in cfg.speed.sweep_concurrency:
        raise SmaError("speed.sweep_concurrency must include 1")
    _reject_forbidden_args(cfg.engine.extra_args)
    if cfg.engine.bind_host not in {"127.0.0.1", "localhost"}:
        # Binding beyond loopback is allowed only when the operator sets it,
        # but it is recorded. No hostname is copied from the machine.
        pass
    if cfg.telemetry.interval_s <= 0:
        raise SmaError("telemetry.interval_s must be > 0")
    if cfg.telemetry.max_sample_gap_s <= 0:
        raise SmaError("telemetry.max_sample_gap_s must be > 0")
    if not 0 < cfg.telemetry.min_sample_ratio <= 1:
        raise SmaError("telemetry.min_sample_ratio must be in (0, 1]")
    if cfg.timeouts.image_pull_s < 1:
        raise SmaError("timeouts.image_pull_s must be >= 1")
    if cfg.queue.retry_backoff_s < 0:
        raise SmaError("queue.retry_backoff_s must be >= 0")


# KV cache type, context, and KV pool flags of vLLM, SGLang, and llama-server.
_PLAN_FLAGS = {
    "--kv-cache-dtype",
    "--max-model-len",
    "--kv-cache-memory-bytes",
    "--context-length",
    "--max-total-tokens",
    "-ctk",
    "-ctv",
    "--cache-type-k",
    "--cache-type-v",
    "-c",
    "--ctx-size",
    "--kv-unified-per-slot",
}


def _reject_forbidden_args(extra: list[str]) -> None:
    index = 0
    while index < len(extra):
        arg = extra[index]
        if arg == "--quantization" or arg.startswith("--quantization="):
            raise SmaError(
                "SMA never quantizes. Remove --quantization and point sma at "
                "an already quantized checkpoint."
            )
        if arg == "--tensor-parallel-size" or arg.startswith("--tensor-parallel-size="):
            if arg.startswith("--tensor-parallel-size="):
                value = arg.split("=", 1)[1]
            else:
                if index + 1 >= len(extra):
                    raise SmaError("--tensor-parallel-size is missing a value")
                value = extra[index + 1]
            if value != "1":
                raise SmaError(
                    "SMA does not use tensor parallelism, including across Sparks. "
                    f"Refusing --tensor-parallel-size {value}."
                )
        if arg in {"--host", "--port"} or arg.startswith(("--host=", "--port=")):
            raise SmaError(f"SMA sets {arg.split('=')[0]} itself. Remove it from extra_args.")
        flag = arg.split("=", 1)[0]
        if flag in _PLAN_FLAGS:
            raise SmaError(
                f"SMA sets {flag} itself: the KV cache is always 8-bit, and the memory plan sizes the context "
                "and the KV pool (docs/serving-design.md). Remove it from extra_args."
            )
        index += 1


def hf_token(cfg: Config) -> str | None:
    primary = os.environ.get(cfg.hf.token_env)
    if primary:
        return primary
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if key == cfg.hf.token_env:
            continue
        value = os.environ.get(key)
        if value:
            return value
    return None
