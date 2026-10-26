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
    image: str = DEFAULT_IMAGE
    image_digest: str | None = None
    port: int = 8000
    bind_host: str = "127.0.0.1"
    gpus: str = "all"
    shm_size: str = "16g"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    max_num_seqs: int = 64
    moe_backend: str = "flashinfer_b12x"
    moe_backend_fallbacks: list[str] = field(default_factory=lambda: ["marlin"])
    linear_backend: str = "flashinfer_b12x"
    batch_invariant: bool = True
    no_flashinfer_autotune_quality: bool = True
    no_async_scheduling_with_batch_invariant: bool = True
    trust_remote_code: bool = False
    extra_args: list[str] = field(default_factory=list)
    server_stop_timeout_s: int = 30
    clock_cap_mhz: int | None = None
    apply_clock_cap: bool = False
    container_prefix: str = "sma"


@dataclass
class TimeoutConfig:
    server_start_s: int = 900
    health_request_s: float = 5
    health_poll_s: float = 2
    command_s: int = 120
    bench_s: int = 7200
    eval_s: int = 14400
    docker_s: int = 300
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
    num_windows: int = 4
    corpus_version: str = "sma-corpus-v1"


@dataclass
class QualitySuite:
    name: str
    limit: int | None = None
    tasks: str | None = None


@dataclass
class QualityConfig:
    enabled: bool = True
    concurrency: int = 8
    model_backend: str = "local-chat-completions"
    executable: str = "lm_eval"
    suites: list[QualitySuite] = field(
        default_factory=lambda: [
            QualitySuite("ifeval", 50),
            QualitySuite("gsm8k", 50),
            QualitySuite("mmlu_pro", 50),
            QualitySuite("gpqa_diamond", 20),
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


@dataclass
class QueueConfig:
    dir: str = "~/.local/share/sma/queue"
    poll_s: float = 5.0
    stale_claim_s: float = 3600.0
    heartbeat_s: float = 30.0


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


def _build(cls: type, raw: dict[str, Any] | None, path: str) -> Any:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SmaError(f"{path} must be a mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(raw) - known)
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


def _suites(value: Any, path: str) -> list[QualitySuite]:
    if not isinstance(value, list) or not value:
        raise SmaError(f"{path} must be a non-empty list")
    suites: list[QualitySuite] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or "name" not in item:
            raise SmaError(f"{path}[{index}] needs a name")
        unknown = set(item) - {"name", "limit", "tasks"}
        if unknown:
            raise SmaError(f"{path}[{index}] has unknown keys: {sorted(unknown)}")
        limit = item.get("limit")
        suites.append(
            QualitySuite(
                name=str(item["name"]),
                limit=None if limit is None else int(limit),
                tasks=item.get("tasks"),
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
    if cfg.engine.max_model_len < cfg.logprobs.window_tokens:
        raise SmaError(
            "engine.max_model_len must be at least logprobs.window_tokens "
            "so a full window fits"
        )
    if cfg.engine.gpu_memory_utilization <= 0 or cfg.engine.gpu_memory_utilization > 1:
        raise SmaError("engine.gpu_memory_utilization must be in (0, 1]")
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
