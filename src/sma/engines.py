"""vLLM is the default engine. SGLang is opt-in and labeled as a different subject."""

from __future__ import annotations

from sma.config import Config
from sma.errors import SmaError
from sma.quant import DetectedQuant

ENGINES = ("vllm", "sglang")
# Versioned multi-arch CUDA 13 tag used for DGX Spark (arm64). `latest-cu130` moves.
DEFAULT_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.10.post1-cu130"


def engine_name(value: str | None, cfg: Config) -> str:
    name = (value or cfg.engine.name or "vllm").strip().lower()
    if name not in ENGINES:
        raise SmaError(f"Unknown engine {name!r}. SMA supports vllm (default) and sglang.")
    return name


def image_for(cfg: Config, engine: str) -> str:
    if engine == "sglang":
        return cfg.engine.sglang_image
    return cfg.engine.image


def configured_digest(cfg: Config, engine: str) -> str | None:
    if engine == "sglang":
        return cfg.engine.sglang_image_digest
    return cfg.engine.image_digest


def image_ref(image: str, digest: str | None = None) -> str:
    """Prefer a resolved digest so later starts use the identical image."""
    if not digest:
        return image
    base = image.split("@", 1)[0]
    token = digest
    if "@" in token:
        token = token.split("@", 1)[1]
    if not token.startswith("sha256:"):
        token = "sha256:" + token
    return f"{base}@{token}"


def compatibility_error(engine: str, detected: DetectedQuant) -> str | None:
    """Refuse quant/engine pairs SMA cannot honestly serve. SMA never converts them."""
    fmt = detected.format
    method = str((detected.sources or {}).get("quant_method") or "").lower()
    if engine == "vllm":
        if fmt == "mixed":
            return (
                "vLLM cannot load this mixed checkpoint "
                f"({detected.detail}). SMA will not rewrite it. "
                "Mixed FP8/NVFP4 checkpoints such as quant_method dgpp_mixed need "
                "--mixed and an engine that implements that format. "
                "Stock vLLM does not. --engine sglang is opt-in and still refuses "
                "architectures its image does not register."
            )
        if fmt in {"bf16", "nvfp4", "nvfp4_w4a16", "mxfp4", "fp8"}:
            return None
        if fmt in {"fp16", "fp32", "unknown", "other", ""}:
            return None
        return f"vLLM support for quant format {fmt} is not implemented in SMA."
    if fmt == "nvfp4" and "compressed" in method:
        return (
            "SGLang in SMA accepts NVFP4 from ModelOpt checkpoints, not compressed-tensors. "
            f"This checkpoint is {detected.detail}. Re-run with --engine vllm."
        )
    if fmt == "nvfp4_w4a16":
        return (
            "This NVFP4 checkpoint is weight-only W4A16. "
            "SMA serves that on vLLM only. Re-run with --engine vllm."
        )
    if fmt in {"bf16", "nvfp4", "mxfp4", "fp8", "mixed"}:
        return None
    if fmt in {"fp16", "fp32", "unknown", "other", ""}:
        return None
    return f"SGLang support for quant format {fmt} is not implemented in SMA."
