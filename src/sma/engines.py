"""Which engine serves a checkpoint. The format decides, before launch, and the run records why.

Safetensors go to vLLM, or to SGLang with --engine sglang or when preflight shows vLLM cannot serve the
architecture. GGUF goes to llama.cpp. A failed start is never retried on another engine.
"""

from __future__ import annotations

from sma.config import Config
from sma.errors import SmaError
from sma.quant import DetectedQuant

ENGINES = ("vllm", "sglang", "llamacpp")
# Versioned multi-arch CUDA 13 tag used for DGX Spark (arm64). `latest-cu130` moves.
DEFAULT_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.10.post1-cu130"

# Owner decision 2026-10-27: quantized checkpoints run their real kernels. vLLM's batch-invariant mode
# replaces them (NVFP4 by emulation, FP8 through DeepGEMM or a BF16 dequant), so it stays off for these.
REAL_KERNELS_ONLY = frozenset({"nvfp4", "mxfp4", "mixed", "fp8"})


def batch_invariance_replaces_kernels(engine: str, quant_format: str | None) -> bool:
    """True when batch invariance would swap this checkpoint's quantized kernels, so it must stay off."""
    return engine == "vllm" and quant_format in REAL_KERNELS_ONLY


def engine_name(value: str | None, cfg: Config, quant_flag: str | None = None) -> str:
    """The engine for this checkpoint format. --engine may pick SGLang for safetensors, nothing else."""
    requested = (value or "").strip().lower() or None
    if requested is not None and requested not in ENGINES:
        raise SmaError(f"Unknown engine {requested!r}. SMA supports vllm (default), sglang, and llamacpp.")
    if quant_flag == "gguf":
        if requested not in (None, "llamacpp"):
            raise SmaError(
                f"GGUF files run on llama.cpp, not {requested}. Drop --engine, or pass --engine llamacpp."
            )
        return "llamacpp"
    if requested == "llamacpp":
        raise SmaError(
            "llama.cpp runs GGUF files only, and SMA never converts a checkpoint. "
            "Pass --gguf <quant> with a GGUF repo, or drop --engine llamacpp."
        )
    name = requested or (cfg.engine.name or "vllm").strip().lower()
    if name not in ("vllm", "sglang"):
        raise SmaError(f"Unknown engine {name!r}. SMA supports vllm (default) and sglang for safetensors.")
    return name


def selected_by(value: str | None, quant_flag: str | None) -> str:
    """Why this engine: the checkpoint format, the --engine flag, or the configured default."""
    if quant_flag == "gguf":
        return "format"
    return "flag" if value else "default"


def image_for(cfg: Config, engine: str) -> str:
    if engine == "sglang":
        return cfg.engine.sglang_image
    if engine == "llamacpp":
        return cfg.engine.llamacpp_image
    return cfg.engine.image


def configured_digest(cfg: Config, engine: str) -> str | None:
    if engine == "sglang":
        return cfg.engine.sglang_image_digest
    if engine == "llamacpp":
        return cfg.engine.llamacpp_image_digest
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


def _method_token(detected: DetectedQuant) -> str:
    raw = str((detected.sources or {}).get("quant_method") or "")
    return raw.strip().lower().replace("-", "_")


def compatibility_error(engine: str, detected: DetectedQuant) -> str | None:
    """Refuse quant_method x engine pairs SMA cannot honestly serve.

    The decision is the checkpoint's quant_method, not only the coarse format.
    SMA never converts one format into another.
    """
    fmt = detected.format
    method = _method_token(detected)
    if engine == "llamacpp" or fmt == "gguf":
        if engine == "llamacpp" and fmt == "gguf":
            return None
        return (
            f"{engine} cannot serve a {fmt} checkpoint: GGUF runs on llama.cpp, safetensors on vLLM or SGLang. "
            "SMA never converts one into the other."
        )
    if fmt == "nvfp4_w4a16":
        return (
            "This checkpoint is weight-only NVFP4 (W4A16), not W4A4 NVFP4 "
            f"({detected.detail}). --nvfp4 will not label it nvfp4. "
            "SMA does not serve W4A16."
        )
    if fmt == "nvfp4_w4a8":
        return (
            "This checkpoint is NVFP4 weights with FP8 activations (W4A8), not W4A4 NVFP4 "
            f"({detected.detail}). --nvfp4 will not label it nvfp4. "
            "SMA does not serve W4A8."
        )
    if engine == "vllm":
        if method == "dgpp_mixed":
            return (
                "Stock vLLM does not implement quant_method dgpp_mixed "
                f"({detected.detail}). SMA will not rewrite it and will not "
                "install a plugin. --engine sglang refuses dgpp_mixed as well."
            )
        if fmt == "mixed" and "compressed" not in method:
            return (
                "vLLM mixed support in SMA is only for compressed-tensors configs "
                "that count both FP8 and NVFP4. "
                f"This checkpoint is {detected.detail}."
            )
        if fmt in {"bf16", "nvfp4", "mxfp4", "fp8", "mixed"}:
            return None
        if fmt in {"fp16", "fp32", "unknown", "other", ""}:
            return None
        return f"vLLM support for quant format {fmt} is not implemented in SMA."
    # SGLang.
    if method == "dgpp_mixed":
        return (
            "Stock SGLang refuses quant_method dgpp_mixed "
            f"({detected.detail}). SMA does not install a Flint or DGPP plugin."
        )
    if "compressed" in method:
        return (
            "SGLang in SMA refuses compressed-tensors checkpoints, including "
            "NVFP4, FP8, and mixed FP8/NVFP4. That combination is not verified "
            f"on the pinned image. This checkpoint is {detected.detail}. "
            "Re-run with --engine vllm."
        )
    if fmt == "mixed":
        return (
            "SGLang mixed support is not verified for this quant_method "
            f"({detected.detail}). SMA will not serve it."
        )
    if fmt in {"bf16", "nvfp4", "mxfp4", "fp8"}:
        return None
    if fmt in {"fp16", "fp32", "unknown", "other", ""}:
        return None
    return f"SGLang support for quant format {fmt} is not implemented in SMA."
