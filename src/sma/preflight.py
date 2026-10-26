"""Fail before a long download or a server start when the engine cannot load the arch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from sma.config import Config
from sma.errors import SmaError
from sma.proc import run_command
from sma.quant import DetectedQuant

# Used only when the container cannot be asked. Unknown architectures still fail.
BUNDLED_VLLM_ARCHS = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Phi3ForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3TextForCausalLM",
    "GPT2LMHeadModel",
    "GPTJForCausalLM",
    "GPTNeoXForCausalLM",
    "FalconForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "GptOssForCausalLM",
    "CohereForCausalLM",
    "InternLM2ForCausalLM",
    "OlmoForCausalLM",
    "OlmoeForCausalLM",
    "JambaForCausalLM",
    "PhiMoEForCausalLM",
    "DbrxForCausalLM",
    "GraniteForCausalLM",
    "GraniteMoeForCausalLM",
    "GlmForCausalLM",
    "Glm4ForCausalLM",
    "ChatGLMModel",
    "StableLmForCausalLM",
}

BUNDLED_SGLANG_ARCHS = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "GptOssForCausalLM",
    "Phi3ForCausalLM",
    "InternLM2ForCausalLM",
}

Registry = Callable[[str, str], set[str] | None]


@dataclass(frozen=True)
class ArchCheck:
    ok: bool
    source: str
    message: str


def bundled_archs(engine: str) -> set[str]:
    if engine == "sglang":
        return set(BUNDLED_SGLANG_ARCHS)
    return set(BUNDLED_VLLM_ARCHS)


def query_container_archs(engine: str, image: str, *, timeout_s: float = 120) -> set[str] | None:
    """Ask the engine image which architectures it registers. None if it cannot be asked."""
    if engine == "vllm":
        code = (
            "import json\n"
            "from vllm.model_executor.models.registry import ModelRegistry\n"
            "print(json.dumps(sorted(ModelRegistry.get_supported_archs())))\n"
        )
    else:
        code = (
            "import json\n"
            "from sglang.srt.models.registry import ModelRegistry\n"
            "models = getattr(ModelRegistry, 'models', {})\n"
            "print(json.dumps(sorted(models)))\n"
        )
    try:
        result = run_command(
            ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3", image, "-c", code],
            timeout_s=timeout_s,
            what=f"{engine} architecture registry",
        )
    except SmaError:
        return None
    if not result.ok:
        return None
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return {str(item) for item in parsed}
    return None


def _has_remote_code(config: dict[str, Any], trust_remote_code: bool) -> bool:
    auto_map = config.get("auto_map")
    if not trust_remote_code or not isinstance(auto_map, dict) or not auto_map:
        return False
    return any("CausalLM" in str(key) or "AutoModel" in str(key) for key in auto_map)


def _sglang_wheel(files: list[str]) -> str | None:
    for name in files:
        lowered = name.lower()
        if lowered.startswith("sglang/") and lowered.endswith(".whl"):
            return name
    return None


def check_architecture(
    *,
    engine: str,
    detected: DetectedQuant,
    config: dict[str, Any],
    repo_files: list[str],
    trust_remote_code: bool,
    registry: Registry | None,
    image: str,
) -> ArchCheck:
    architectures = list(detected.architectures)
    if _has_remote_code(config, trust_remote_code):
        return ArchCheck(
            True,
            "auto_map",
            "trust_remote_code is set and config.auto_map names a causal LM. "
            "The engine will be asked to load that remote code.",
        )
    looked_up: set[str] | None = None
    source = "bundled"
    if registry is not None:
        try:
            looked_up = registry(engine, image)
        except Exception:
            looked_up = None
    if looked_up is not None:
        supported = looked_up
        source = "container"
    else:
        supported = bundled_archs(engine)
        source = "bundled"
    if any(arch in supported for arch in architectures):
        return ArchCheck(True, source, f"{architectures[0] if architectures else detected.model_type} is registered")
    wheel = _sglang_wheel(repo_files)
    extra = ""
    if engine == "vllm":
        extra = (
            " vLLM's registry does not include this architecture, and the checkpoint "
            "has no auto_map. Re-run with --engine sglang only if that image actually "
            "registers it."
        )
        if detected.model_type == "flint" or any("Flint" in arch for arch in architectures) or wheel:
            extra += (
                " RecursiveSI/flint ships an SGLang wheel"
                + (f" ({wheel})" if wheel else "")
                + " and is not a vLLM model. A stock SGLang image still does not "
                "register FlintForCausalLM; SMA will not pretend that it does."
            )
    else:
        extra = (
            " This SGLang image's registry does not include the architecture. "
            "SMA does not install a checkpoint wheel into the image."
        )
        if wheel:
            extra += f" The repo contains {wheel}, which the pinned image does not load by itself."
    where = "the container registry" if source == "container" else "SMA's bundled registry (the image could not be queried)"
    names = ", ".join(architectures) or (detected.model_type or "unknown")
    return ArchCheck(
        False,
        source,
        f"{engine} cannot load {names}. Checked {where}.{extra}",
    )


def default_registry(cfg: Config) -> Registry:
    cache: dict[tuple[str, str], set[str] | None] = {}

    def _lookup(engine: str, image: str) -> set[str] | None:
        key = (engine, image)
        if key not in cache:
            cache[key] = query_container_archs(engine, image, timeout_s=cfg.timeouts.docker_s)
        return cache[key]

    return _lookup

