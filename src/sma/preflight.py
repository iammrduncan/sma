"""Fail before a long download or a server start when the engine cannot load the arch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from sma.config import Config
from sma.errors import SmaError
from sma.engines import image_ref
from sma.logsetup import log_event
from sma.proc import run_command, run_command_streaming
from sma.quant import DetectedQuant

# Used only when the container cannot be asked. A miss here warns and continues,
# except known-unsupported architectures such as Flint, which are always refused.
# Names below are the public class names of common checkpoints. The container
# registry, when it can be queried, is the authority.
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
    "Glm4MoeForCausalLM",
    "ChatGLMModel",
    "StableLmForCausalLM",
    "NemotronHForCausalLM",
    "Llama4ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
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
    "Qwen3NextForCausalLM",
    "NemotronHForCausalLM",
    "Llama4ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Glm4MoeForCausalLM",
    "Qwen2VLForConditionalGeneration",
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


def _digest_from_inspect(text: str) -> str | None:
    stripped = text.strip()
    if not stripped or stripped == "null":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and "@sha256:" in item:
                return item.split("@", 1)[1]
        return None
    if "sha256:" in stripped:
        token = stripped.split("sha256:", 1)[1].strip().strip('"')
        return "sha256:" + token.split()[0].strip('"')
    return None


def query_container_archs(
    engine: str,
    image: str,
    *,
    pull_timeout_s: float = 3600,
    run_timeout_s: float = 120,
) -> tuple[set[str] | None, str | None]:
    """Pull the image, then ask the recorded digest which architectures it registers.

    Returns (architectures, digest). Architectures is None when the image could
    not be pulled or asked. The docker run uses image@digest, not the moving tag.
    """
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
    log_event("image_pull_start", engine=engine, image=image)
    try:
        pull = run_command_streaming(
            ["docker", "pull", image],
            timeout_s=pull_timeout_s,
            what=f"docker pull {engine} image",
        )
    except SmaError as exc:
        log_event("image_pull_failed", engine=engine, image=image, error=str(exc))
        return None, None
    if not pull.ok:
        log_event("image_pull_failed", engine=engine, image=image, error=(pull.stderr or pull.stdout)[-400:])
        return None, None
    try:
        inspect = run_command(
            ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
            timeout_s=run_timeout_s,
            what=f"{engine} image digest",
        )
    except SmaError as exc:
        log_event("image_inspect_failed", engine=engine, error=str(exc))
        return None, None
    digest = _digest_from_inspect(inspect.stdout) if inspect.ok else None
    if not digest:
        log_event("image_digest_missing", engine=engine, image=image)
        return None, None
    run_image = image_ref(image, digest)
    try:
        result = run_command(
            ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3", run_image, "-c", code],
            timeout_s=run_timeout_s,
            what=f"{engine} architecture registry",
        )
    except SmaError as exc:
        log_event("registry_query_failed", engine=engine, image=run_image, error=str(exc))
        return None, digest
    if not result.ok:
        log_event("registry_query_failed", engine=engine, image=run_image, error=(result.stderr or "")[-400:])
        return None, digest
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                return None, digest
            if isinstance(parsed, list):
                return {str(item) for item in parsed}, digest
    return None, digest


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


def _is_flint(detected: DetectedQuant, repo_files: list[str]) -> bool:
    if detected.model_type == "flint" or any("Flint" in arch for arch in detected.architectures):
        return True
    return _sglang_wheel(repo_files) is not None and (
        detected.model_type == "flint" or any("Flint" in arch for arch in detected.architectures)
    )


def _flint_message(engine: str, detected: DetectedQuant, repo_files: list[str]) -> str:
    wheel = _sglang_wheel(repo_files)
    names = ", ".join(detected.architectures) or (detected.model_type or "FlintForCausalLM")
    extra = ""
    if engine == "vllm":
        extra = " Re-run with --engine sglang only if that image's own registry lists it."
    wheel_bit = f" The repo contains {wheel}." if wheel else ""
    return (
        f"{engine} cannot load {names}. FlintForCausalLM is not a stock model. "
        "SMA does not install a checkpoint wheel and does not add a Flint plugin."
        + wheel_bit
        + extra
        + " A stock SGLang image still does not register FlintForCausalLM."
    )


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
    names = ", ".join(architectures) or (detected.model_type or "unknown")
    if _is_flint(detected, repo_files):
        return ArchCheck(False, "unsupported", _flint_message(engine, detected, repo_files))
    if _has_remote_code(config, trust_remote_code):
        return ArchCheck(
            True,
            "auto_map",
            "trust_remote_code is set and config.auto_map names a causal LM. "
            "The engine will be asked to load that remote code.",
        )
    looked_up: set[str] | None = None
    if registry is not None:
        try:
            looked_up = registry(engine, image)
        except Exception:
            looked_up = None
    if looked_up is not None:
        if any(arch in looked_up for arch in architectures):
            return ArchCheck(True, "container", f"{architectures[0] if architectures else names} is registered")
        extra = ""
        if engine == "vllm":
            extra = (
                " vLLM's registry does not include this architecture, and the checkpoint "
                "has no auto_map. Re-run with --engine sglang only if that image actually "
                "registers it."
            )
        else:
            extra = (
                " This SGLang image's registry does not include the architecture. "
                "SMA does not install a checkpoint wheel into the image."
            )
        return ArchCheck(
            False,
            "container",
            f"{engine} cannot load {names}. Checked the container registry.{extra}",
        )
    bundled = bundled_archs(engine)
    if any(arch in bundled for arch in architectures):
        return ArchCheck(
            True,
            "bundled",
            "WARNING: the image registry could not be queried. "
            f"{architectures[0] if architectures else names} is on SMA's built-in list, "
            "so the run continues. This is not a check of the container.",
        )
    return ArchCheck(
        True,
        "bundled",
        "WARNING: the image registry could not be queried and "
        f"{names} is not on SMA's built-in list. The run continues. "
        "This is not proof the engine can load it.",
    )


class ContainerRegistry:
    """Pulls an image once, records its digest, and caches the architecture set."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.last_digest: str | None = None
        self._cache: dict[tuple[str, str], set[str] | None] = {}

    def __call__(self, engine: str, image: str) -> set[str] | None:
        key = (engine, image)
        if key not in self._cache:
            archs, digest = query_container_archs(
                engine,
                image,
                pull_timeout_s=float(self.cfg.timeouts.image_pull_s),
                run_timeout_s=float(self.cfg.timeouts.docker_s),
            )
            self._cache[key] = archs
            if digest:
                self.last_digest = digest
        return self._cache[key]


def default_registry(cfg: Config) -> ContainerRegistry:
    return ContainerRegistry(cfg)

