"""Match a checkpoint's declared quantization to the flag the user passed.

SMA never quantizes. It only reads config.json and hf_quant_config.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sma.errors import QuantMismatch

QUANT_FLAGS = ("bf16", "nvfp4", "mxfp4", "mixed")

_MOE_TYPES = {
    "qwen2_moe",
    "qwen3_moe",
    "qwen3_moe_text",
    "qwen3_5_moe",
    "mixtral",
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
    "gpt_oss",
    "phimoe",
    "arctic",
    "grok",
    "jamba",
    "olmoe",
    "nemotron_h",
    "flint",
}

# Gated delta-net attention is rejected by vLLM batch invariance (issue #48613).
_GATED_DELTANET_TYPES = {
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_6",
    "qwen3_next",
    "qwen3_next_moe",
}


@dataclass(frozen=True)
class DetectedQuant:
    format: str
    detail: str
    dtype: str | None
    model_type: str | None
    architectures: list[str]
    is_moe: bool
    num_experts: int | None
    experts_per_tok: int | None
    batch_invariant_supported: bool
    batch_invariant_reason: str | None
    producer: str | None
    sources: dict[str, Any]
    composition: dict[str, Any] | None = None


def _nested_text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    if isinstance(text, dict):
        return text
    return {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _algo_and_method(blob: dict[str, Any]) -> tuple[str, str]:
    method = str(blob.get("quant_method") or "")
    algo = str(blob.get("quant_algo") or "")
    nested = _as_dict(blob.get("quantization"))
    if not algo:
        algo = str(nested.get("quant_algo") or "")
    if not method and nested.get("quant_method"):
        method = str(nested.get("quant_method"))
    return algo, method


def _norm_token(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _family(token: str) -> str:
    folded = _norm_token(token)
    if not folded:
        return ""
    if "mxfp4" in folded:
        return "mxfp4"
    if "w4a16" in folded and "nvfp4" in folded:
        return "nvfp4_w4a16"
    if "nvfp4" in folded or folded in {"fp4", "fp4_weight_only"}:
        return "nvfp4"
    if "fp8" in folded or "float8" in folded:
        return "fp8"
    return ""


def _scheme_family(scheme: Any) -> str:
    """Family of one weight or activation scheme. Empty when the scheme is absent."""
    if not isinstance(scheme, dict):
        return ""
    bits = scheme.get("num_bits")
    kind = _norm_token(str(scheme.get("type") or ""))
    if bits == 4 and kind == "float":
        return "nvfp4"
    if bits == 8 and kind == "float":
        return "fp8"
    return ""


def _weight_only_nvfp4(group: dict[str, Any], weight_family: str) -> bool:
    """fp4 weights with null or missing input activations are W4A16, not W4A4."""
    if weight_family != "nvfp4":
        return False
    if "input_activations" not in group or group.get("input_activations") is None:
        return True
    acts = group.get("input_activations")
    if not isinstance(acts, dict):
        return True
    act_family = _scheme_family(acts)
    bits = acts.get("num_bits")
    if act_family == "nvfp4" or bits == 4:
        return False
    return True


def composition_from_precision_map(blob: dict[str, Any]) -> dict[str, Any] | None:
    """Count per-projection formats. dgpp_mixed stores these under routed_experts."""
    routed = blob.get("routed_experts")
    by_layer = None
    if isinstance(routed, dict):
        by_layer = routed.get("precision_by_layer")
    if not isinstance(by_layer, dict):
        by_layer = blob.get("precision_by_layer")
    if not isinstance(by_layer, dict) or not by_layer:
        return None
    counts: dict[str, int] = {}
    for projections in by_layer.values():
        if not isinstance(projections, dict):
            continue
        for precision in projections.values():
            family = _family(str(precision)) or _norm_token(str(precision))
            if not family:
                continue
            counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None
    activations = blob.get("activations")
    return {
        "method": str(blob.get("quant_method") or "per-layer"),
        "activations": str(activations) if activations else None,
        "formats": counts,
        "layers": len(by_layer),
        "projections": sum(counts.values()),
        "source": "precision_by_layer",
    }


def _group_family(group: dict[str, Any], top: str) -> str:
    weights = group.get("weights")
    weight_family = _scheme_family(weights) if isinstance(weights, dict) else ""
    if not weight_family:
        weight_family = _family(str(group.get("format") or "")) or top
    if weight_family == "nvfp4" and _weight_only_nvfp4(group, "nvfp4"):
        return "nvfp4_w4a16"
    return weight_family


def _compressed_tensors_families(blob: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}

    def _add(family: str) -> None:
        if family:
            counts[family] = counts.get(family, 0) + 1

    top = _family(str(blob.get("format") or ""))
    groups = blob.get("config_groups")
    if isinstance(groups, dict) and groups:
        for group in groups.values():
            if not isinstance(group, dict):
                continue
            _add(_group_family(group, top))
        return counts
    # A bare nvfp4 format with no activation scheme is not evidence of W4A4.
    if top == "nvfp4":
        _add("nvfp4_w4a16")
    else:
        _add(top)
    return counts


_UNQUANTIZED = {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "float"}


def _quantized_formats(counts: dict[str, int]) -> set[str]:
    return {name for name in counts if name not in _UNQUANTIZED}


def _recipe_name(blob: dict[str, Any]) -> str:
    for key in ("recipe", "quant_recipe", "precision"):
        raw = blob.get(key)
        if isinstance(raw, str) and raw.strip():
            return _norm_token(raw)
    return ""


def classify_blob(blob: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """Return (format, composition). Format is '' when the blob is not a quant config."""
    if not blob:
        return "", None
    method = _norm_token(str(blob.get("quant_method") or ""))
    algo, method_from_helper = _algo_and_method(blob)
    if not method:
        method = _norm_token(method_from_helper)
    per_layer = composition_from_precision_map(blob)
    counted = _quantized_formats(per_layer["formats"]) if per_layer else set()
    # The method name is not enough. mixed means both FP8 and NVFP4 were counted.
    if per_layer and "fp8" in counted and "nvfp4" in counted:
        return "mixed", per_layer
    if method == "dgpp_mixed" or (method.endswith("_mixed") and "compressed" not in method):
        if per_layer and counted == {"fp8"}:
            return "fp8", per_layer
        if per_layer and counted == {"nvfp4"}:
            return "nvfp4", per_layer
        recipe = _recipe_name(blob)
        if recipe in {"fp8", "float8", "float8_e4m3fn"} and "nvfp4" not in counted:
            return "fp8", per_layer
        if recipe in {"nvfp4", "fp4"} and "fp8" not in counted:
            return "nvfp4", per_layer
        return "other", per_layer
    if method == "compressed_tensors":
        counts = _compressed_tensors_families(blob)
        families = set(counts)
        composition = {
            "method": "compressed-tensors",
            "activations": None,
            "formats": counts,
            "layers": None,
            "projections": sum(counts.values()),
            "source": "config_groups" if blob.get("config_groups") else "format",
        }
        if families == {"nvfp4"} or families == {"nvfp4_w4a16"}:
            only = next(iter(families))
            return only, composition
        if families == {"fp8"}:
            return "fp8", composition
        if "nvfp4" in families and "fp8" in families:
            return "mixed", composition
        if "nvfp4" in families and len(families) == 1:
            return "nvfp4", composition
        return "other", composition or None
    folded_algo = algo.upper().replace("-", "_")
    if "MXFP4" in folded_algo or method in {"mxfp4", "gpt_oss_mxfp4"}:
        return "mxfp4", None
    if "W4A16" in folded_algo and "NVFP4" in folded_algo:
        return "nvfp4_w4a16", None
    if "NVFP4" in folded_algo or folded_algo in {"FP4", "FP4_WEIGHT_ONLY"} or method == "modelopt_fp4":
        return "nvfp4", None
    if "FP8" in folded_algo or method in {"fp8", "modelopt_fp8"}:
        return "fp8", None
    if method in {"awq", "gptq", "bitsandbytes", "fbgemm_fp8"}:
        return "other", None
    if folded_algo or method:
        return "other", None
    return "", None


def classify_algo(algo: str, method: str) -> str:
    """Backwards-compatible wrapper. Prefer classify_blob for real configs."""
    family, _composition = classify_blob({"quant_method": method, "quant_algo": algo, "format": algo})
    return family


def _dtype_name(config: dict[str, Any]) -> str | None:
    raw = config.get("torch_dtype", config.get("dtype"))
    if raw is None and isinstance(config.get("text_config"), dict):
        nested = config["text_config"]
        raw = nested.get("torch_dtype", nested.get("dtype"))
    if not isinstance(raw, str):
        return None
    folded = raw.lower()
    if folded in {"bfloat16", "bf16"}:
        return "bfloat16"
    if folded in {"float16", "fp16", "half"}:
        return "float16"
    if folded in {"float32", "fp32", "float"}:
        return "float32"
    return folded


def _producer(config: dict[str, Any], hf_quant: dict[str, Any] | None) -> str | None:
    blobs = [_as_dict(hf_quant), _as_dict(config.get("quantization_config"))]
    nested = _as_dict(_as_dict(hf_quant).get("quantization"))
    if nested:
        blobs.append(nested)
    for blob in blobs:
        producer = blob.get("producer")
        if isinstance(producer, str) and producer.strip():
            return producer.strip()
        if isinstance(producer, dict):
            name = producer.get("name") or producer.get("producer_name")
            version = producer.get("version")
            if name:
                text = str(name)
                if version:
                    text = f"{text} {version}"
                return text
    return None


def _moe_fields(config: dict[str, Any]) -> tuple[bool, int | None, int | None]:
    text = _nested_text_config(config)
    model_type = str(config.get("model_type") or text.get("model_type") or "")
    num_experts = _first_int(
        config.get("num_local_experts"),
        config.get("num_experts"),
        config.get("n_routed_experts"),
        config.get("moe_num_experts"),
        text.get("num_local_experts"),
        text.get("num_experts"),
        text.get("n_routed_experts"),
    )
    per_tok = _first_int(
        config.get("num_experts_per_tok"),
        config.get("num_experts_per_token"),
        config.get("moe_top_k"),
        config.get("num_experts_per_topk"),
        text.get("num_experts_per_tok"),
        text.get("moe_top_k"),
    )
    is_moe = num_experts is not None or model_type.lower() in _MOE_TYPES
    return is_moe, num_experts, per_tok


def _batch_invariant(model_type: str | None, architectures: list[str]) -> tuple[bool, str | None]:
    folded = (model_type or "").lower()
    if folded in _GATED_DELTANET_TYPES:
        return (
            False,
            f"model_type {model_type} uses gated delta-net attention, which vLLM "
            "rejects under VLLM_BATCH_INVARIANT (issue #48613). "
            "Quality will run at the fixed recorded concurrency without the flag.",
        )
    for name in architectures:
        if "GatedDeltaNet" in name or "Qwen3_5" in name or "Qwen3Next" in name:
            return (
                False,
                f"architecture {name} is treated as gated delta-net. "
                "Batch invariance is not enabled.",
            )
    return True, None


def detect_quant(
    config: dict[str, Any] | None,
    hf_quant: dict[str, Any] | None = None,
) -> DetectedQuant:
    config = config or {}
    text = _nested_text_config(config)
    model_type = config.get("model_type") or text.get("model_type")
    architectures = config.get("architectures") or text.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [str(architectures)]
    architectures = [str(item) for item in architectures]
    is_moe, num_experts, per_tok = _moe_fields(config)
    supported, reason = _batch_invariant(
        str(model_type) if model_type else None, architectures
    )
    dtype = _dtype_name(config)

    hf_blob = _as_dict(hf_quant)
    cfg_blob = _as_dict(config.get("quantization_config"))
    hf_fmt, hf_comp = classify_blob(hf_blob) if hf_blob else ("", None)
    cfg_fmt, cfg_comp = classify_blob(cfg_blob) if cfg_blob else ("", None)
    if hf_fmt and cfg_fmt and hf_fmt != cfg_fmt:
        raise QuantMismatch(
            "hf_quant_config.json and config.json quantization_config disagree "
            f"({hf_fmt} vs {cfg_fmt}). SMA will not guess. Refusing this checkpoint."
        )
    declared = hf_fmt or cfg_fmt
    composition = hf_comp or cfg_comp
    source = "hf_quant_config.json" if hf_fmt else "config.json quantization_config"
    quant_method = ""
    chosen = hf_blob if hf_fmt else cfg_blob
    if chosen:
        quant_method = str(chosen.get("quant_method") or "")
    if declared:
        detail = f"{declared} via {source}"
        if declared == "nvfp4_w4a16":
            detail = (
                f"{detail}. NVFP4 weight-only (W4A16). This is not W4A4 NVFP4. "
                "--nvfp4 refuses it and will not label it nvfp4."
            )
        elif declared == "mixed" and composition and composition.get("formats"):
            rendered = ", ".join(
                f"{count} {name}" for name, count in sorted(composition["formats"].items())
            )
            detail = f"{detail} ({quant_method or 'mixed'}: {rendered})"
        elif declared == "nvfp4" and quant_method:
            detail = f"{detail}; quant_method {quant_method}"
        return DetectedQuant(
            format=declared,
            detail=detail,
            dtype=dtype,
            model_type=str(model_type) if model_type else None,
            architectures=architectures,
            is_moe=is_moe,
            num_experts=num_experts,
            experts_per_tok=per_tok,
            batch_invariant_supported=supported,
            batch_invariant_reason=reason,
            producer=_producer(config, hf_quant),
            sources={
                "hf_quant_config": hf_fmt or None,
                "quantization_config": cfg_fmt or None,
                "quant_method": quant_method or None,
            },
            composition=composition,
        )

    if dtype == "bfloat16":
        fmt = "bf16"
        detail = "no quantization_config; dtype bfloat16"
    elif dtype == "float16":
        fmt = "fp16"
        detail = "no quantization_config; dtype float16"
    elif dtype == "float32":
        fmt = "fp32"
        detail = "no quantization_config; dtype float32"
    elif dtype is None:
        fmt = "unknown"
        detail = "no quantization_config and no dtype"
    else:
        fmt = "other"
        detail = f"no quantization_config; dtype {dtype}"
    return DetectedQuant(
        format=fmt,
        detail=detail,
        dtype=dtype,
        model_type=str(model_type) if model_type else None,
        architectures=architectures,
        is_moe=is_moe,
        num_experts=num_experts,
        experts_per_tok=per_tok,
        batch_invariant_supported=supported,
        batch_invariant_reason=reason,
        producer=None,
        sources={"hf_quant_config": None, "quantization_config": None, "dtype": dtype, "quant_method": None},
        composition=None,
    )


def assert_flag_matches(flag: str, detected: DetectedQuant) -> str:
    """Return the quant label to store. Raise QuantMismatch on disagreement."""
    if flag not in QUANT_FLAGS:
        raise QuantMismatch(f"Unknown quant flag {flag!r}. Expected one of {QUANT_FLAGS}.")
    fmt = detected.format
    if flag == "bf16":
        if fmt != "bf16":
            raise QuantMismatch(
                f"--bf16 does not match this checkpoint ({detected.detail}). "
                "SMA does not quantize or cast. Pass the flag for the checkpoint you downloaded "
                "(--nvfp4, --mxfp4, or --mixed) or point sma at a BF16 repo."
            )
        return "bf16"
    if flag == "nvfp4":
        if fmt == "mixed":
            raise QuantMismatch(
                "This checkpoint is mixed-precision, not NVFP4 "
                f"({detected.detail}). Re-run with --mixed. "
                "SMA does not flatten a mixed checkpoint into NVFP4."
            )
        if fmt == "mxfp4":
            raise QuantMismatch(
                "This checkpoint is MXFP4, not NVFP4. "
                "Re-run with --mxfp4. SMA does not convert it."
            )
        if fmt == "nvfp4_w4a16":
            raise QuantMismatch(
                "This checkpoint is weight-only NVFP4 (W4A16), not W4A4 NVFP4 "
                f"({detected.detail}). --nvfp4 will not label it nvfp4. "
                "SMA does not serve W4A16 and does not add a second flag for it."
            )
        if fmt != "nvfp4":
            raise QuantMismatch(
                f"--nvfp4 does not match this checkpoint ({detected.detail}). "
                "SMA does not quantize. Download an NVFP4 checkpoint and pass its repo."
            )
        return "nvfp4"
    if flag == "mixed":
        if fmt != "mixed":
            raise QuantMismatch(
                f"--mixed does not match this checkpoint ({detected.detail}). "
                "--mixed requires both FP8 and NVFP4 in the counted per-layer or group "
                "formats. A quant_method name alone is not enough. SMA does not mix precisions."
            )
        return "mixed"
    # mxfp4
    if fmt == "mixed":
        raise QuantMismatch(
            "This checkpoint is mixed FP8/NVFP4, not native MXFP4. Re-run with --mixed."
        )
    if fmt == "nvfp4" or fmt == "nvfp4_w4a16":
        raise QuantMismatch(
            "This checkpoint is NVFP4, not native MXFP4. "
            "Re-run with --nvfp4. SMA does not convert it."
        )
    if fmt != "mxfp4":
        raise QuantMismatch(
            f"--mxfp4 does not match this checkpoint ({detected.detail}). "
            "MXFP4 is only for checkpoints that already ship that way (for example gpt-oss)."
        )
    return "mxfp4"
