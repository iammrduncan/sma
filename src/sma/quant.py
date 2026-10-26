"""Match a checkpoint's declared quantization to --bf16, --nvfp4, or --mxfp4.

SMA never quantizes. It only reads config.json and hf_quant_config.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sma.errors import QuantMismatch

QUANT_FLAGS = ("bf16", "nvfp4", "mxfp4")

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


def classify_algo(algo: str, method: str) -> str:
    """Return a format name, or '' when the blob does not declare a quant."""
    folded_algo = algo.upper().replace("-", "_")
    folded_method = method.lower().replace("-", "_")
    if "MXFP4" in folded_algo or folded_method in {"mxfp4", "gpt_oss_mxfp4"}:
        return "mxfp4"
    if "W4A16" in folded_algo and "NVFP4" in folded_algo:
        return "nvfp4_w4a16"
    if "NVFP4" in folded_algo or folded_algo in {"FP4", "FP4_WEIGHT_ONLY"}:
        return "nvfp4"
    if "FP8" in folded_algo or folded_method in {"fp8", "modelopt_fp8"}:
        return "fp8"
    if folded_method in {"modelopt_fp4"}:
        return "nvfp4"
    if folded_method in {"awq", "gptq", "compressed_tensors", "bitsandbytes", "fbgemm_fp8"}:
        return "other"
    if folded_algo or folded_method:
        if folded_method == "modelopt" and not folded_algo:
            return "other"
        return "other"
    return ""


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
    hf_fmt = classify_algo(*_algo_and_method(hf_blob)) if hf_blob else ""
    cfg_fmt = classify_algo(*_algo_and_method(cfg_blob)) if cfg_blob else ""
    if hf_fmt and cfg_fmt and hf_fmt != cfg_fmt:
        raise QuantMismatch(
            "hf_quant_config.json and config.json quantization_config disagree "
            f"({hf_fmt} vs {cfg_fmt}). SMA will not guess. Refusing this checkpoint."
        )
    declared = hf_fmt or cfg_fmt
    source = "hf_quant_config.json" if hf_fmt else "config.json quantization_config"
    if declared:
        detail = declared
        if declared == "nvfp4_w4a16":
            detail = (
                "NVFP4 weight-only (W4A16). This is not W4A4 NVFP4. "
                "It is accepted by --nvfp4 and labeled nvfp4_w4a16."
            )
        return DetectedQuant(
            format=declared,
            detail=f"{declared} via {source}. {detail}" if declared == "nvfp4_w4a16" else f"{declared} via {source}",
            dtype=dtype,
            model_type=str(model_type) if model_type else None,
            architectures=architectures,
            is_moe=is_moe,
            num_experts=num_experts,
            experts_per_tok=per_tok,
            batch_invariant_supported=supported,
            batch_invariant_reason=reason,
            producer=_producer(config, hf_quant),
            sources={"hf_quant_config": hf_fmt or None, "quantization_config": cfg_fmt or None},
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
        sources={"hf_quant_config": None, "quantization_config": None, "dtype": dtype},
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
                "(--nvfp4 or --mxfp4) or point sma at a BF16 repo."
            )
        return "bf16"
    if flag == "nvfp4":
        if fmt == "mxfp4":
            raise QuantMismatch(
                "This checkpoint is MXFP4, not NVFP4. "
                "Re-run with --mxfp4. SMA does not convert it."
            )
        if fmt == "nvfp4_w4a16":
            return "nvfp4_w4a16"
        if fmt != "nvfp4":
            raise QuantMismatch(
                f"--nvfp4 does not match this checkpoint ({detected.detail}). "
                "SMA does not quantize. Download an NVFP4 checkpoint and pass its repo."
            )
        return "nvfp4"
    # mxfp4
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
