import json
from pathlib import Path

import pytest

from sma.errors import QuantMismatch, SmaError
from sma.params import active_params, billions, count_header, size_class
from sma.quant import assert_flag_matches, detect_quant
from sma.safetensors import read_header
from tests.fakes import write_safetensors, write_snapshot
from sma.fetch import inspect_snapshot


def test_bf16_matches_and_nvfp4_does_not() -> None:
    detected = detect_quant({"model_type": "llama", "torch_dtype": "bfloat16"})
    assert assert_flag_matches("bf16", detected) == "bf16"
    with pytest.raises(QuantMismatch, match="--nvfp4"):
        assert_flag_matches("nvfp4", detected)


def test_nvfp4_from_hf_quant_config_rejects_bf16() -> None:
    detected = detect_quant(
        {"model_type": "llama", "torch_dtype": "bfloat16"},
        {"quantization": {"quant_algo": "NVFP4", "group_size": 16}},
    )
    assert detected.format == "nvfp4"
    assert detected.producer is None
    with pytest.raises(QuantMismatch, match="--bf16"):
        assert_flag_matches("bf16", detected)
    assert assert_flag_matches("nvfp4", detected) == "nvfp4"


def test_mxfp4_and_nvfp4_are_not_interchangeable() -> None:
    mx = detect_quant({"quantization_config": {"quant_method": "mxfp4"}, "model_type": "gpt_oss"})
    assert mx.is_moe is True
    assert assert_flag_matches("mxfp4", mx) == "mxfp4"
    with pytest.raises(QuantMismatch, match="--mxfp4"):
        assert_flag_matches("nvfp4", mx)
    nv = detect_quant({}, {"quantization": {"quant_algo": "NVFP4"}})
    with pytest.raises(QuantMismatch, match="--nvfp4"):
        assert_flag_matches("mxfp4", nv)


def test_w4a16_is_labeled_and_not_called_w4a4() -> None:
    detected = detect_quant({}, {"quantization": {"quant_algo": "W4A16_NVFP4"}})
    assert detected.format == "nvfp4_w4a16"
    with pytest.raises(QuantMismatch, match="W4A16"):
        assert_flag_matches("nvfp4", detected)


def test_fp16_is_not_bf16() -> None:
    detected = detect_quant({"torch_dtype": "float16"})
    with pytest.raises(QuantMismatch, match="float16"):
        assert_flag_matches("bf16", detected)


def test_disagreeing_quant_files_are_refused() -> None:
    with pytest.raises(QuantMismatch, match="disagree"):
        detect_quant(
            {"quantization_config": {"quant_method": "mxfp4"}},
            {"quantization": {"quant_algo": "NVFP4"}},
        )


def test_gated_deltanet_disables_batch_invariance() -> None:
    detected = detect_quant({"model_type": "qwen3_5", "torch_dtype": "bfloat16"})
    assert detected.batch_invariant_supported is False
    assert detected.batch_invariant_reason
    assert "48613" in detected.batch_invariant_reason


def test_inspect_snapshot_mismatch_message(tmp_path: Path) -> None:
    write_snapshot(tmp_path, quant="nvfp4")
    with pytest.raises(QuantMismatch, match="nvfp4"):
        inspect_snapshot(tmp_path, repo="org/nv", revision_sha="abc", quant_flag="bf16")


def test_moe_active_params_and_size_class(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(
        path,
        {
            "model.embed.weight": [10, 10],
            "model.layers.0.mlp.experts.0.w": [10, 10],
            "model.layers.0.mlp.experts.1.w": [10, 10],
            "model.layers.0.mlp.shared_expert.w": [10, 10],
        },
    )
    total, expert, other = count_header(read_header(path))
    assert (total, expert, other) == (400, 200, 200)
    active, method = active_params(
        total=total, expert=expert, other=other, num_experts=2, experts_per_tok=1, is_moe=True
    )
    assert active == 300
    assert "num_experts_per_tok" in method
    assert size_class(3_000_000_000) == "1-3B"
    assert size_class(3_050_000_000) == "3-8B"
    assert size_class(20_000_000_000) == "15-20B"
    assert size_class(20_050_000_000) == "20-34B"
    assert size_class(940_000_000) == "<1B"
    assert billions(3_000_000_000) == 3.0


def test_safetensors_rejects_a_huge_header(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\xff" * 8)
    with pytest.raises(ValueError, match="header"):
        read_header(path)


def test_extra_quantization_flag_is_rejected() -> None:
    from sma.config import validate_config, default_config

    cfg = default_config()
    cfg.engine.extra_args = ["--quantization", "fp8"]
    with pytest.raises(SmaError, match="never quantizes"):
        validate_config(cfg)
    cfg.engine.extra_args = ["--tensor-parallel-size", "2"]
    with pytest.raises(SmaError, match="tensor parallelism"):
        validate_config(cfg)


def test_producer_is_recorded(tmp_path: Path) -> None:
    write_snapshot(tmp_path, quant="nvfp4")
    result = inspect_snapshot(tmp_path, repo="org/nv", revision_sha="abc", quant_flag="nvfp4")
    assert result.detected.producer == "example-recipe 0"
    payload = json.loads((tmp_path / "config.json").read_text())
    assert payload["torch_dtype"] == "bfloat16"
