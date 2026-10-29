import json
from pathlib import Path

import pytest

from sma.errors import SmaError
from sma.memplan import (
    LADDER_START,
    Need,
    context_ladder,
    kv_layout_from_hf_config,
    plan_memory,
    stream_kv_bytes,
)

CONFIGS = Path(__file__).parent / "data" / "configs"
GB = 1_000_000_000
BUDGET = 100 * GB
OVERHEAD = 5 * GB


def _config(name: str) -> dict:
    return json.loads((CONFIGS / f"{name}.json").read_text())


def _layout(name: str):
    return kv_layout_from_hf_config(_config(name))


def _plan(layout, weights: int, kv_type: str = "fp8", needs: list[Need] | None = None):
    return plan_memory(
        layout=layout,
        weights_bytes=weights,
        kv_type=kv_type,
        budget_bytes=BUDGET,
        overhead_bytes=OVERHEAD,
        needs=needs or [],
    )


# Layouts of real configs. Values per token are K plus V, summed over full-attention layers.


def test_qwen3_8b_is_full_attention_with_a_40k_native_context() -> None:
    layout = _layout("qwen3-8b")
    assert layout.total_layers == 36
    assert layout.full_attention_layers == 36
    assert layout.full_values_per_token == 2 * 36 * 8 * 128 == 73_728
    assert layout.sliding_attention_layers == 0
    # sliding_window is null and use_sliding_window is false.
    assert layout.sliding_window is None
    assert layout.state_bytes_per_stream == 0
    assert (layout.native_context, layout.native_source) == (40_960, "max_position_embeddings")
    assert layout.rope_stretched is False
    assert layout.notes == ()


def test_qwen3_30b_a3b_2507_is_full_attention_at_256k() -> None:
    layout = _layout("qwen3-30b-a3b-instruct-2507")
    assert layout.full_attention_layers == 48
    assert layout.full_values_per_token == 2 * 48 * 4 * 128 == 49_152
    assert layout.state_bytes_per_stream == 0
    assert layout.native_context == 262_144
    assert layout.notes == ()


def test_qwen3_6_hybrid_reads_text_config_and_estimates_linear_attention_state() -> None:
    layout = _layout("qwen3.6-35b-a3b-nvfp4")
    assert layout.total_layers == 40
    assert layout.full_attention_layers == 10
    assert layout.full_values_per_token == 2 * 10 * 2 * 256 == 10_240
    # 30 Gated DeltaNet layers: 32 value heads x 128 x 128 in float32, plus a 3-step conv over
    # 2 x 16 x 128 + 32 x 128 channels in bfloat16.
    per_layer = 32 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 32 * 128) * 2
    assert layout.state_bytes_per_stream == 30 * per_layer == 64_389_120
    assert layout.native_context == 262_144
    assert layout.rope_stretched is False
    assert any("30 linear-attention layers" in note and "estimate" in note for note in layout.notes)


def test_qwen3_6_without_layer_types_falls_back_to_full_attention_interval() -> None:
    config = _config("qwen3.6-35b-a3b-nvfp4")
    del config["text_config"]["layer_types"]
    assert kv_layout_from_hf_config(config) == _layout("qwen3.6-35b-a3b-nvfp4")


def test_lfm2_5_counts_conv_layers_as_state() -> None:
    layout = _layout("lfm2.5-8b-a1b")
    assert layout.total_layers == 24
    assert layout.full_attention_layers == 6
    # No head_dim in the config: 2048 hidden / 32 heads = 64.
    assert layout.full_values_per_token == 2 * 6 * 8 * 64 == 6_144
    assert layout.state_bytes_per_stream == 18 * (3 - 1) * 2048 * 2
    assert layout.native_context == 128_000
    assert any("18 convolution layers" in note for note in layout.notes)


def test_nemotron_h_counts_attention_blocks_and_mamba_state() -> None:
    layout = _layout("nemotron-3.5-lightning-30b-a3b-nvfp4")
    assert layout.total_layers == 52
    assert layout.full_attention_layers == 6
    assert layout.full_values_per_token == 2 * 6 * 2 * 128 == 3_072
    # 23 Mamba2 layers: 64 heads x 64 x 128 state in float32, plus a 3-step conv over
    # 64 x 64 + 2 x 8 x 128 channels in bfloat16.
    per_layer = 64 * 64 * 128 * 4 + 3 * (64 * 64 + 2 * 8 * 128) * 2
    assert layout.state_bytes_per_stream == 23 * per_layer == 49_082_368
    assert layout.native_context == 1_048_576
    assert any("23 Mamba layers" in note for note in layout.notes)


def test_nemotron_h_older_hybrid_override_pattern_gives_the_same_layout() -> None:
    config = _config("nemotron-3.5-lightning-30b-a3b-nvfp4")
    chars = {"attention": "*", "mamba": "M", "moe": "E", "mlp": "-"}
    config["hybrid_override_pattern"] = "".join(chars[kind] for kind in config.pop("layers_block_type"))
    assert kv_layout_from_hf_config(config) == _layout("nemotron-3.5-lightning-30b-a3b-nvfp4")


def test_yarn_in_config_makes_native_the_original_length() -> None:
    config = _config("qwen3-8b")
    config["max_position_embeddings"] = 131_072
    config["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32_768}
    layout = kv_layout_from_hf_config(config)
    assert layout.rope_stretched is True
    assert layout.native_context == 32_768
    assert layout.native_source == "rope_scaling.original_max_position_embeddings"
    assert any("never stretches RoPE" in note for note in layout.notes)


def test_stretching_rope_parameters_per_layer_type_are_found() -> None:
    config = _config("qwen3-8b")
    config["original_max_position_embeddings"] = 8_192
    config["rope_parameters"] = {
        "full_attention": {"rope_type": "linear", "factor": 8.0},
        "sliding_attention": {"rope_type": "default"},
    }
    layout = kv_layout_from_hf_config(config)
    assert (layout.native_context, layout.native_source, layout.rope_stretched) == (
        8_192,
        "original_max_position_embeddings",
        True,
    )


def test_stretching_without_an_original_length_uses_the_config_maximum() -> None:
    # Gemma 3 ships linear scaling as part of the model and names no original length.
    config = _config("qwen3-8b")
    config["rope_scaling"] = {"rope_type": "linear", "factor": 8.0}
    layout = kv_layout_from_hf_config(config)
    assert (layout.native_context, layout.native_source) == (40960, "max_position_embeddings")
    assert layout.rope_stretched is True
    assert any("no original length" in note for note in layout.notes)


def test_no_context_length_at_all_is_refused_by_the_ladder() -> None:
    config = _config("qwen3-8b")
    del config["max_position_embeddings"]
    with pytest.raises(SmaError, match="no native context length"):
        context_ladder(kv_layout_from_hf_config(config).native_context)


def test_default_rope_type_is_not_stretching() -> None:
    # Qwen3.6 carries rope_parameters with rope_type "default" and mrope settings.
    assert _layout("qwen3.6-35b-a3b-nvfp4").rope_stretched is False


def test_sliding_layers_keep_only_the_window() -> None:
    config = _config("qwen3-8b")
    config["num_hidden_layers"] = 6
    config["layer_types"] = ["sliding_attention"] * 5 + ["full_attention"]
    config["sliding_window"] = 1024
    config["use_sliding_window"] = True
    layout = kv_layout_from_hf_config(config)
    per_layer = 2 * 8 * 128
    assert (layout.full_values_per_token, layout.sliding_values_per_token) == (per_layer, 5 * per_layer)
    assert layout.sliding_window == 1024
    assert stream_kv_bytes(layout, 4096, "fp8") == per_layer * 4096 + 5 * per_layer * 1024
    assert stream_kv_bytes(layout, 512, "fp8") == 6 * per_layer * 512


def test_sliding_layers_with_the_window_switched_off_count_as_full() -> None:
    config = _config("qwen3-8b")
    config["layer_types"] = ["sliding_attention"] * 36
    config["sliding_window"] = 4096
    layout = kv_layout_from_hf_config(config)  # use_sliding_window is false
    assert layout.full_attention_layers == 36
    assert layout.sliding_window is None
    assert any("counted as full attention" in note for note in layout.notes)


def test_mla_keeps_one_latent_per_layer() -> None:
    config = {
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 163_840,
    }
    layout = kv_layout_from_hf_config(config)
    assert layout.full_values_per_token == 27 * (512 + 64)
    assert any("MLA" in note for note in layout.notes)


def test_config_without_layer_count_or_heads_is_refused() -> None:
    with pytest.raises(SmaError, match="num_hidden_layers"):
        kv_layout_from_hf_config({"hidden_size": 4096})
    with pytest.raises(SmaError, match="head"):
        kv_layout_from_hf_config({"num_hidden_layers": 4, "max_position_embeddings": 4096})


def test_unknown_layer_type_is_refused_not_guessed() -> None:
    config = _config("lfm2.5-8b-a1b")
    config["layer_types"][0] = "retention"
    with pytest.raises(SmaError, match="'retention'"):
        kv_layout_from_hf_config(config)


def test_unknown_kv_type_is_refused() -> None:
    with pytest.raises(SmaError, match="Unknown KV cache type 'auto'"):
        stream_kv_bytes(_layout("qwen3-8b"), 1024, "auto")


# Context ladder.


def test_ladder_starts_at_a_smaller_native_maximum() -> None:
    assert context_ladder(40_960) == [40_960, 32_768, 16_384, 8_192, 4_096, 2_048, 1_024]


def test_ladder_starts_at_256k_for_long_native_contexts() -> None:
    assert context_ladder(1_048_576)[:3] == [LADDER_START, 131_072, 65_536]
    assert context_ladder(131_072)[:2] == [131_072, 65_536]
    assert context_ladder(128_000)[:2] == [128_000, 65_536]
    assert context_ladder(262_144)[-1] == 1_024


def test_ladder_refuses_an_unknown_native_context() -> None:
    with pytest.raises(SmaError, match="no native context length"):
        context_ladder(None)


# plan_memory.


def test_plan_fits_at_256k_and_sizes_the_pool() -> None:
    plan = _plan(_layout("qwen3-30b-a3b-instruct-2507"), 26_700_000_000)
    assert plan.context == 262_144
    assert plan.context_reason == "262,144 tokens fits"
    assert plan.pool_bytes == BUDGET - 26_700_000_000 - OVERHEAD
    assert plan.kv_bytes_per_token == 49_152
    assert plan.pool_tokens == plan.pool_bytes // 49_152
    assert plan.streams_32k == plan.pool_bytes // (49_152 * 32_768)
    assert plan.footprint_bytes == BUDGET


def test_plan_names_a_smaller_native_maximum() -> None:
    plan = _plan(_layout("qwen3-8b"), 16 * GB)
    assert plan.context == 40_960
    assert plan.context_reason == "native maximum 40,960 tokens fits"
    assert plan.streams_32k == plan.pool_bytes // (73_728 * 32_768)


def test_plan_steps_down_when_one_stream_does_not_fit() -> None:
    plan = _plan(_layout("qwen3-30b-a3b-instruct-2507"), 90 * GB)
    # 5 GB pool: 256K needs 12.9 GB, 128K 6.4 GB, 64K 3.2 GB.
    assert plan.context == 65_536
    assert plan.context_reason == "stepped down from 262,144 to 65,536 tokens to fit the 100 GB budget"
    assert plan.streams_full_context == 1


def test_buffers_per_token_shrink_the_pool_and_count_in_the_footprint() -> None:
    layout = _layout("qwen3-30b-a3b-instruct-2507")
    plan = plan_memory(
        layout=layout, weights_bytes=26_700_000_000, kv_type="fp8", budget_bytes=BUDGET,
        overhead_bytes=OVERHEAD, needs=[], buffer_bytes_per_token=49_152,
    )
    # Buffers as large as the KV per token halve the pool.
    assert plan.pool_bytes == (BUDGET - 26_700_000_000 - OVERHEAD) // 2
    assert plan.buffer_bytes == plan.pool_tokens * 49_152
    assert plan.footprint_bytes == 26_700_000_000 + OVERHEAD + plan.pool_bytes + plan.buffer_bytes <= BUDGET
    assert plan.to_json()["buffer_gb"] == round(plan.buffer_bytes / GB, 2)


def test_max_streams_caps_the_pool_at_those_streams() -> None:
    layout = _layout("qwen3-8b")
    capped = plan_memory(
        layout=layout, weights_bytes=16 * GB, kv_type="fp8", budget_bytes=BUDGET, overhead_bytes=OVERHEAD,
        needs=[], max_streams=2,
    )
    assert capped.context == 40_960
    assert capped.pool_bytes == 2 * stream_kv_bytes(layout, 40_960, "fp8")
    assert (capped.pool_streams, capped.streams_full_context) == (2, 2)
    # A cap above what the budget holds changes nothing.
    loose = plan_memory(
        layout=layout, weights_bytes=16 * GB, kv_type="fp8", budget_bytes=BUDGET, overhead_bytes=OVERHEAD,
        needs=[], max_streams=1_000,
    )
    assert loose.pool_streams is None
    assert loose.pool_bytes == _plan(layout, 16 * GB).pool_bytes


def test_plan_refuses_when_weights_alone_exceed_the_budget() -> None:
    with pytest.raises(SmaError, match=r"\(96 GB\).*\(5 GB\) need 101 GB, more than the 100 GB"):
        _plan(_layout("qwen3-8b"), 96 * GB)


def test_plan_refuses_when_no_step_fits_one_stream() -> None:
    with pytest.raises(SmaError, match="One stream at 1,024 tokens"):
        _plan(_layout("qwen3-30b-a3b-instruct-2507"), 94_990_000_000)


def test_each_benchmark_gets_the_largest_step_its_streams_fit() -> None:
    needs = [Need("single", 1, 1_024), Need("swe", 8, 32_768), Need("many", 64, 4_096)]
    plan = _plan(_layout("qwen3-30b-a3b-instruct-2507"), 26_700_000_000, needs=needs)
    # 68.3 GB pool. 8 streams at 128K need 51.5 GB. 64 streams at 16K need 51.5 GB.
    assert plan.benchmarks == {"single": 262_144, "swe": 131_072, "many": 16_384}
    assert plan.to_json()["benchmarks"] == plan.benchmarks


def test_benchmark_whose_minimum_context_does_not_fit_is_refused() -> None:
    with pytest.raises(SmaError, match=r"'swe' runs 4 streams and needs at least 131,072 tokens each"):
        _plan(_layout("qwen3-30b-a3b-instruct-2507"), 80 * GB, needs=[Need("swe", 4, 131_072)])


def test_benchmark_needing_more_than_the_model_context_is_refused() -> None:
    with pytest.raises(SmaError, match="model's context is 40,960 tokens"):
        _plan(_layout("qwen3-8b"), 16 * GB, needs=[Need("long", 1, 65_536)])


def test_plan_json_is_plain() -> None:
    plan = _plan(_layout("qwen3.6-35b-a3b-nvfp4"), 23_400_000_000)
    data = plan.to_json()
    assert json.loads(json.dumps(data)) == data
    assert data["ladder"][0] == 262_144
    assert data["weights_gb"] == 23.4
    assert data["notes"] == list(plan.notes)


def test_spec_deployment_sentence_for_qwen3_6_nvfp4() -> None:
    # docs/serving-design.md: "23.4 GB of weights, 256K per stream, 26 full-length streams within 100 GB."
    plan = _plan(_layout("qwen3.6-35b-a3b-nvfp4"), 23_400_000_000)
    assert (plan.context, plan.streams_full_context) == (262_144, 26)


# The worked example in docs/serving-design.md: a 40B MoE with 5 GB of overhead at 256K.
# The spec table left out the linear-attention state (64.4 MB per stream for the Qwen3.6 layout).
# With it included, the NVFP4 and UD-Q4_K_XL hybrid cells fit 24 streams, not 25.
WORKED_LEGS = {
    "bf16": (80 * GB, "fp8"),
    "nvfp4": (40 * GB * 534 // 800, "fp8"),
    "ud_q4_k_xl": (40 * GB * 464 // 800, "q8_0"),
}


@pytest.mark.parametrize(
    "leg, fixture, stream_gb, streams",
    [
        ("bf16", "qwen3-30b-a3b-instruct-2507", 12.88, 1),
        ("nvfp4", "qwen3-30b-a3b-instruct-2507", 12.88, 5),
        ("ud_q4_k_xl", "qwen3-30b-a3b-instruct-2507", 13.69, 5),
        ("bf16", "qwen3.6-35b-a3b-nvfp4", 2.75, 5),
        ("nvfp4", "qwen3.6-35b-a3b-nvfp4", 2.75, 24),
        ("ud_q4_k_xl", "qwen3.6-35b-a3b-nvfp4", 2.92, 24),
    ],
)
def test_worked_example_table(leg: str, fixture: str, stream_gb: float, streams: int) -> None:
    weights, kv_type = WORKED_LEGS[leg]
    plan = _plan(_layout(fixture), weights, kv_type)
    assert plan.context == 262_144
    assert plan.to_json()["stream_gb"] == stream_gb
    assert plan.streams_full_context == streams


def test_worked_example_bf16_full_attention_uses_97_9_gb() -> None:
    weights, kv_type = WORKED_LEGS["bf16"]
    plan = _plan(_layout("qwen3-30b-a3b-instruct-2507"), weights, kv_type)
    used = plan.weights_bytes + plan.overhead_bytes + plan.streams_full_context * plan.stream_bytes
    assert round(used / GB, 1) == 97.9


def test_need_must_ask_for_at_least_one_stream() -> None:
    with pytest.raises(SmaError, match="at least 1 stream"):
        Need("empty", 0, 1024)
