"""GGUF reader: header parsing, the model summary, the KV layout, and quant file selection."""

from __future__ import annotations

import io
import struct

import pytest

from sma.errors import SmaError
from sma.gguf import (FILE_TYPES, GGML_BLOCK_SIZES, GGML_TYPES, gguf_quant_tags, match_gguf_files, read_header,
                      read_header_file, summarize)
from sma.memplan import KvLayout
from tests.gguf_writer import (ARRAY, BOOL, FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, STRING, UINT8, UINT16, UINT32,
                               UINT64, Typed, gguf_bytes, write_gguf)

F32, Q8_0, Q4_K = 0, 8, 12

EVERY_TYPE = {
    "test.uint8": Typed(UINT8, 255),
    "test.int8": Typed(INT8, -128),
    "test.uint16": Typed(UINT16, 65535),
    "test.int16": Typed(INT16, -32768),
    "test.uint32": Typed(UINT32, 2**32 - 1),
    "test.int32": Typed(INT32, -(2**31)),
    "test.float32": Typed(FLOAT32, 0.5),
    "test.bool": Typed(BOOL, True),
    "test.string": "héllo",
    "test.uint64": Typed(UINT64, 2**64 - 1),
    "test.int64": Typed(INT64, -(2**63)),
    "test.float64": Typed(FLOAT64, 1e-300),
    "test.int_array": Typed(ARRAY, [1, -2, 3], item_type=INT32),
    "test.string_array": ["a", "", "ç"],
    "test.bool_array": [True, False],
    "test.empty_array": [],
    "test.nested_array": [[1, 2], [], Typed(ARRAY, ["x"], item_type=STRING)],
}
TENSORS = [
    ("token_embd.weight", (256, 4), Q4_K),  # 4 blocks of 256 values, 144 bytes each
    ("blk.0.attn_q.weight", (64, 2), Q8_0),  # 4 blocks of 32 values, 34 bytes each
    ("output_norm.weight", (64,), F32),
    ("future.weight", (8,), 99),  # a ggml type SMA does not know
]


def model_meta(arch: str = "llama", extra: dict | None = None) -> dict:
    """A small model's metadata. extra keys without a general., tokenizer., or split. prefix get the arch prefix."""
    meta = {
        "general.architecture": arch,
        "general.name": "Tiny",
        "general.file_type": 15,
        f"{arch}.block_count": 4,
        f"{arch}.context_length": 8192,
        f"{arch}.embedding_length": 512,
        f"{arch}.attention.head_count": 8,
        f"{arch}.attention.head_count_kv": 2,
    }
    for key, value in (extra or {}).items():
        full = key if key.startswith(("general.", "tokenizer.", "split.")) else f"{arch}.{key}"
        if value is None:
            meta.pop(full, None)
        else:
            meta[full] = value
    return meta


def summary(meta: dict, tensors: list = ()):
    return summarize(read_header(io.BytesIO(gguf_bytes(meta, tensors)).read))


def layout(arch: str, extra: dict) -> KvLayout:
    return summary(model_meta(arch, extra)).layout


def chunked(blob: bytes, size: int):
    stream = io.BytesIO(blob)
    return lambda n: stream.read(min(n, size))


# Header


@pytest.mark.parametrize("chunk", [None, 3])
def test_read_header_round_trips_every_value_type_and_tensor(chunk):
    blob = gguf_bytes(EVERY_TYPE, TENSORS)
    header = read_header(io.BytesIO(blob).read if chunk is None else chunked(blob, chunk))

    expected = {key: value.value if isinstance(value, Typed) else value for key, value in EVERY_TYPE.items()}
    expected["test.nested_array"] = [[1, 2], [], ["x"]]
    assert header.metadata == expected
    assert (header.version, header.alignment) == (3, 32)
    assert [(t.name, t.shape, t.type_name, t.n_elements, t.n_bytes, t.offset) for t in header.tensors] == [
        ("token_embd.weight", (256, 4), "Q4_K", 1024, 576, 0),
        ("blk.0.attn_q.weight", (64, 2), "Q8_0", 128, 136, 576),
        ("output_norm.weight", (64,), "F32", 64, 256, 736),
        ("future.weight", (8,), "type 99", 8, None, 992),
    ]


def test_read_header_reads_version_2():
    assert read_header(io.BytesIO(gguf_bytes({"a": 1}, version=2)).read).version == 2


def test_read_header_reads_nothing_past_the_tensor_infos():
    header_only = gguf_bytes(EVERY_TYPE, TENSORS)
    stream = io.BytesIO(gguf_bytes(EVERY_TYPE, TENSORS, data=True))
    read_header(stream.read)
    assert len(stream.getvalue()) > len(header_only)
    assert stream.tell() == len(header_only)


def test_read_header_refuses_every_truncated_prefix():
    blob = gguf_bytes({"general.architecture": "llama", "test.array": [1, 2, 3], "test.s": ["ab"]}, TENSORS[:2])
    for cut in range(len(blob)):
        with pytest.raises(SmaError, match="GGUF header is truncated"):
            read_header(io.BytesIO(blob[:cut]).read)


def test_read_header_refuses_a_file_that_is_not_gguf():
    with pytest.raises(SmaError, match="not a GGUF file"):
        read_header(io.BytesIO(b"<!DOCTYPE html><html>").read)


@pytest.mark.parametrize("version", [1, 4])
def test_read_header_refuses_unsupported_versions(version):
    with pytest.raises(SmaError, match=f"GGUF version {version} is not supported"):
        read_header(io.BytesIO(gguf_bytes({}, version=version)).read)


def test_read_header_refuses_a_hostile_length_before_reading_it():
    reads = []
    blob = io.BytesIO(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**40))

    def read(n):
        reads.append(n)
        return blob.read(n)

    with pytest.raises(SmaError, match="64 MiB limit"):
        read_header(read)
    assert max(reads) == 8


def test_read_header_file_reads_a_path_and_names_it_in_errors(tmp_path):
    good = write_gguf(tmp_path / "tiny.gguf", model_meta(), TENSORS, data=True)
    assert read_header_file(good).metadata["general.architecture"] == "llama"
    pointer = tmp_path / "pointer.gguf"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(SmaError, match="pointer.gguf: not a GGUF file"):
        read_header_file(pointer)


def test_type_tables_follow_llama_cpp():
    assert (GGML_TYPES[30], GGML_TYPES[39], GGML_TYPES[42]) == ("BF16", "MXFP4", "Q2_0")
    assert 4 not in GGML_TYPES and 31 not in GGML_TYPES
    assert set(GGML_BLOCK_SIZES) == set(GGML_TYPES)
    assert (FILE_TYPES[15], FILE_TYPES[32], FILE_TYPES[38]) == ("MOSTLY_Q4_K_M", "MOSTLY_BF16", "MOSTLY_MXFP4_MOE")
    assert 4 not in FILE_TYPES


# Summary


def test_summarize_reports_file_type_tensors_and_params():
    info = summary(model_meta(), TENSORS)
    assert (info.architecture, info.name) == ("llama", "Tiny")
    assert (info.file_type, info.file_type_id) == ("MOSTLY_Q4_K_M", 15)
    assert info.tensor_types == {"Q4_K": 1, "Q8_0": 1, "F32": 1, "type 99": 1}
    assert info.tensor_bytes == {"Q4_K": 576, "Q8_0": 136, "F32": 256}
    assert info.params_total == 1024 + 128 + 64 + 8
    assert (info.split_count, info.is_moe, info.expert_count, info.expert_used_count) == (1, False, None, None)
    assert any("type 99" in note for note in info.notes)


def test_summarize_names_unknown_and_missing_file_types():
    assert summary(model_meta(extra={"general.file_type": 999})).file_type == "ftype 999"
    missing = summary(model_meta(extra={"general.file_type": None}))
    assert (missing.file_type, missing.file_type_id) == (None, None)


def test_summarize_reads_expert_counts():
    info = summary(model_meta(extra={"expert_count": 128, "expert_used_count": 8}))
    assert (info.is_moe, info.expert_count, info.expert_used_count) == (True, 128, 8)


TOKENIZER = {
    "tokenizer.ggml.model": "gpt2",
    "tokenizer.ggml.tokens": ["a", "b", "ab"],
    "tokenizer.ggml.merges": ["a b"],
    "tokenizer.chat_template": "{{ messages }}",
}


def test_tokenizer_fingerprint_is_stable_and_tracks_tokenizer_keys_only():
    base = summary(model_meta(extra=TOKENIZER)).tokenizer_fingerprint
    assert base.startswith("sha256:")
    reordered = dict(reversed(list(model_meta(extra=TOKENIZER).items())))
    assert summary(reordered).tokenizer_fingerprint == base
    other_model = model_meta(extra={**TOKENIZER, "general.name": "Other", "context_length": 4096})
    assert summary(other_model).tokenizer_fingerprint == base
    changed = model_meta(extra={**TOKENIZER, "tokenizer.ggml.tokens": ["a", "b", "ba"]})
    assert summary(changed).tokenizer_fingerprint != base
    assert summary(model_meta()).tokenizer_fingerprint is None


def test_native_context_is_context_length_without_rope_stretching():
    info = summary(model_meta())
    assert (info.native_context, info.native_source, info.rope_stretched) == (8192, "llama.context_length", False)


def test_native_context_under_yarn_is_the_original_length():
    info = summary(model_meta("qwen3", {
        "context_length": 131072,
        "rope.scaling.type": "yarn",
        "rope.scaling.factor": 4.0,
        "rope.scaling.original_context_length": 32768,
    }))
    assert (info.native_context, info.native_source, info.rope_stretched) == (
        32768, "qwen3.rope.scaling.original_context_length", True)


def test_native_context_is_unknown_when_stretching_has_no_original_length():
    found = layout("llama", {"rope.scaling.type": "linear", "rope.scaling.factor": 2.0})
    assert (found.native_context, found.rope_stretched) == (None, True)
    assert any("original_context_length" in note for note in found.notes)


def test_scaling_factor_of_one_is_not_stretching():
    found = layout("llama", {"rope.scaling.type": "yarn", "rope.scaling.factor": 1.0,
                             "rope.scaling.original_context_length": 4096})
    assert (found.native_context, found.rope_stretched) == (8192, False)


def test_summarize_totals_a_split_set_only_when_every_split_is_read():
    split = {"split.count": Typed(UINT16, 2)}
    first = read_header(io.BytesIO(gguf_bytes(
        model_meta(extra={**split, "split.no": Typed(UINT16, 0)}), TENSORS[:2])).read)
    second = read_header(io.BytesIO(gguf_bytes({**split, "split.no": Typed(UINT16, 1)}, TENSORS[2:3])).read)

    partial = summarize(first)
    assert (partial.split_count, partial.params_total) == (2, None)
    assert any("1 of 2 split files" in note for note in partial.notes)
    whole = summarize(first, second)
    assert whole.params_total == 1024 + 128 + 64
    assert whole.tensor_types == {"Q4_K": 1, "Q8_0": 1, "F32": 1}
    with pytest.raises(SmaError, match="not one split set"):
        summarize(first, first)


def test_summarize_refuses_files_it_cannot_plan():
    with pytest.raises(SmaError, match="no general.architecture"):
        summary({"general.name": "projector"})
    with pytest.raises(SmaError, match="no llama.block_count"):
        summary(model_meta(extra={"block_count": None}))
    with pytest.raises(SmaError, match="llama.context_length is 'long', not a count"):
        summary(model_meta(extra={"context_length": "long"}))
    with pytest.raises(SmaError, match="cannot find attention or recurrent layers"):
        layout("rwkv7", {"attention.head_count": 0, "attention.head_count_kv": 0})


# KV layout


def test_layout_plain_llama_counts_every_layer_with_the_embedding_head_size():
    found = layout("llama", {"block_count": 32, "embedding_length": 4096, "attention.head_count": 32,
                             "attention.head_count_kv": 8})
    assert found == KvLayout(
        total_layers=32, full_attention_layers=32, full_values_per_token=32 * 8 * (128 + 128),
        sliding_attention_layers=0, sliding_values_per_token=0, sliding_window=None, state_bytes_per_stream=0,
        native_context=8192, native_source="llama.context_length", rope_stretched=False)


def test_layout_falls_back_to_head_count_without_head_count_kv():
    found = layout("llama", {"attention.head_count_kv": None})
    assert found.full_values_per_token == 4 * 8 * (64 + 64)


def test_layout_lfm2_counts_only_layers_with_kv_heads_and_conv_state():
    # LFM2.5-1.2B's real metadata: 6 attention layers among 16, the rest short convolutions.
    kv_heads = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0]
    found = layout("lfm2", {"block_count": 16, "embedding_length": 2048, "feed_forward_length": 8192,
                            "attention.head_count": 32, "attention.head_count_kv": kv_heads, "shortconv.l_cache": 3})
    assert (found.total_layers, found.full_attention_layers, found.full_values_per_token) == (16, 6, 6 * 8 * 128)
    # Each of the 10 conv layers keeps (l_cache - 1) x 2048 F32 values per stream.
    assert found.state_bytes_per_stream == 10 * 2 * 2048 * 4


NEMOTRON_SSM = {"ssm.conv_kernel": 4, "ssm.inner_size": 4096, "ssm.state_size": 128, "ssm.group_count": 8}


def test_layout_nemotron_h_separates_attention_mamba_and_mlp_layers():
    # M is Mamba (no KV, no feed-forward), E an MoE or MLP layer (feed-forward only), A attention (KV only).
    # The 11th block is an MTP layer, which llama.cpp keeps out of the main KV cache.
    pattern = "MEMEAMEMEA"
    kv_heads = [2 if c == "A" else 0 for c in pattern] + [2]
    feed_forward = [1856 if c == "E" else 0 for c in pattern] + [0]
    found = layout("nemotron_h_moe", {
        "block_count": 11, "nextn_predict_layers": 1, "attention.head_count": 32, "attention.head_count_kv": kv_heads,
        "feed_forward_length": feed_forward, "attention.key_length": 128, "attention.value_length": 128,
        **NEMOTRON_SSM})
    assert (found.total_layers, found.full_attention_layers, found.full_values_per_token) == (10, 2, 2 * 2 * 256)
    mamba_values = 3 * (4096 + 2 * 8 * 128) + 128 * 4096  # conv state plus SSM state
    assert found.state_bytes_per_stream == 4 * mamba_values * 4
    assert any("MTP" in note for note in found.notes)


def test_layout_granite_hybrid_counts_mamba_layers_when_feed_forward_is_one_value():
    # Granite-H has an MLP in every layer, so feed_forward_length is one value and every layer without KV is Mamba.
    found = layout("granitehybrid", {
        "block_count": 10, "embedding_length": 1536, "feed_forward_length": 512, "attention.head_count": 12,
        "attention.head_count_kv": [0, 0, 0, 0, 0, 4, 0, 0, 0, 0], "ssm.conv_kernel": 4, "ssm.inner_size": 3072,
        "ssm.state_size": 128, "ssm.group_count": 1})
    assert (found.full_attention_layers, found.full_values_per_token) == (1, 4 * 256)
    assert found.state_bytes_per_stream == 9 * (3 * (3072 + 2 * 128) + 128 * 3072) * 4


QWEN35 = {"attention.head_count": 16, "attention.head_count_kv": 2, "attention.key_length": 256,
          "attention.value_length": 256, "ssm.conv_kernel": 4, "ssm.inner_size": 4096, "ssm.state_size": 128,
          "ssm.group_count": 16, "ssm.time_step_rank": 32}


def test_layout_qwen35moe_keeps_kv_only_in_every_fourth_layer():
    found = layout("qwen35moe", {"block_count": 40, "full_attention_interval": 4, "context_length": 262144, **QWEN35})
    assert (found.total_layers, found.full_attention_layers, found.full_values_per_token) == (40, 10, 10240)
    assert found.state_bytes_per_stream == 30 * (3 * (4096 + 2 * 16 * 128) + 128 * 4096) * 4
    assert found.notes == ()


def test_layout_qwen3next_without_an_interval_key_uses_the_llama_cpp_default():
    # Real Qwen3-Next GGUFs have no full_attention_interval. llama.cpp then uses 4.
    found = layout("qwen3next", {"block_count": 48, **QWEN35})
    assert (found.full_attention_layers, found.full_values_per_token) == (12, 12288)
    assert any("default of 4" in note for note in found.notes)


def test_layout_recurrent_layer_flags_override_the_interval():
    found = layout("qwen35", {"block_count": 8, "full_attention_interval": 4,
                              "attention.recurrent_layers": [True, False] * 4, **QWEN35})
    assert found.full_attention_layers == 4


GEMMA = {"attention.head_count": 8, "attention.head_count_kv": 4, "attention.key_length": 256,
         "attention.value_length": 256}


def test_layout_sliding_window_period_leaves_every_nth_layer_full():
    # Period 6: five sliding layers, then one full layer (llama.cpp's set_swa_pattern).
    found = layout("gemma3", {"block_count": 12, "attention.sliding_window": 1024,
                              "attention.sliding_window_pattern": 6, **GEMMA})
    assert (found.full_attention_layers, found.full_values_per_token) == (2, 2 * 4 * 512)
    assert (found.sliding_attention_layers, found.sliding_values_per_token, found.sliding_window) == (
        10, 10 * 4 * 512, 1024)


def test_layout_sliding_window_flags_use_the_sliding_head_size_and_skip_shared_kv():
    # Gemma 4 style: a flag per layer, smaller heads on sliding layers, and the last layers reuse earlier KV.
    found = layout("gemma4", {
        "block_count": 8, "attention.head_count": 8, "attention.head_count_kv": 2, "attention.key_length": 512,
        "attention.value_length": 512, "attention.key_length_swa": 256, "attention.value_length_swa": 256,
        "attention.sliding_window": 512, "attention.sliding_window_pattern": [True, True, False] * 2 + [True, False],
        "attention.shared_kv_layers": 2})
    # Layers 6 and 7 are shared. Of layers 0 to 5, layers 2 and 5 are full.
    assert (found.full_attention_layers, found.full_values_per_token) == (2, 2 * 2 * 1024)
    assert (found.sliding_attention_layers, found.sliding_values_per_token) == (4, 4 * 2 * 512)


def test_layout_sliding_window_without_a_pattern_counts_every_layer_as_full():
    # Real gemma3 and gpt-oss GGUFs carry no pattern key; llama.cpp hard-codes one. SMA overcounts instead of guessing.
    found = layout("gpt-oss", {"block_count": 24, "attention.sliding_window": 128, **GEMMA})
    assert (found.full_attention_layers, found.sliding_attention_layers, found.sliding_window) == (24, 0, None)
    assert any("does not say which layers slide" in note for note in found.notes)


# Quant file selection

SMALL = [
    ".gitattributes", "README.md", "config.json", "imatrix_unsloth.gguf_file",
    "Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-IQ4_XS.gguf", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf",
    "Qwen3-0.6B-UD-Q4_K_XL.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf",
]
BIG_STEM = "Qwen3-235B-A22B-Instruct-2507"
BIG = (
    ["README.md", f"{BIG_STEM}-UD-TQ1_0.gguf"]
    + [f"UD-Q4_K_XL/{BIG_STEM}-UD-Q4_K_XL-0000{i}-of-00003.gguf" for i in (3, 1, 2)]
    + [f"BF16/{BIG_STEM}-BF16-{i:05d}-of-00010.gguf" for i in range(1, 11)]
)


def test_match_picks_one_file_ignoring_case():
    assert match_gguf_files(SMALL, "Q8_0") == ["Qwen3-0.6B-Q8_0.gguf"]
    assert match_gguf_files(SMALL, "ud-q4_k_xl") == ["Qwen3-0.6B-UD-Q4_K_XL.gguf"]
    assert match_gguf_files(SMALL, "Q4_K_XL") == ["Qwen3-0.6B-UD-Q4_K_XL.gguf"]


def test_match_picks_a_complete_split_set_in_order():
    assert match_gguf_files(BIG, "UD-Q4_K_XL") == [
        f"UD-Q4_K_XL/{BIG_STEM}-UD-Q4_K_XL-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
    assert len(match_gguf_files(BIG, "BF16")) == 10
    assert match_gguf_files(BIG, "UD-TQ1_0") == [f"{BIG_STEM}-UD-TQ1_0.gguf"]


def test_match_needs_the_whole_token():
    assert match_gguf_files(SMALL, "Q4_K_M") == ["Qwen3-0.6B-Q4_K_M.gguf"]
    for partial in ("Q4_K", "Q4", "IQ4", "K_M"):
        with pytest.raises(SmaError, match=f"No GGUF model file in this repo matches '{partial}'"):
            match_gguf_files(SMALL, partial)


def test_match_skips_vision_projectors_and_mtp_drafts():
    # F16 appears only in mmproj-F16.gguf, and BF16 is a different token.
    with pytest.raises(SmaError, match=r"Available quant selectors: BF16, IQ4_XS, Q4_K_M, Q8_0, UD-Q4_K_XL\."):
        match_gguf_files(SMALL, "F16")
    gemma = ["gemma-4-E4B-it-Q8_0.gguf", "MTP/mtp-gemma-4-E4B-it-Q8_0.gguf", "mtp-gemma-4-E4B-it.gguf",
             "mmproj-BF16.gguf"]
    assert match_gguf_files(gemma, "Q8_0") == ["gemma-4-E4B-it-Q8_0.gguf"]


def test_match_refuses_an_ambiguous_selector_and_lists_the_choices():
    with pytest.raises(SmaError, match="more than one GGUF model: Model-Q4_K_XL.gguf, Model-UD-Q4_K_XL.gguf.*"
                                       "Available quant selectors: Q4_K_XL, UD-Q4_K_XL"):
        match_gguf_files(["Model-Q4_K_XL.gguf", "Model-UD-Q4_K_XL.gguf"], "Q4_K_XL")
    # One file and a split set of the same quant are two models.
    with pytest.raises(SmaError, match="more than one GGUF model"):
        match_gguf_files(["M-Q8_0.gguf", "Q8_0/M-Q8_0-00001-of-00002.gguf", "Q8_0/M-Q8_0-00002-of-00002.gguf"], "Q8_0")


def test_match_refuses_an_incomplete_split_set():
    listing = [path for path in BIG if "00002-of-00003" not in path]
    with pytest.raises(SmaError, match=r"incomplete: found parts \[1, 3\] of 3"):
        match_gguf_files(listing, "UD-Q4_K_XL")


def test_match_refuses_an_empty_selector():
    with pytest.raises(SmaError, match="selector is empty"):
        match_gguf_files(SMALL, " ")


def test_quant_tags_come_from_file_and_folder_names_and_each_selects_one_model():
    assert gguf_quant_tags(SMALL) == ["BF16", "IQ4_XS", "Q4_K_M", "Q8_0", "UD-Q4_K_XL"]
    assert gguf_quant_tags(BIG) == ["BF16", "UD-Q4_K_XL", "UD-TQ1_0"]
    for listing in (SMALL, BIG):
        for tag in gguf_quant_tags(listing):
            assert match_gguf_files(listing, tag)
