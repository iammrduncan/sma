"""The speed bench's tokenizer for a GGUF the bench image's transformers cannot read.

On a Spark, LFM3-1.7B's speed step failed: vLLM's bench client loads a GGUF's tokenizer through
transformers 5.6, which raised "GGUF model with architecture lfm3 is not supported yet". The source model's
tokenizer.json (LiquidAI/LFM3-1.7B-Base) matched the GGUF token for token, so it stands in when it matches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.gguf import base_model_repo, tokenizer_mismatch
from sma.pipeline import RunSpec, execute
from tests.fakes import FixtureFetcher
from tests.gguf_writer import write_gguf
from tests.harness import make_config, make_services

TOKENS = ["a", "b", "ab", "<|end|>", "[PAD4]", "[PAD5]"]
TYPES = [1, 1, 1, 3, 5, 5]  # normal, normal, normal, control, unused, unused
MERGES = ["a b"]
SOURCE = "Org/Tiny-Base"


def _tokenizer_json(vocab: dict[str, int] | None = None, merges: list | None = None) -> dict:
    return {
        "model": {"type": "BPE", "vocab": vocab or {"a": 0, "b": 1, "ab": 2}, "merges": merges if merges is not None else [["a", "b"]]},
        "added_tokens": [{"id": 3, "content": "<|end|>"}],
    }


def _meta(**extra) -> dict:
    return {
        "general.architecture": "tinyarch",
        "general.name": "Tiny",
        "general.file_type": 32,
        "general.base_model.count": 1,
        "general.base_model.0.repo_url": f"https://huggingface.co/{SOURCE}",
        "tinyarch.block_count": 2,
        "tinyarch.context_length": 40960,
        "tinyarch.embedding_length": 32,
        "tinyarch.attention.head_count": 4,
        "tinyarch.attention.head_count_kv": 2,
        "tinyarch.attention.key_length": 8,
        "tinyarch.attention.value_length": 8,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.tokens": TOKENS,
        "tokenizer.ggml.token_type": TYPES,
        "tokenizer.ggml.merges": MERGES,
        **extra,
    }


def test_a_matching_tokenizer_passes_and_each_difference_is_named() -> None:
    meta = _meta()
    assert base_model_repo(meta) == SOURCE
    assert base_model_repo({"general.base_model.0.repo_url": "https://example.com/x"}) is None
    assert tokenizer_mismatch(meta, _tokenizer_json()) is None
    assert "token ids differ, first 2" in tokenizer_mismatch(meta, _tokenizer_json({"a": 0, "b": 1, "ba": 2}))
    assert "past the GGUF's 6 tokens" in tokenizer_mismatch(meta, _tokenizer_json({"a": 0, "b": 1, "ab": 2, "x": 9}))
    assert "BPE merges differ" in tokenizer_mismatch(meta, _tokenizer_json(merges=[["b", "a"]]))
    # A GGUF id the tokenizer lacks must be unused padding.
    used = _meta(**{"tokenizer.ggml.token_type": [1, 1, 1, 3, 1, 5]})
    assert "not unused padding, first 4" in tokenizer_mismatch(used, _tokenizer_json())


def _services(tmp_path: Path, cfg, tokenizer: dict | None):
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    snapshot = tmp_path / "gguf"
    snapshot.mkdir()
    write_gguf(snapshot / "Tiny-BF16.gguf", _meta(), [("token_embd.weight", (256, 4), 30)], data=True)
    fetcher = FixtureFetcher(snapshot, repo="example/tiny-GGUF", quant_flag="gguf")
    if tokenizer is not None:
        folder = tmp_path / "tokenizer"
        folder.mkdir()
        (folder / "tokenizer.json").write_text(json.dumps(tokenizer))
        fetcher.tokenizers[SOURCE] = folder
    services.fetcher = fetcher
    # The bench image's transformers reads other architectures, not this one.
    runtime.gguf_archs = {"llama", "qwen3"}
    return services, fetcher


def test_the_bench_uses_the_source_models_tokenizer_when_it_matches(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, fetcher = _services(tmp_path, cfg, _tokenizer_json())
    run_id = execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="BF16"), cfg, services)
    run_dir = next(Path(cfg.runs_dir).glob(f"*/*/*/{run_id}"))
    engine = json.loads((run_dir / "manifest.json").read_text())["engine"]
    assert engine["bench_tokenizer"]["source"] == "repo"
    assert (engine["bench_tokenizer"]["repo"], engine["bench_tokenizer"]["revision"]) == (SOURCE, "b" * 40)
    assert "cannot read GGUF architecture tinyarch" in engine["bench_tokenizer"]["reason"]
    benches = [call for call in services.bencher.calls if "bench serve" in call]
    assert benches and all("--tokenizer /bench-tokenizer" in call for call in benches)
    assert all(f"{tmp_path / 'tokenizer'}:/bench-tokenizer:ro" in call for call in benches)
    # Asked once to choose, and once more to mount in this process at most.
    assert set(fetcher.tokenizer_requests) <= {(SOURCE, None), (SOURCE, "b" * 40)}


def test_speed_refuses_a_source_tokenizer_that_does_not_match(tmp_path: Path) -> None:
    from sma.errors import SmaError

    cfg = make_config(tmp_path)
    services, _fetcher = _services(tmp_path, cfg, _tokenizer_json({"a": 0, "b": 1, "ba": 2}))
    with pytest.raises(SmaError, match="tokenizer does not match the GGUF's: 1 token ids differ"):
        execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="BF16"), cfg, services)


def test_a_gguf_the_bench_reads_keeps_its_own_tokenizer(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, fetcher = _services(tmp_path, cfg, None)
    services.runtime.gguf_archs = {"tinyarch"}
    run_id = execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="BF16"), cfg, services)
    run_dir = next(Path(cfg.runs_dir).glob(f"*/*/*/{run_id}"))
    engine = json.loads((run_dir / "manifest.json").read_text())["engine"]
    assert engine["bench_tokenizer"] == {"source": "gguf", "file": "Tiny-BF16.gguf"}
    assert all("--tokenizer /model/Tiny-BF16.gguf" in call for call in services.bencher.calls if "bench serve" in call)
    assert fetcher.tokenizer_requests == []
