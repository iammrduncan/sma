import json
import socket
from pathlib import Path

import pytest

from sma.commands import bench_argv, bench_cells, serve_argv
from sma.config import default_config, load_config
from sma.errors import SmaError
from sma.identity import assert_alias_safe
from sma.records import new_manifest, publish
from sma.schemautil import validate_instance
from sma.scrub import scrub_text
from sma.speedmath import decode_tok_s, prefill_tok_s


def test_serve_command_pins_the_contract() -> None:
    cfg = default_config()
    bf16 = serve_argv(
        cfg,
        mode="quality",
        served_name="org/model",
        is_moe=False,
        quant_flag="bf16",
        moe_backend=None,
        batch_invariant=True,
    )
    assert bf16[0:2] == ["vllm", "serve"]
    assert "--tensor-parallel-size" in bf16
    assert bf16[bf16.index("--tensor-parallel-size") + 1] == "1"
    assert "--no-enable-flashinfer-autotune" in bf16
    assert "--no-async-scheduling" in bf16
    assert "--moe-backend" not in bf16
    assert "--max-logprobs" in bf16
    speed = serve_argv(
        cfg,
        mode="speed",
        served_name="org/model",
        is_moe=True,
        quant_flag="bf16",
        moe_backend="flashinfer_b12x",
        batch_invariant=False,
    )
    assert "--no-enable-flashinfer-autotune" not in speed
    assert "--moe-backend" not in speed
    nvfp4 = serve_argv(
        cfg,
        mode="quality",
        served_name="org/model",
        is_moe=True,
        quant_flag="nvfp4",
        moe_backend="flashinfer_b12x",
        batch_invariant=True,
    )
    assert nvfp4[nvfp4.index("--moe-backend") + 1] == "flashinfer_b12x"
    dense = serve_argv(
        cfg,
        mode="quality",
        served_name="org/model",
        is_moe=False,
        quant_flag="nvfp4",
        moe_backend=None,
        batch_invariant=True,
    )
    assert dense[dense.index("--linear-backend") + 1] == "flashinfer_b12x"


def test_bench_command_and_derived_rates() -> None:
    cfg = default_config()
    cell = next(item for item in bench_cells(cfg) if item.name.startswith("prefill_c1"))
    argv = bench_argv(cfg, container="sma-run-speed", served_name="org/model", cell=cell)
    text = " ".join(argv)
    assert "vllm bench serve" in text
    assert "--percentile-metrics ttft,tpot,itl,e2el" in text
    assert "--metric-percentiles 50,90,99" in text
    assert "--save-result" in text
    assert "--max-concurrency 1" in text
    assert prefill_tok_s(2048, 100.0) == pytest.approx(20480.0)
    assert decode_tok_s(20.0) == pytest.approx(50.0)
    assert prefill_tok_s(2048, 0) is None
    assert decode_tok_s(None) is None


def test_blank_manifest_matches_the_schema(tmp_path: Path) -> None:
    cfg = default_config()
    cfg.runs_dir = str(tmp_path / "runs")
    manifest = new_manifest(
        cfg=cfg,
        repo="example/tiny",
        quant_flag="bf16",
        revision=None,
        run_id="20260101T000000Z-abcd1234",
        reference_run_id=None,
        pair_job_id=None,
    )
    run_dir = tmp_path / "runs" / "example_tiny" / "vllm" / "bf16" / manifest["run_id"]
    run_dir.mkdir(parents=True)
    summary = publish(cfg, run_dir, manifest)
    validate_instance(json.loads((run_dir / "manifest.json").read_text()), "manifest.schema.json")
    validate_instance(summary, "summary.schema.json")
    assert summary["perplexity"] is None
    assert summary["speed"]["prefill_tok_s"] is None
    assert summary["engine"] == "vllm"
    assert summary["energy"]["joules"] is None


def test_alias_rejects_a_hostname(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu-host-1")
    monkeypatch.setattr(socket, "getfqdn", lambda: "gpu-host-1")
    with pytest.raises(SmaError, match="hostname"):
        assert_alias_safe("gpu-host-1")
    with pytest.raises(SmaError, match="dot"):
        assert_alias_safe("spark.lab")


def test_scrub_removes_hostname_and_token(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda: "secret-host")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_value")
    text = scrub_text("failed on secret-host token hf_test_token_value")
    assert "secret-host" not in text
    assert "hf_test_token_value" not in text
    assert "<host>" in text
    assert "<redacted>" in text


def test_repo_config_loads() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "sma.yaml")
    assert cfg.engine.name == "vllm"
    assert cfg.engine.image == "nvcr.io/nvidia/vllm:26.05.post1-py3"
    assert cfg.engine.sglang_image == "lmsysorg/sglang:v0.5.10.post1-cu130"
    assert cfg.engine.sglang_attention_backend == "flashinfer"
    assert cfg.logprobs.corpus_version == "sma-corpus-v2"
    assert cfg.telemetry.sw_power_cap_invalidates is False
    assert cfg.engine.moe_backend == "flashinfer_b12x"
    assert cfg.determinism.batch_sizes == [1, 8, 64]
    assert cfg.logprobs.top_k == 64
    assert cfg.host_alias == "gpu-host-1"
    assert [suite.name for suite in cfg.quality.suites] == ["ifeval", "gsm8k", "mmlu_pro", "gpqa_diamond"]
