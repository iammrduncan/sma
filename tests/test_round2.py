"""Tests for the second review. Fakes follow the command, not the path SMA hopes for."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.backends import SGLANG_DATASET_CONTAINER_PATH, get_backend
from sma.commands import BenchCell, bench_argv
from sma.config import default_config
from sma.engines import compatibility_error
from sma.errors import QuantMismatch, SmaError
from sma.pipeline import RunSpec, execute
from sma.preflight import check_architecture, query_container_archs
from sma.proc import CommandResult
from sma.quant import assert_flag_matches, detect_quant
from sma.queue import JobQueue
from sma.speedmath import load_bench, percentile_ms
from sma.telemetry import Sample, coverage_problem
from tests.fakes import FakeBencher, bench_output_host_path, write_snapshot
from tests.harness import make_config, make_services


def test_bench_command_writes_the_attempt_directory_sma_reads(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    command_path = next(run_dir.glob("speed/cells/*/attempt-1/command.json"))
    argv = json.loads(command_path.read_text())["argv"]
    declared = bench_output_host_path(argv, command_path.parent / "bench.json")
    assert declared == command_path.parent / "bench.json"
    assert (command_path.parent / "bench.json").is_file()
    assert not (command_path.parents[1] / "bench.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["speed"]["prefill_tok_s"] is not None


def test_fake_bencher_does_not_write_the_expected_path_when_the_command_disagrees(tmp_path: Path) -> None:
    bencher = FakeBencher()
    expected = tmp_path / "run" / "speed" / "cells" / "prefill_c1_r1" / "attempt-1" / "bench.json"
    argv = [
        "docker",
        "exec",
        "box",
        "vllm",
        "bench",
        "serve",
        "--result-dir",
        "/sma-run/speed/cells/prefill_c1_r1",
        "--result-filename",
        "bench.json",
    ]
    bencher.run(argv, expected, 1, tmp_path / "bench.log")
    assert not expected.is_file()
    assert (tmp_path / "run" / "speed" / "cells" / "prefill_c1_r1" / "bench.json").is_file()


def test_sglang_serve_enables_metrics_and_caps_running_requests() -> None:
    cfg = default_config()
    argv = get_backend("sglang").serve_argv(
        cfg,
        mode="speed",
        served_name="org/model",
        is_moe=False,
        quant_flag="bf16",
        moe_backend=None,
        batch_invariant=False,
    )
    assert "--enable-metrics" in argv
    assert argv[argv.index("--max-running-requests") + 1] == str(cfg.engine.max_num_seqs)
    assert argv[argv.index("--random-seed") + 1] == str(cfg.seed)


def test_sglang_bench_uses_a_pinned_local_dataset_and_the_attempt_file() -> None:
    cfg = default_config()
    cell = BenchCell("prefill_c1_r1", "prefill", 128, 16, 1, 1)
    argv = bench_argv(
        cfg,
        container="sma-x-speed",
        served_name="org/model",
        cell=cell,
        engine="sglang",
        result_dir="/sma-run/speed/cells/prefill_c1_r1/attempt-2",
    )
    text = " ".join(argv)
    assert "--dataset-name random" in text
    assert f"--dataset-path {SGLANG_DATASET_CONTAINER_PATH}" in text
    assert "--random-range-ratio 1.0" in text
    assert f"--seed {cfg.seed}" in text
    assert "--tokenize-prompt" in text
    assert "--output-details" in text
    assert "sharegpt" not in text
    assert argv[argv.index("--output-file") + 1].endswith("attempt-2/bench.json")


def test_failed_metrics_scrape_is_null_and_does_not_fail_the_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def boom(*_args: object, **_kwargs: object) -> str:
        raise SmaError("metrics endpoint refused")

    monkeypatch.setattr("sma.pipeline.scrape_metrics", boom)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    metrics = json.loads(next(run_dir.glob("speed/cells/*/attempt-1/metrics.json")).read_text())
    assert metrics["scrape_error"]
    assert metrics["diff"]["watched"]["vllm:generation_tokens_total"] is None
    assert "null" in metrics["metrics_note"] or "failed" in metrics["metrics_note"]
    assert json.loads((run_dir / "manifest.json").read_text())["steps"]["speed"]["status"] == "done"


def test_weight_only_compressed_tensors_is_w4a16_and_nvfp4_refuses_it() -> None:
    detected = detect_quant(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "format": "nvfp4-pack-quantized",
                        "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                        "input_activations": None,
                    }
                },
            },
        }
    )
    assert detected.format == "nvfp4_w4a16"
    assert detected.format != "nvfp4"
    with pytest.raises(QuantMismatch, match="W4A16"):
        assert_flag_matches("nvfp4", detected)
    assert "W4A16" in (compatibility_error("vllm", detected) or "")


def test_dgpp_mixed_without_both_formats_is_not_mixed() -> None:
    fp8_only = detect_quant(
        {
            "model_type": "flint",
            "architectures": ["FlintForCausalLM"],
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "dgpp_mixed",
                "recipe": "fp8",
                "activations": "bfloat16",
                "routed_experts": {
                    "precision_by_layer": {
                        "0": {"gate": "fp8", "up": "fp8", "down": "bf16"},
                    }
                },
            },
        }
    )
    assert fp8_only.format != "mixed"
    with pytest.raises(QuantMismatch, match="both FP8 and NVFP4"):
        assert_flag_matches("mixed", fp8_only)
    no_map = detect_quant(
        {
            "model_type": "flint",
            "architectures": ["FlintForCausalLM"],
            "quantization_config": {"quant_method": "dgpp_mixed", "recipe": "fp8"},
        }
    )
    assert no_map.format != "mixed"
    with pytest.raises(QuantMismatch, match="--mixed"):
        assert_flag_matches("mixed", no_map)


def test_engine_compatibility_follows_quant_method() -> None:
    mixed = detect_quant(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "a": {
                        "weights": {"num_bits": 8, "type": "float"},
                        "input_activations": {"num_bits": 8, "type": "float"},
                    },
                    "b": {
                        "format": "nvfp4-pack-quantized",
                        "weights": {"num_bits": 4, "type": "float"},
                        "input_activations": {"num_bits": 4, "type": "float"},
                    },
                },
            },
        }
    )
    assert mixed.format == "mixed"
    assert compatibility_error("vllm", mixed) is None
    assert "compressed-tensors" in (compatibility_error("sglang", mixed) or "")
    dgpp = detect_quant(
        {
            "model_type": "flint",
            "architectures": ["FlintForCausalLM"],
            "quantization_config": {
                "quant_method": "dgpp_mixed",
                "routed_experts": {
                    "precision_by_layer": {"0": {"gate": "fp8", "up": "nvfp4", "down": "fp8"}}
                },
            },
        }
    )
    assert "dgpp_mixed" in (compatibility_error("sglang", dgpp) or "")
    assert "dgpp_mixed" in (compatibility_error("vllm", dgpp) or "")


def test_sglang_dgpp_is_refused_before_the_snapshot_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sma.pipeline import HubFetcher

    cfg = make_config(tmp_path)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "dgpp_mixed",
            "routed_experts": {
                "precision_by_layer": {"0": {"gate": "fp8", "up": "nvfp4", "down": "fp8"}}
            },
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class _Info:
        sha = "abc"
        siblings: list[object] = []

    class _Api:
        def __init__(self, token: str | None = None) -> None:
            del token

        def model_info(self, repo_id: str, revision: str | None = None) -> _Info:
            del repo_id, revision
            return _Info()

    def _download(**_kwargs: object) -> str:
        return str(config_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot download ran after a refused quant")

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    monkeypatch.setattr("sma.pipeline.download_snapshot", _boom)
    fetcher = HubFetcher(cfg, registry=lambda _engine, _image: {"Qwen3ForCausalLM"}, engine="sglang")
    with pytest.raises(SmaError, match="dgpp_mixed"):
        fetcher.fetch("org/mixed", None, "mixed")


def test_registry_query_pulls_before_it_runs_the_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _ok(argv: list[str], **_kwargs: object) -> CommandResult:
        calls.append(list(argv))
        if argv[:2] == ["docker", "pull"]:
            return CommandResult(argv, 0, "", "Downloaded newer image\n")
        if argv[:3] == ["docker", "image", "inspect"]:
            return CommandResult(argv, 0, '["example@sha256:abc"]', "")
        if "run" in argv:
            image = argv[argv.index("--entrypoint") + 2]
            assert image == "example:tag@sha256:abc"
            return CommandResult(argv, 0, '["LlamaForCausalLM"]\n', "")
        return CommandResult(argv, 1, "", "unexpected")

    monkeypatch.setattr("sma.preflight.run_command", _ok)
    monkeypatch.setattr("sma.preflight.run_command_streaming", _ok)
    archs, digest = query_container_archs("vllm", "example:tag", pull_timeout_s=10, run_timeout_s=10)
    assert archs == {"LlamaForCausalLM"}
    assert digest == "sha256:abc"
    assert calls[0][:2] == ["docker", "pull"]
    assert calls[1][:3] == ["docker", "image", "inspect"]
    assert calls[2][1] == "run"


def test_bundled_fallback_warns_and_continues_but_flint_is_refused(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path, model_type="llama", architectures=["SomeFutureForCausalLM"])
    from sma.fetch import inspect_snapshot

    detected = inspect_snapshot(snapshot, repo="org/future", revision_sha="abc", quant_flag="bf16").detected
    warned = check_architecture(
        engine="vllm",
        detected=detected,
        config={"model_type": "llama", "architectures": ["SomeFutureForCausalLM"]},
        repo_files=[],
        trust_remote_code=False,
        registry=lambda _engine, _image: None,
        image="example:tag",
    )
    assert warned.ok is True
    assert warned.source == "bundled"
    assert "WARNING" in warned.message
    flint = check_architecture(
        engine="sglang",
        detected=detect_quant({"model_type": "flint", "architectures": ["FlintForCausalLM"], "torch_dtype": "bfloat16"}),
        config={"model_type": "flint", "architectures": ["FlintForCausalLM"]},
        repo_files=["sglang/flint_sglang-0.1.4-py3-none-any.whl"],
        trust_remote_code=False,
        registry=lambda _engine, _image: None,
        image="example:tag",
    )
    assert flint.ok is False
    assert "Flint" in flint.message
    assert "wheel" in flint.message


def test_sparse_telemetry_is_unverified_without_using_the_failure_counter() -> None:
    samples = [
        Sample(0.0, 40.0, 100.0, False, "speed"),
        Sample(10.0, 40.0, 100.0, False, "speed"),
    ]
    problem = coverage_problem(
        samples, start=0.0, end=10.0, interval_s=1.0, min_ratio=0.8, max_gap_s=2.5
    )
    assert problem is not None
    dense = [Sample(float(i), 40.0, 80.0, False, "speed") for i in range(11)]
    assert coverage_problem(dense, start=0.0, end=10.0, interval_s=1.0, min_ratio=0.8, max_gap_s=2.5) is None


def test_sglang_jsonl_last_line_and_missing_p90_from_per_request_data(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    first = {"median_ttft_ms": 1, "p99_ttft_ms": 9}
    second = {
        "median_ttft_ms": 10,
        "p99_ttft_ms": 40,
        "median_tpot_ms": 5,
        "p99_tpot_ms": 9,
        "median_itl_ms": 5,
        "p99_itl_ms": 9,
        "median_e2e_latency_ms": 100,
        "p90_e2e_latency_ms": 150,
        "p99_e2e_latency_ms": 200,
        "output_throughput": 12,
        "ttfts": [0.01, 0.02, 0.03, 0.04],
        "itls": [[0.004, 0.006], [0.005, 0.005], [0.007], [0.009]],
    }
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    payload = load_bench(path)
    assert payload is not None
    assert payload["median_ttft_ms"] == 10
    assert percentile_ms(payload, "ttft", 50) == pytest.approx(10)
    assert percentile_ms(payload, "ttft", 90) is not None
    assert percentile_ms(payload, "e2el", 90) == pytest.approx(150)
    assert any("p90_ttft_ms" in note for note in payload["percentile_notes"])
    missing = {"median_ttft_ms": 10, "p99_ttft_ms": 40, "median_tpot_ms": 5, "p99_tpot_ms": 8}
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(missing), encoding="utf-8")
    parsed = load_bench(bare)
    assert parsed is not None
    assert percentile_ms(parsed, "ttft", 90) is None
    assert any("null" in note for note in parsed["percentile_notes"])


def test_corrupt_pending_job_is_quarantined(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    (queue.path / "pending" / "job-bad.json").write_text("{", encoding="utf-8")
    good = queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    claimed = queue.claim("worker", "gpu-host-1")
    assert claimed is not None
    assert claimed[1]["job_id"] == good["job_id"]
    assert (queue.path / "corrupt" / "job-bad.json").is_file()
    assert (queue.path / "corrupt" / "job-bad.json.error").is_file()


def test_reassign_starts_fresh_and_blocks_the_old_host(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    job = queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    path, claimed = queue.claim("worker", "gpu-host-1")
    claimed["pinned_host_alias"] = "gpu-host-1"
    claimed["legs"][0]["run_id"] = "run-old"
    claimed["legs"][0]["status"] = "failed"
    from sma.atomic import atomic_write_json

    atomic_write_json(path, claimed)
    queue.reclaim_stale(0, grace_s=0)
    moved = queue.reassign(job["job_id"], "gpu-host-2")
    assert moved["pinned_host_alias"] == "gpu-host-2"
    assert moved["legs"][0]["run_id"] is None
    assert moved["legs"][0]["status"] == "pending"
    assert queue.claim("old", "gpu-host-1") is None
    again = queue.claim("new", "gpu-host-2")
    assert again is not None
    assert again[1]["legs"][0]["run_id"] is None


def test_retry_waits_and_sorts_behind_a_fresh_job(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    first = queue.add([{"repo": "org/old", "quant": "bf16"}], kind="single")
    path, claimed = queue.claim("worker", "gpu-host-1")
    claimed["attempts"] = 1
    queue.release_to_pending(path, claimed, delay_s=3600)
    second = queue.add([{"repo": "org/new", "quant": "bf16"}], kind="single")
    nxt = queue.claim("worker", "gpu-host-1")
    assert nxt is not None
    assert nxt[1]["job_id"] == second["job_id"]
    assert nxt[1]["job_id"] != first["job_id"]


def test_compare_labels_a_different_image_digest(tmp_path: Path) -> None:
    import math

    from sma.compare import compare_run_dirs

    def write(name: str, digest: str) -> None:
        run = tmp_path / name
        (run / "logprobs").mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "tokenizer_fingerprint": "sha256:same",
                    "architecture": "llama",
                    "quant_flag": "bf16",
                    "engine": {"name": "vllm", "image_digest": digest},
                }
            ),
            encoding="utf-8",
        )
        dump = {
            "corpus_version": "sma-corpus-v3",
            "window_tokens": 2,
            "top_k": 1,
            "perplexity": 2.0,
            "windows": [
                {
                    "index": 0,
                    "token_ids": [1, 2],
                    "positions": [
                        {
                            "index": 1,
                            "true_token_id": 2,
                            "true_logprob": math.log(0.5),
                            "top": [{"id": 2, "logprob": math.log(0.5)}],
                        }
                    ],
                }
            ],
        }
        (run / "logprobs" / "dump.json").write_text(json.dumps(dump), encoding="utf-8")

    write("a", "sha256:aaa")
    write("b", "sha256:bbb")
    result = compare_run_dirs(tmp_path / "a", tmp_path / "b")
    assert result["image_digest_match"] is False
    assert any("IMAGE DIGEST" in note for note in result["notes"])
