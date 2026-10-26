"""Round-3 checks: local tokenizer, failed requests, dataset size, quarantine, W4A8."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.backends import BENCH_OFFLINE_ENV
from sma.commands import BenchCell, bench_argv
from sma.config import default_config
from sma.corpus import CORPUS_VERSION, load_corpus
from sma.engines import compatibility_error
from sma.errors import QuantMismatch, SmaError
from sma.pipeline import RunSpec, execute
from sma.quant import assert_flag_matches, detect_quant
from sma.queue import JobQueue
from sma.speedmath import load_bench, normalize_bench, percentile_ms
from tests.fakes import FakeBencher, bench_output_host_path
from tests.harness import make_config, make_services


def _cell() -> BenchCell:
    return BenchCell("prefill_c1_r1", "prefill", 128, 16, 1, 1)


def test_bench_argv_and_env_pin_the_local_tokenizer() -> None:
    cfg = default_config()
    cell = _cell()
    vllm = bench_argv(
        cfg,
        container="sma-box-speed",
        served_name="org/model",
        cell=cell,
        engine="vllm",
        result_dir="/sma-run/speed/cells/prefill_c1_r1/attempt-1",
    )
    assert vllm[vllm.index("--model") + 1] == "org/model"
    assert vllm[vllm.index("--tokenizer") + 1] == "/model"
    assert "HF_HUB_OFFLINE=1" in vllm
    assert "TRANSFORMERS_OFFLINE=1" in vllm
    assert vllm[:2] == ["docker", "exec"]
    assert BENCH_OFFLINE_ENV == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    sglang = bench_argv(
        cfg,
        container="sma-box-speed",
        served_name="org/model",
        cell=cell,
        engine="sglang",
        result_dir="/sma-run/speed/cells/prefill_c1_r1/attempt-1",
    )
    assert sglang[sglang.index("--model") + 1] == "/model"
    assert sglang[sglang.index("--tokenizer") + 1] == "/model"
    assert "HF_HUB_OFFLINE=1" in sglang
    assert "TRANSFORMERS_OFFLINE=1" in sglang
    assert "org/model" not in sglang


def test_failed_request_is_dropped_from_sglang_p90() -> None:
    payload = normalize_bench(
        {
            "ttfts": [0.1, 0.2, 0.0],
            "errors": [None, None, "timeout"],
            "itls": [[0.01], [0.02], []],
        }
    )
    # p90 of the two successful TTFTs, 100 ms and 200 ms.
    assert percentile_ms(payload, "ttft", 90) == pytest.approx(190.0)
    assert any("failed request" in note for note in payload["percentile_notes"])


def test_incomplete_bench_marks_the_cell_invalid_and_retries(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    class ShortBencher(FakeBencher):
        def run(self, argv: list[str], result_path: Path, timeout_s: float, log_path: Path) -> None:
            super().run(argv, result_path, timeout_s, log_path)
            target = bench_output_host_path(argv, result_path)
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["completed"] = 1
            payload["errors"] = ["request failed"]
            target.write_text(json.dumps(payload), encoding="utf-8")

    services.bencher = ShortBencher()
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    attempts = list(run_dir.glob("speed/cells/prefill_c1_r1/attempt-*"))
    assert len(attempts) == 2
    summary = json.loads((run_dir / "summary.json").read_text())
    prefill = next(cell for cell in summary["speed"]["cells"] if cell["name"] == "prefill_c1_r1")
    assert prefill["validity"] == "invalid"
    assert "completed" in prefill["invalid_reason"]
    assert summary["speed"]["prefill_tok_s"] is None


def test_sglang_dataset_refuses_more_prompts_than_rows() -> None:
    cfg = default_config()
    cfg.speed.num_prompts = 257
    with pytest.raises(SmaError, match="will not cap"):
        bench_argv(
            cfg,
            container="sma-box-speed",
            served_name="org/model",
            cell=_cell(),
            engine="sglang",
            result_dir="/sma-run/speed/cells/prefill_c1_r1/attempt-1",
        )


def test_queue_oserror_is_not_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = JobQueue(tmp_path)
    good = queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    missing = queue.path / "pending" / "job-missing.json"
    busy = queue.path / "pending" / "job-busy.json"
    missing.write_text("{}\n", encoding="utf-8")
    busy.write_text("{}\n", encoding="utf-8")
    real_read = __import__("sma.queue", fromlist=["_read"])._read

    def flaky(path: Path) -> dict:
        if path.name == "job-missing.json":
            raise FileNotFoundError(path)
        if path.name == "job-busy.json":
            raise OSError("resource busy")
        return real_read(path)

    monkeypatch.setattr("sma.queue._read", flaky)
    claimed = queue.claim("worker", "gpu-host-1")
    assert claimed is not None
    assert claimed[1]["job_id"] == good["job_id"]
    assert not (queue.path / "corrupt").exists()
    assert busy.is_file()
    assert missing.is_file()


def test_nvfp4_weights_with_fp8_activations_are_w4a8() -> None:
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
                        "input_activations": {"num_bits": 8, "type": "float"},
                    }
                },
            },
        }
    )
    assert detected.format == "nvfp4_w4a8"
    assert "W4A8" in detected.detail
    with pytest.raises(QuantMismatch, match="W4A8"):
        assert_flag_matches("nvfp4", detected)
    assert "W4A8" in (compatibility_error("vllm", detected) or "")


def test_corpus_note_says_the_step_fails_below_48_windows() -> None:
    _text, meta = load_corpus(CORPUS_VERSION)
    note = meta["note"].lower()
    assert "fails" in note
    assert "48" in note
    assert "fewer" in note


def test_load_bench_keeps_a_clean_summary_when_there_are_no_per_request_failures(tmp_path: Path) -> None:
    path = tmp_path / "bench.json"
    path.write_text(
        json.dumps(
            {
                "completed": 2,
                "errors": [],
                "median_ttft_ms": 10,
                "p99_ttft_ms": 40,
                "ttfts": [0.01, 0.02],
            }
        ),
        encoding="utf-8",
    )
    payload = load_bench(path)
    assert payload is not None
    assert percentile_ms(payload, "ttft", 90) == pytest.approx(19.0)
