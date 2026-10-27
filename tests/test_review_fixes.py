import json
import math
from pathlib import Path

import pytest

from sma.commands import serve_argv
from sma.config import default_config
from sma.corpus import CORPUS_VERSION, load_corpus
from sma.engines import compatibility_error
from sma.errors import QuantMismatch, SmaError
from sma.kl import kl_position
from sma.pipeline import RunSpec, execute
from sma.preflight import check_architecture
from sma.prom import diff_samples, parse_prometheus
from sma.quant import assert_flag_matches, detect_quant
from sma.queue import JobQueue, run_worker
from sma.records import load_manifest
from sma.telemetry import Sample, integrate_joules
from sma.records import _energy
from tests.fakes import write_snapshot
from tests.harness import make_config, make_services


def test_compressed_tensors_nvfp4_is_nvfp4() -> None:
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
                        "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                    }
                },
            },
        }
    )
    assert detected.format == "nvfp4"
    assert assert_flag_matches("nvfp4", detected) == "nvfp4"
    assert compatibility_error("vllm", detected) is None
    assert "compressed-tensors" in (compatibility_error("sglang", detected) or "")


def test_dgpp_mixed_counts_and_refuses_nvfp4() -> None:
    detected = detect_quant(
        {
            "model_type": "flint",
            "architectures": ["FlintForCausalLM"],
            "torch_dtype": "bfloat16",
            "num_experts": 256,
            "quantization_config": {
                "quant_method": "dgpp_mixed",
                "activations": "bfloat16",
                "routed_experts": {
                    "precision_by_layer": {
                        "2": {"gate": "fp8", "up": "fp8", "down": "fp8"},
                        "12": {"gate": "fp8", "up": "nvfp4", "down": "fp8"},
                        "31": {"gate": "nvfp4", "up": "nvfp4", "down": "nvfp4"},
                    }
                },
            },
        }
    )
    assert detected.format == "mixed"
    assert detected.composition["formats"] == {"fp8": 5, "nvfp4": 4}
    assert detected.composition["activations"] == "bfloat16"
    with pytest.raises(QuantMismatch, match="--mixed"):
        assert_flag_matches("nvfp4", detected)
    assert assert_flag_matches("mixed", detected) == "mixed"
    assert "dgpp_mixed" in (compatibility_error("vllm", detected) or "")
    assert "dgpp_mixed" in (compatibility_error("sglang", detected) or "")


def test_flint_preflight_fails_on_vllm_and_stock_sglang(tmp_path: Path) -> None:
    snapshot = write_snapshot(
        tmp_path,
        model_type="flint",
        architectures=["FlintForCausalLM"],
    )
    (snapshot / "sglang").mkdir()
    (snapshot / "sglang" / "flint_sglang-0.1.4-py3-none-any.whl").write_bytes(b"wheel")
    from sma.fetch import inspect_snapshot

    result = inspect_snapshot(snapshot, repo="RecursiveSI/flint", revision_sha="abc", quant_flag="bf16")
    files = [path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()]
    vllm = check_architecture(
        engine="vllm",
        detected=result.detected,
        config=result.config,
        repo_files=files,
        trust_remote_code=False,
        registry=lambda _engine, _image: {"LlamaForCausalLM", "Qwen3ForCausalLM"},
        image="nvcr.io/nvidia/vllm:test",
    )
    assert vllm.ok is False
    assert "--engine sglang" in vllm.message
    assert "FlintForCausalLM" in vllm.message
    sglang = check_architecture(
        engine="sglang",
        detected=result.detected,
        config=result.config,
        repo_files=files,
        trust_remote_code=False,
        registry=lambda _engine, _image: {"LlamaForCausalLM"},
        image="lmsysorg/sglang:test",
    )
    assert sglang.ok is False
    assert "flint_sglang" in sglang.message


def test_quality_serve_disables_prefix_cache_and_sglang_uses_flashinfer() -> None:
    cfg = default_config()
    flags = serve_argv(
        cfg,
        mode="quality",
        served_name="Qwen/Qwen3-8B",
        is_moe=False,
        quant_flag="bf16",
        moe_backend=None,
        batch_invariant=True,
    )
    assert "--no-enable-prefix-caching" in flags
    assert "--tensor-parallel-size" in flags
    sglang = serve_argv(
        cfg,
        mode="quality",
        served_name="Qwen/Qwen3-8B",
        is_moe=False,
        quant_flag="bf16",
        moe_backend=None,
        batch_invariant=False,
        engine="sglang",
    )
    assert "--attention-backend" in sglang
    assert sglang[sglang.index("--attention-backend") + 1] == "flashinfer"
    assert "--disable-radix-cache" in sglang
    assert "--tp-size" in sglang
    assert sglang[sglang.index("--tp-size") + 1] == "1"
    speed = serve_argv(
        cfg,
        mode="speed",
        served_name="Qwen/Qwen3-8B",
        is_moe=False,
        quant_flag="bf16",
        moe_backend=None,
        batch_invariant=False,
        engine="sglang",
    )
    assert "--disable-radix-cache" not in speed


def test_labeled_metrics_match_without_the_label_string() -> None:
    before = parse_prometheus('vllm:generation_tokens_total{model_name="m",engine="0"} 3\n')
    after = parse_prometheus('vllm:generation_tokens_total{model_name="m",engine="0"} 8\n')
    diff = diff_samples(before, after)
    assert diff["watched"]["vllm:generation_tokens_total"] == 5


def test_energy_sums_windows_and_does_not_bridge_the_gap() -> None:
    def span(start: float, end: float, power: float, phase: str) -> list[Sample]:
        rows = []
        cursor = start
        while cursor <= end + 1e-9:
            rows.append(Sample(cursor, 40.0, power, False, phase))
            cursor += 1.0
        return rows

    samples = span(0.0, 10.0, 100.0, "a") + span(100.0, 110.0, 50.0, "b")
    windows = {
        "windows": [
            {"valid": True, "start_unix": 0.0, "end_unix": 10.0},
            {"valid": True, "start_unix": 100.0, "end_unix": 110.0},
        ]
    }
    joules, method = _energy(samples, windows, max_gap_s=2.5)
    assert joules == pytest.approx(1500.0)
    assert method == "nvidia-smi-1hz-trapezoid-per-window"
    bridged = integrate_joules(samples, max_gap_s=1000)
    assert bridged is not None and bridged > 2000
    assert integrate_joules(samples, max_gap_s=2.5) == pytest.approx(1500.0)


def test_corpus_is_natural_text_and_hashed() -> None:
    text, meta = load_corpus(CORPUS_VERSION)
    assert meta["license"] == "public-domain"
    assert meta["commercial_use"] is True
    assert meta["dataset"] == "project-gutenberg-1342"
    assert meta["revision"] == "1342-0.txt"
    assert meta["sha256"]
    assert meta["words"] > 70_000
    assert text.startswith("It is a truth universally acknowledged")
    assert "Chapter I." not in text[:40]
    assert meta["version"] == "sma-corpus-v3"
    assert "illustration" in meta["slice"]


def test_worker_skips_claim_when_the_lock_is_held(tmp_path: Path) -> None:
    from sma.locks import GpuLock

    cfg = default_config()
    cfg.queue.dir = str(tmp_path / "q")
    cfg.lock_path = str(tmp_path / "gpu.lock")
    queue = JobQueue(cfg.queue_path())
    queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    held = GpuLock(cfg.lock_file())
    held.acquire()
    try:
        with pytest.raises(SmaError, match="GPU lock"):
            run_worker(cfg, lambda _cfg: None, once=True, executor=lambda *a, **k: "x", spec_cls=RunSpec)
    finally:
        held.release()
    assert list((cfg.queue_path() / "pending").glob("*.json"))
    assert list((cfg.queue_path() / "claimed").glob("*.json")) == []


def test_worker_continues_after_a_bad_model(tmp_path: Path) -> None:
    cfg = default_config()
    cfg.queue.dir = str(tmp_path / "q")
    cfg.lock_path = str(tmp_path / "gpu.lock")
    cfg.queue.max_attempts = 1
    queue = JobQueue(cfg.queue_path())
    queue.add([{"repo": "org/bad", "quant": "bf16"}], kind="single")
    queue.add([{"repo": "org/good", "quant": "bf16"}], kind="single")
    seen: list[str] = []

    def fake(spec: RunSpec, _cfg, _services, on_created=None) -> str:
        seen.append(spec.repo)
        if spec.repo == "org/bad":
            raise SmaError("bad model")
        if on_created:
            on_created("run-good")
        return "run-good"

    completed = run_worker(
        cfg,
        lambda _cfg: None,
        once=False,
        exit_when_idle=True,
        executor=fake,
        spec_cls=RunSpec,
    )
    assert completed == 1
    assert set(seen) == {"org/bad", "org/good"}
    assert list((cfg.queue_path() / "failed").glob("*.json"))
    assert list((cfg.queue_path() / "done").glob("*.json"))


def test_reclaimed_job_stays_on_the_host_that_started_it(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    job = queue.add([{"repo": "org/a", "quant": "bf16"}, {"repo": "org/b", "quant": "nvfp4"}], kind="pair")
    path, claimed = queue.claim("w", "gpu-host-1")
    assert claimed["heartbeat"]
    on_disk = json.loads(path.read_text())
    assert on_disk["heartbeat"]
    claimed["pinned_host_alias"] = "gpu-host-1"
    claimed["heartbeat"] = "2000-01-01T00:00:00Z"
    from sma.atomic import atomic_write_json

    atomic_write_json(path, claimed)
    moved = queue.reclaim_stale(10)
    assert moved == [job["job_id"]]
    assert queue.claim("other", "gpu-host-2") is None
    again = queue.claim("same", "gpu-host-1")
    assert again is not None
    assert again[1]["pinned_host_alias"] == "gpu-host-1"


def test_missing_telemetry_marks_speed_unverified(tmp_path: Path) -> None:
    from sma.telemetry import TelemetryRecorder

    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)

    def quiet(run_dir: Path, config):
        recorder = TelemetryRecorder(run_dir / "telemetry.csv", config)
        recorder.start(sampler=False)
        return recorder

    services.telemetry_factory = quiet
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = json.loads(next((tmp_path / "runs").glob(f"*/*/*/{run_id}/summary.json")).read_text())
    assert summary["speed"]["prefill_tok_s"] is None
    assert all(cell["validity"] == "unverified" for cell in summary["speed"]["cells"])
    assert runtime.running == set()


def test_na_throttle_flags_are_unverified_and_power_cap_is_configurable(tmp_path: Path) -> None:
    from sma.telemetry import TelemetryRecorder

    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def na_factory(run_dir: Path, config):
        recorder = TelemetryRecorder(run_dir / "telemetry.csv", config)
        recorder.start(sampler=False)

        def set_phase(phase: str) -> None:
            TelemetryRecorder.set_phase(recorder, phase)
            if phase.startswith("speed:") and ":a" in phase:
                recorder.append_parsed(
                    "1500, 45, 140, [N/A], [N/A], [N/A], [N/A]",
                    phase=phase,
                )

        recorder.set_phase = set_phase  # type: ignore[method-assign]
        return recorder

    services.telemetry_factory = na_factory
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = json.loads(next((tmp_path / "runs").glob(f"*/*/*/{run_id}/summary.json")).read_text())
    assert summary["speed"]["cells"][0]["validity"] == "unverified"
    assert summary["speed"]["prefill_tok_s"] is None

    cfg2 = make_config(tmp_path / "cap")
    cfg2.telemetry.sw_power_cap_invalidates = False
    services2, _f, _r = make_services(tmp_path / "cap", cfg2)

    def cap_factory(run_dir: Path, config):
        recorder = TelemetryRecorder(run_dir / "telemetry.csv", config)
        recorder.start(sampler=False)

        def set_phase(phase: str) -> None:
            TelemetryRecorder.set_phase(recorder, phase)
            if phase.startswith("speed:") and ":a" in phase:
                recorder.append_parsed(
                    "1500, 45, 140, Not Active, Not Active, Not Active, Active",
                    phase=phase,
                )

        recorder.set_phase = set_phase  # type: ignore[method-assign]
        return recorder

    services2.telemetry_factory = cap_factory
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg2, services2)
    summary = json.loads(next((tmp_path / "cap" / "runs").glob(f"*/*/*/{run_id}/summary.json")).read_text())
    assert summary["speed"]["cells"][0]["validity"] == "valid"
    assert summary["speed"]["prefill_tok_s"] is not None


def test_compare_accepts_bf16_vs_mixed_and_refuses_cross_engine(tmp_path: Path) -> None:
    from sma.compare import compare_run_dirs

    def write_run(name: str, quant: str, engine: str) -> None:
        run = tmp_path / name
        (run / "logprobs").mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": name,
                    "tokenizer_fingerprint": "sha256:same",
                    "architecture": "llama",
                    "quant_flag": quant,
                    "engine": {"name": engine},
                }
            ),
            encoding="utf-8",
        )
        dump = {
            "corpus_version": "sma-corpus-v2",
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

    write_run("bf16", "bf16", "vllm")
    write_run("mixed", "mixed", "vllm")
    write_run("sglang", "bf16", "sglang")
    from sma.errors import CompareRefused

    result = compare_run_dirs(tmp_path / "bf16", tmp_path / "mixed")
    assert result["cross_engine"] is False
    with pytest.raises(CompareRefused, match="allow-cross-engine"):
        compare_run_dirs(tmp_path / "bf16", tmp_path / "sglang")
    labeled = compare_run_dirs(tmp_path / "bf16", tmp_path / "sglang", allow_cross_engine=True)
    assert labeled["cross_engine"] is True
    assert "CROSS-ENGINE" in labeled["notes"][0]


def test_run_records_digest_phase_backends_and_metrics(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.speed.derive_stat = "p90"
    services, _fetcher, runtime = make_services(tmp_path, cfg, quant="compressed-nvfp4")
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="nvfp4"), cfg, services)
    manifest = load_manifest(next((tmp_path / "runs").glob(f"*/*/*/{run_id}")))
    summary = json.loads((next((tmp_path / "runs").glob(f"*/*/*/{run_id}")) / "summary.json").read_text())
    assert manifest["quant_detected"] == "nvfp4"
    assert manifest["engine"]["image_digest"].startswith("sha256:")
    joined = " ".join(runtime.specs[0].docker_argv)
    assert "@sha256:" in joined
    assert "--no-enable-prefix-caching" in manifest["engine"]["flags_quality"]
    assert manifest["engine"]["moe_backend_quality"] is None
    assert summary["speed"]["prefill_tok_s"] == pytest.approx(2048 / 0.110)
    cell = next((tmp_path / "runs").glob(f"*/*/*/{run_id}/speed/cells/*/attempt-1/metrics.json"))
    diff = json.loads(cell.read_text())["diff"]
    assert diff["watched"]["vllm:generation_tokens_total"] == 1
    assert diff["missing_reason"]
    assert "vllm:generation_tokens_total" not in diff["missing_reason"]
    assert manifest["engine"]["deterministic_quality"]["prefix_cache"] is False
    assert summary["thermal"]["sw_power_cap_invalidates"] is False
    assert summary["speed"]["unverified_cells"] == 0
    assert (next((tmp_path / "runs").glob(f"*/*/*/{run_id}")) / "logprobs" / "corpus.json").is_file()
    corpus = json.loads((next((tmp_path / "runs").glob(f"*/*/*/{run_id}/logprobs/corpus.json"))).read_text())
    assert corpus["corpus_sha256"]
    assert corpus["corpus_license"] == "public-domain"


def test_compressed_tensors_fp8_and_nvfp4_is_mixed() -> None:
    detected = detect_quant(
        {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "fp8": {
                        "format": "float-quantized",
                        "weights": {"num_bits": 8, "type": "float", "group_size": 128},
                    },
                    "nvfp4": {
                        "format": "nvfp4-pack-quantized",
                        "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                        "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                    },
                },
            },
        }
    )
    assert detected.format == "mixed"
    assert set(detected.composition["formats"]) == {"fp8", "nvfp4"}
    with pytest.raises(QuantMismatch, match="--mixed"):
        assert_flag_matches("nvfp4", detected)


def test_sglang_backend_contract() -> None:
    from sma.backends import get_backend
    from sma.commands import BenchCell

    cfg = default_config()
    backend = get_backend("sglang")
    assert backend.health_paths() == ("/health", "/v1/models")
    assert backend.metrics_path() == "/metrics"
    det = backend.deterministic(cfg)
    assert det["attention_backend"] == "flashinfer"
    assert det["radix_cache"] is False
    assert det["prefix_cache"] is False
    cell = BenchCell("prefill_c1_r1", "prefill", 8, 4, 1, 1)
    argv = backend.bench_argv(
        cfg,
        container="sma-test-speed",
        served_name="org/model",
        cell=cell,
        result_dir="/sma-run/speed/cells/prefill_c1_r1/attempt-1",
    )
    assert "sglang.bench_serving" in argv
    assert "vllm" not in argv
    empty = diff_samples({}, {}, watched=backend.watched_metrics())
    assert empty["watched"]["sglang:generation_tokens_total"] is None
    assert "null" in (empty["missing_reason"] or "")


def test_fresh_heartbeat_none_is_not_reclaimed(tmp_path: Path) -> None:
    import os
    import time

    from sma.atomic import atomic_write_json

    queue = JobQueue(tmp_path)
    job = queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    path, claimed = queue.claim("w", "gpu-host-1")
    claimed["heartbeat"] = None
    claimed["claim"] = None
    atomic_write_json(path, claimed)
    assert queue.reclaim_stale(10, grace_s=30) == []
    old = time.time() - 120
    os.utime(path, (old, old))
    assert queue.reclaim_stale(10, grace_s=30) == [job["job_id"]]


def test_pair_job_refuses_two_engines(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    with pytest.raises(SmaError, match="same engine"):
        queue.add(
            [
                {"repo": "org/a", "quant": "bf16", "engine": "vllm"},
                {"repo": "org/b", "quant": "mixed", "engine": "sglang"},
            ],
            kind="pair",
        )


def test_failed_telemetry_with_samples_is_unverified(tmp_path: Path) -> None:
    from sma.telemetry import TelemetryRecorder

    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def failed_factory(run_dir: Path, config):
        recorder = TelemetryRecorder(run_dir / "telemetry.csv", config)
        recorder.start(sampler=False)
        recorder.failures = 1

        def set_phase(phase: str) -> None:
            TelemetryRecorder.set_phase(recorder, phase)
            if phase.startswith("speed:") and ":a" in phase:
                recorder.append_parsed(
                    "1500, 45, 80, Not Active, Not Active, Not Active, Not Active",
                    phase=phase,
                )

        recorder.set_phase = set_phase  # type: ignore[method-assign]
        return recorder

    services.telemetry_factory = failed_factory
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = json.loads(next((tmp_path / "runs").glob(f"*/*/*/{run_id}/summary.json")).read_text())
    # A later successful sample resets the failure counter, so the counter is not
    # what makes a cell unverified. This short window has a real sample.
    assert summary["speed"]["cells"][0]["validity"] == "valid"


def test_resume_on_another_alias_is_refused_and_warmup_reruns(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    warmup = json.loads((run_dir / "speed" / "warmup.json").read_text())
    import time

    time.sleep(1.1)
    runtime.running.clear()
    if runtime.server is not None:
        runtime.server.stop()
        runtime.server = None
    (run_dir / "steps" / "serve_speed.done").unlink()
    (run_dir / "steps" / "speed.done").unlink()
    execute(
        RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_id),
        cfg,
        services,
    )
    again = json.loads((run_dir / "speed" / "warmup.json").read_text())
    assert again["finished"] != warmup["finished"]
    cfg.host_alias = "gpu-host-2"
    with pytest.raises(SmaError, match="pinned"):
        execute(
            RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_id),
            cfg,
            services,
        )
