import json
import socket
from pathlib import Path

import pytest

from sma.errors import QuantMismatch, SmaError
from sma.pipeline import RunSpec, SimulatedKill, execute
from sma.records import load_manifest
from sma.schemautil import validate_instance
from tests.fakes import write_snapshot
from tests.harness import make_config, make_services


def _run(tmp_path: Path, **kwargs):
    cfg = make_config(tmp_path)
    services, fetcher, runtime = make_services(tmp_path, cfg, **kwargs)
    run_id = execute(
        RunSpec(repo="example/tiny", quant_flag=kwargs.get("quant", "bf16")),
        cfg,
        services,
    )
    return cfg, services, fetcher, runtime, run_id


def test_full_run_writes_a_valid_record_and_stops_the_server(tmp_path: Path) -> None:
    cfg, services, _fetcher, runtime, run_id = _run(tmp_path)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    manifest = load_manifest(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    validate_instance(manifest, "manifest.schema.json")
    validate_instance(summary, "summary.schema.json")
    assert manifest["revision_sha"] == "a" * 40
    assert manifest["engine"]["tensor_parallel_size"] == 1
    assert manifest["engine"]["image_digest"] == runtime.digest
    assert manifest["engine"]["batch_invariant_quality"] is True
    assert "VLLM_BATCH_INVARIANT" in manifest["engine"]["env_quality"]
    assert "--no-enable-flashinfer-autotune" in manifest["engine"]["flags_quality"]
    assert "--moe-backend" not in manifest["engine"]["flags_quality"]
    assert manifest["steps"]["fetch"]["status"] == "done"
    assert manifest["steps"]["teardown"]["status"] == "done"
    assert (run_dir / "steps" / "fetch.done").is_file()
    assert (run_dir / "determinism.json").is_file()
    assert json.loads((run_dir / "determinism.json").read_text())["passed"] is True
    assert (run_dir / "logprobs" / "dump.json").is_file()
    assert summary["perplexity"] is not None
    assert summary["determinism_passed"] is True
    assert summary["speed"]["prefill_tok_s"] == pytest.approx(2048 / 0.1)
    assert summary["speed"]["decode_tok_s"] == pytest.approx(50.0)
    assert summary["speed"]["output_tok_s_c4"] == pytest.approx(12.5)
    assert summary["quality"]["ifeval"]["scorer"] == "nemo-evaluator 0.3.0"
    assert summary["quality"]["ifeval"]["score"] == 0.5
    assert (summary["quality"]["ifeval"]["ci_lower"], summary["quality"]["ifeval"]["n"]) == (0.25, 2)
    assert runtime.running == set()
    assert runtime.stop_count >= 1
    text = (run_dir / "manifest.json").read_text()
    assert socket.gethostname() not in text


def test_quant_mismatch_does_not_start_vllm(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg, quant="nvfp4")
    with pytest.raises(QuantMismatch, match="--bf16"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    assert runtime.specs == []
    assert runtime.running == set()
    manifest_path = next((tmp_path / "runs").glob("*/*/*/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["steps"]["fetch"]["status"] == "failed"
    assert "NVFP4" in manifest["steps"]["fetch"]["error"] or "nvfp4" in manifest["steps"]["fetch"]["error"]
    assert manifest["steps"]["teardown"]["status"] == "done"


def test_resume_skips_completed_steps(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, fetcher, runtime = make_services(tmp_path, cfg)
    services.hooks.kill_step = "determinism"
    with pytest.raises(SimulatedKill, match="killed mid-step"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    assert fetcher.calls == 1
    assert runtime.running == set()
    run_dir = next((tmp_path / "runs").glob("*/*/*/*"))
    run_id = run_dir.name
    assert not (run_dir / "steps" / "determinism.done").is_file()
    assert (run_dir / "steps" / "fetch.done").is_file()
    execute(
        RunSpec(repo="example/tiny", quant_flag="bf16", resume_run_id=run_id),
        cfg,
        services,
    )
    assert fetcher.calls == 1
    manifest = load_manifest(run_dir)
    assert manifest["steps"]["fetch"]["attempts"] == 1
    assert manifest["steps"]["determinism"]["status"] == "done"
    assert manifest["steps"]["determinism"]["attempts"] == 2
    assert manifest["timestamps"]["finished"]
    assert runtime.running == set()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "done"


def test_a_failed_quality_step_does_not_stop_the_run_and_the_server_is_torn_down(tmp_path: Path) -> None:
    # Owner decision 2026-10-29: a failed score or suite is recorded, and the other steps still run.
    from sma.errors import RunIncomplete

    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)

    def explode(*_args, **_kwargs):
        raise SmaError("nemo-evaluator exploded")

    services.evaluator.run = explode
    with pytest.raises(RunIncomplete, match="these failed: quality"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    assert runtime.running == set()
    assert runtime.stop_count >= 1
    run_dir = next((tmp_path / "runs").glob("*/*/*/*"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["steps"]["quality"]["status"] == "failed"
    assert "quality suite(s) failed: ifeval" in manifest["steps"]["quality"]["error"]
    assert manifest["steps"]["speed"]["status"] == "done"
    assert manifest["steps"]["teardown"]["status"] == "done"
    assert manifest["timestamps"]["finished"]
    scores = json.loads((run_dir / "quality" / "ifeval" / "scores.json").read_text())
    assert scores["primary_score"] is None and "nemo-evaluator exploded" in scores["error"]
    assert not (run_dir / "quality" / "ifeval" / ".done").exists()


def test_one_failed_quality_suite_does_not_stop_the_next(tmp_path: Path) -> None:
    from sma.config import QualitySuite
    from sma.errors import RunIncomplete

    cfg = make_config(tmp_path)
    cfg.quality.suites = [QualitySuite("ifeval", 2, bench="lm-eval://ifeval"), QualitySuite("gsm8k", 2, bench="sma:gsm8k")]
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run = services.evaluator.run

    def ifeval_fails(argv, output_dir, *args, **kwargs):
        if output_dir.name == "ifeval":
            raise SmaError("ifeval's harness crashed")
        return run(argv, output_dir, *args, **kwargs)

    services.evaluator.run = ifeval_fails
    with pytest.raises(RunIncomplete):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob("*/*/*/*"))
    assert (run_dir / "quality" / "gsm8k" / ".done").is_file()
    assert json.loads((run_dir / "quality" / "gsm8k" / "scores.json").read_text())["primary_score"] is not None


def test_a_failed_fetch_still_stops_the_run(tmp_path: Path) -> None:
    # Without the checkpoint nothing after it can run.
    cfg = make_config(tmp_path)
    services, fetcher, _runtime = make_services(tmp_path, cfg)

    def no_checkpoint(*_args, **_kwargs):
        raise SmaError("the repo is gone")

    fetcher.fetch = no_checkpoint
    with pytest.raises(SmaError, match="the repo is gone"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)


def test_moe_nvfp4_falls_back_and_records_the_backend(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg, quant="nvfp4", moe=True)
    runtime.fail_backends.add("flashinfer_b12x")
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="nvfp4"), cfg, services)
    manifest = load_manifest(next((tmp_path / "runs").glob(f"*/*/*/{run_id}")))
    assert manifest["engine"]["moe_backend_used"] == "marlin"
    assert manifest["engine"]["moe_backend_fallback"] is True
    assert manifest["is_moe"] is True
    joined = " ".join(manifest["engine"]["flags_quality"])
    assert "--moe-backend marlin" in joined
    assert any("fell back" in warning for warning in manifest["warnings"])
    assert runtime.running == set()


def test_throttled_speed_cell_is_invalid_and_not_a_headline(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.speed.throttle_retries = 1
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def inject(ctx) -> None:
        assert ctx.telemetry is not None
        ctx.telemetry.append_parsed(
            "800, 96, 130, Active, Not Active, Not Active, Active",
        )

    services.hooks.before_speed_cell = inject
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = json.loads(next((tmp_path / "runs").glob(f"*/*/*/{run_id}/summary.json")).read_text())
    assert summary["speed"]["invalid_cells"] > 0
    assert summary["speed"]["prefill_tok_s"] is None
    assert summary["speed"]["decode_tok_s"] is None
    assert any(cell["valid"] is False for cell in summary["speed"]["cells"])


def test_hostname_in_a_step_error_is_scrubbed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda: "secret-host")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_value")
    cfg = make_config(tmp_path)
    services, fetcher, runtime = make_services(tmp_path, cfg)

    def fetch(*_args, **_kwargs):
        fetcher.calls += 1
        raise SmaError("download failed on secret-host using hf_test_token_value")

    services.fetcher.fetch = fetch
    with pytest.raises(SmaError):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    text = next((tmp_path / "runs").glob("*/*/*/*/manifest.json")).read_text()
    assert "secret-host" not in text
    assert "hf_test_token_value" not in text
    assert runtime.specs == []


def test_gated_deltanet_does_not_set_batch_invariance(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    snapshot = write_snapshot(tmp_path / "gdn", model_type="qwen3_5")
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    services.fetcher.snapshot = snapshot
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    manifest = load_manifest(next((tmp_path / "runs").glob(f"*/*/*/{run_id}")))
    assert manifest["engine"]["batch_invariant_quality"] is False
    assert manifest["engine"]["env_quality"] == {}
    assert any("gated delta-net" in warning for warning in manifest["warnings"])
    assert runtime.running == set()
