"""Shannon's final fixes: SGLang prompt guard, and distinct invalidation reasons."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.cli import main
from sma.config import default_config, load_config, validate_config
from sma.errors import SmaError
from sma.pipeline import RunSpec, execute
from sma.telemetry import TelemetryRecorder
from tests.harness import make_config, make_services


def test_sglang_prompt_guard_runs_at_config_load(tmp_path: Path) -> None:
    cfg = default_config()
    cfg.speed.num_prompts = 257
    validate_config(cfg)
    cfg.engine.name = "sglang"
    with pytest.raises(SmaError, match="will not cap"):
        validate_config(cfg)
    path = tmp_path / "sma.yaml"
    path.write_text("engine:\n  name: sglang\nspeed:\n  num_prompts: 257\n", encoding="utf-8")
    with pytest.raises(SmaError, match="will not cap"):
        load_config(path)


def test_cli_engine_sglang_fails_before_a_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "sma.yaml"
    path.write_text("speed:\n  num_prompts: 257\n", encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("the run started")

    monkeypatch.setattr("sma.pipeline.execute", boom)
    code = main(
        [
            "run",
            "Qwen/Qwen3-0.6B",
            "--bf16",
            "--engine",
            "sglang",
            "--config",
            str(path),
        ]
    )
    assert code == 2
    assert "will not cap" in capsys.readouterr().err


def test_throttled_cell_cools_down_and_records_the_reason(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.speed.throttle_retries = 1
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def inject(ctx) -> None:
        assert ctx.telemetry is not None
        ctx.telemetry.append_parsed("800, 96, 130, Active, Not Active, Not Active, Not Active")

    services.hooks.before_speed_cell = inject
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = json.loads(next((tmp_path / "runs").glob(f"*/*/*/{run_id}/summary.json")).read_text())
    cell = summary["speed"]["cells"][0]
    assert cell["invalidation_reason"] == "throttled"
    assert cell["validity"] == "invalid"
    warnings = " ".join(summary["warnings"])
    assert "Cooling down" in warnings
    assert "bench failed" not in warnings


def test_telemetry_gap_is_not_described_as_a_throttle(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def quiet(run_dir: Path, config):
        recorder = TelemetryRecorder(run_dir / "telemetry.csv", config)
        recorder.start(sampler=False)
        return recorder

    services.telemetry_factory = quiet
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    summary = json.loads((run_dir / "summary.json").read_text())
    cell = summary["speed"]["cells"][0]
    assert cell["invalidation_reason"] == "telemetry_gap"
    assert cell["validity"] == "unverified"
    attempts = list(run_dir.glob("speed/cells/prefill_c1_r1/attempt-*"))
    assert len(attempts) == 1
    warnings = " ".join(summary["warnings"])
    assert "telemetry gap" in warnings
    assert "Cooling down" not in warnings
