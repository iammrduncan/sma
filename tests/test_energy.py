import json
from pathlib import Path

import pytest

from sma.pipeline import RunSpec, execute
from sma.records import _energy
from sma.telemetry import Sample
from tests.harness import make_config, make_services


def _flat(start: float, end: float, watts: float) -> list[Sample]:
    return [Sample(start + second, 40.0, watts, False, "speed") for second in range(int(end - start) + 1)]


def test_the_energy_counter_wins_over_sampled_power_when_it_was_read() -> None:
    samples = _flat(0.0, 10.0, 100.0)  # integrates to 1000 J
    windows = {"windows": [{"valid": True, "start_unix": 0.0, "end_unix": 10.0, "energy_j": 950.0}]}

    assert _energy(samples, windows, max_gap_s=2.5) == (950.0, "nvml-energy-counter-per-window")


def test_a_window_without_a_counter_reading_falls_back_and_the_method_says_mixed() -> None:
    samples = _flat(0.0, 10.0, 100.0) + _flat(100.0, 110.0, 50.0)
    windows = {
        "windows": [
            {"valid": True, "start_unix": 0.0, "end_unix": 10.0, "energy_j": 950.0},
            {"valid": True, "start_unix": 100.0, "end_unix": 110.0, "energy_j": None},
            {"valid": False, "start_unix": 0.0, "end_unix": 10.0, "energy_j": 9999.0},
        ]
    }

    joules, method = _energy(samples, windows, max_gap_s=2.5)
    assert joules == pytest.approx(950.0 + 500.0)
    assert method.startswith("mixed:")


def test_each_speed_window_records_the_counter_delta_over_its_bench(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    ticks = iter(range(0, 10**9, 5000))  # every read is 5 J after the previous one
    services.energy_counter = lambda: next(ticks)

    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)

    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    windows = json.loads((run_dir / "speed" / "windows.json").read_text())["windows"]
    assert windows and all(window["energy_j"] == pytest.approx(5.0) for window in windows)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["energy"]["method"] == "nvml-energy-counter-per-window"
    valid = sum(1 for window in windows if window["valid"])
    assert summary["energy"]["joules"] == pytest.approx(5.0 * valid)
