from sma.cleanup import drop_page_cache, idle_until_cool
from sma.config import default_config
from sma.doctor import default_checks, failed
from sma.telemetry import integrate_joules, throttled_between
from sma.telemetry import Sample


def test_drop_page_cache_does_not_raise() -> None:
    result = drop_page_cache()
    assert result["attempted"] is True
    assert "ok" in result


def test_idle_until_cool_stops_when_the_sensor_drops() -> None:
    temps = iter([70.0, 60.0, 40.0])

    def read() -> float:
        return next(temps)

    result = idle_until_cool(read, target_c=55, timeout_s=5, poll_s=0)
    assert result["reached"] is True
    assert result["end_c"] == 40.0


def test_joules_are_the_trapezoid_of_power() -> None:
    samples = [
        Sample(0.0, 40.0, 100.0, False, "speed"),
        Sample(10.0, 41.0, 100.0, False, "speed"),
    ]
    assert integrate_joules(samples, max_gap_s=10.0) == 1000.0
    # A 10s hole is a gap at the 1 Hz sample rate, not a window to bridge.
    assert integrate_joules(samples, max_gap_s=2.5) is None


def test_throttle_window(tmp_path) -> None:
    from sma.telemetry import TelemetryRecorder

    cfg = default_config()
    cfg.telemetry.interval_s = 1
    recorder = TelemetryRecorder(tmp_path / "telemetry.csv", cfg)
    recorder.start(sampler=False)
    recorder.append_parsed("1000, 40, 80, Not Active, Not Active, Not Active, Not Active", now=100.0)
    recorder.append_parsed("900, 96, 120, Active, Not Active, Not Active, Not Active", now=150.0)
    recorder.append_parsed("1000, 50, 80, Not Active, Not Active, Not Active, Not Active", now=200.0)
    hits = throttled_between(recorder.path, 140, 160)
    assert len(hits) == 1
    assert throttled_between(recorder.path, 0, 10) == []


def test_doctor_reports_failures_without_raising() -> None:
    checks = default_checks(default_config())
    names = {check.name for check in checks}
    assert {"docker", "gpu", "disk", "hf_token", "host_alias"} <= names
    # This VM has no Spark. Doctor must still return a report.
    assert failed(checks) is True or any(check.status == "fail" for check in checks)
