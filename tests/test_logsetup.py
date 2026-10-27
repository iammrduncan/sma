import json
import logging

from sma.logsetup import _ConsoleFormatter, _JsonFormatter


def _record(message: str, event: str, fields: dict[str, str]) -> logging.LogRecord:
    record = logging.LogRecord("sma", logging.INFO, __file__, 1, message, None, None)
    record.event = event
    record.fields = fields
    return record


def test_the_terminal_gets_readable_lines_and_sma_log_keeps_json() -> None:
    record = _record(
        "step_failed step=determinism run_id=r1 error=boom",
        "step_failed",
        {"step": "determinism", "run_id": "r1", "error": "boom"},
    )

    line = _ConsoleFormatter().format(record)
    assert line.endswith("✗ determinism failed: boom")
    assert "{" not in line

    stored = json.loads(_JsonFormatter().format(record))
    assert stored["event"] == "step_failed"
    assert stored["message"] == "step_failed step=determinism run_id=r1 error=boom"


def test_an_event_without_a_template_still_prints_all_its_fields() -> None:
    record = _record("clock_cap applied=false mhz=2400", "clock_cap", {"applied": "false", "mhz": "2400"})
    assert _ConsoleFormatter().format(record).endswith("clock_cap applied=false mhz=2400")


def test_a_retry_after_the_first_attempt_says_which_attempt() -> None:
    record = _record("step_start step=speed attempt=2", "step_start", {"step": "speed", "attempt": "2"})
    assert _ConsoleFormatter().format(record).endswith("▶ speed (attempt 2)")
