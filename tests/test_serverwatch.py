import json
import socket
from pathlib import Path

import pytest

from sma.errors import SmaError
from sma.httpapi import wait_until_healthy
from sma.pipeline import RunSpec, execute
from sma.proc import follow_to_file
from sma.serverwatch import needed_kv_tokens
from tests.harness import free_port, make_config, make_services


def test_a_followed_log_interleaves_stdout_and_stderr_and_scrubs_them(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda: "secret-host")
    path = tmp_path / "logs" / "server-quality.log"

    following = follow_to_file(
        ["sh", "-c", "echo one; echo two >&2; echo three on secret-host"], path, what="test follow"
    )
    following.stop(timeout_s=5)

    assert path.read_text(encoding="utf-8").splitlines() == ["one", "two", "three on <host>"]
    assert not following.thread.is_alive()


def test_the_health_wait_stops_as_soon_as_the_server_exits() -> None:
    checks = iter([True, True, False])
    waited: list[float] = []

    with pytest.raises(SmaError, match="exited before it became healthy"):
        wait_until_healthy(
            f"http://127.0.0.1:{free_port()}",
            deadline_s=60,
            request_timeout_s=0.2,
            poll_s=0.01,
            alive=lambda: next(checks),
            on_wait=waited.append,
            report_every_s=0,
        )
    assert len(waited) == 2


def test_a_server_that_stops_mid_step_is_named_in_the_step_error(tmp_path: Path, monkeypatch) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)

    def host_stops_the_server(**_kwargs: object) -> None:
        # What earlyoom did on a Spark: the server shut down cleanly while requests were in flight.
        runtime.kill(next(iter(runtime.running)))
        raise SmaError("POST /v1/completions failed after 3 attempts: Connection reset by peer")

    monkeypatch.setattr("sma.pipeline.run_determinism", host_stops_the_server)
    with pytest.raises(SmaError):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)

    run_dir = next((tmp_path / "runs").glob("*/*/*/*/"))
    error = json.loads((run_dir / "manifest.json").read_text())["steps"]["determinism"]["error"]
    assert "Connection reset by peer" in error
    assert "The quality server stopped while this step was running (exit code 0)" in error
    assert "journalctl -u earlyoom" in error
    # The server log was followed live, and no dump replaced it after the server became ready.
    assert any(name.endswith("-quality") for name in runtime.followed)
    assert (run_dir / "logs" / "server-quality.log").read_text(encoding="utf-8").startswith("(followed live)")


def test_a_step_that_fails_while_the_server_runs_gets_no_server_note(tmp_path: Path, monkeypatch) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def bad_answer(**_kwargs: object) -> None:
        raise SmaError("the server answered with something unexpected")

    monkeypatch.setattr("sma.pipeline.run_determinism", bad_answer)
    with pytest.raises(SmaError):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)

    manifest = json.loads(next((tmp_path / "runs").glob("*/*/*/*/manifest.json")).read_text())
    assert manifest["steps"]["determinism"]["error"] == "the server answered with something unexpected"


def _kv_warnings(tmp_path: Path, capacity: int) -> list[str]:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    runtime.logs = lambda _name: f"INFO [kv_cache_utils.py:1731] GPU KV cache size: {capacity:,} tokens\n"
    execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    manifest = json.loads(next((tmp_path / "runs").glob("*/*/*/*/manifest.json")).read_text())
    return [warning for warning in manifest["warnings"] if "KV cache holds" in warning]


def test_a_kv_cache_one_token_short_of_the_widest_load_is_a_run_warning(tmp_path: Path) -> None:
    needed = needed_kv_tokens(make_config(tmp_path))
    warnings = _kv_warnings(tmp_path, needed - 1)
    assert len(warnings) == 1
    assert f"holds {needed - 1:,} tokens. SMA's widest load can need {needed:,}" in warnings[0]


def test_a_kv_cache_that_holds_the_widest_load_adds_no_warning(tmp_path: Path) -> None:
    assert _kv_warnings(tmp_path, needed_kv_tokens(make_config(tmp_path))) == []
