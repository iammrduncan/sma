import json
import os
import threading
from pathlib import Path

import pytest

from sma.atomic import atomic_write_json
from sma.errors import SmaError
from sma.pipeline import RunSpec
from sma.queue import JobQueue, _run_job


def test_atomic_write_keeps_the_previous_file_when_replace_fails(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "manifest.json"
    atomic_write_json(path, {"ok": 1})

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_json(path, {"ok": 2})
    assert json.loads(path.read_text()) == {"ok": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_two_workers_claim_different_jobs(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    queue.add([{"repo": "org/b", "quant": "nvfp4"}], kind="single")
    claimed: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        got = queue.claim(name, "gpu-host-1")
        assert got is not None
        with lock:
            claimed.append(got[1]["job_id"])

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(claimed)) == 2
    assert queue.claim("w3", "gpu-host-1") is None


def test_stale_claim_can_be_reclaimed(tmp_path: Path) -> None:
    queue = JobQueue(tmp_path)
    job = queue.add([{"repo": "org/a", "quant": "bf16"}], kind="single")
    path, claimed = queue.claim("w", "gpu-host-1")
    claimed["heartbeat"] = "2000-01-01T00:00:00Z"
    atomic_write_json(path, claimed)
    moved = queue.reclaim_stale(10)
    assert moved == [job["job_id"]]
    again = queue.claim("w2", "gpu-host-2")
    assert again is not None
    assert again[1]["job_id"] == job["job_id"]


def test_pair_job_runs_both_quants_on_one_worker_and_passes_the_reference(tmp_path: Path) -> None:
    from sma.config import default_config

    queue = JobQueue(tmp_path / "q")
    job = queue.add(
        [
            {"repo": "org/ref", "quant": "bf16"},
            {"repo": "org/fp4", "quant": "nvfp4"},
        ],
        kind="pair",
    )
    path, claimed = queue.claim("w", "gpu-host-1")
    calls: list[RunSpec] = []

    def fake_execute(spec: RunSpec, _cfg, _services, on_created=None) -> str:
        run_id = f"run-{spec.quant_flag}"
        if on_created:
            on_created(run_id)
        calls.append(spec)
        return run_id

    cfg = default_config()
    _run_job(cfg, lambda _cfg: None, queue, path, claimed, fake_execute, RunSpec)
    assert [spec.quant_flag for spec in calls] == ["bf16", "nvfp4"]
    assert calls[0].reference_run_id is None
    assert calls[1].reference_run_id == "run-bf16"
    assert calls[0].lock_held is True
    assert calls[1].pair_job_id == job["job_id"]
    done = list((tmp_path / "q" / "done").glob("*.json"))
    assert len(done) == 1


def test_failed_leg_does_not_start_the_next_quant(tmp_path: Path) -> None:
    from sma.config import default_config

    queue = JobQueue(tmp_path / "q")
    queue.add(
        [
            {"repo": "org/ref", "quant": "bf16"},
            {"repo": "org/fp4", "quant": "nvfp4"},
        ],
        kind="pair",
    )
    path, claimed = queue.claim("w", "gpu-host-1")
    seen: list[str] = []

    def fake_execute(spec: RunSpec, _cfg, _services, on_created=None) -> str:
        seen.append(spec.quant_flag)
        if on_created:
            on_created("run-bf16")
        raise SmaError("quality failed")

    with pytest.raises(SmaError, match="quality failed"):
        _run_job(default_config(), lambda _cfg: None, queue, path, claimed, fake_execute, RunSpec)
    assert seen == ["bf16"]
    failed = json.loads(next((tmp_path / "q" / "failed").glob("*.json")).read_text())
    assert failed["legs"][0]["status"] == "failed"
    assert failed["legs"][1]["status"] == "pending"
