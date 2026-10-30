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


def test_a_leg_with_failed_steps_still_hands_its_run_to_the_next_legs(tmp_path: Path) -> None:
    # Owner decision 2026-10-29: a failed score or test never stops the batch. A run that went through every
    # step with some failed (RunIncomplete) is a reference the 4-bit legs can still use.
    from sma.config import default_config
    from sma.errors import RunIncomplete
    from sma.queue import pair_legs

    queue = JobQueue(tmp_path / "q")
    repo = "org/tiny-GGUF"
    queue.add(pair_legs({"repo": repo, "quant": "gguf", "gguf": "BF16"}, {"repo": repo, "quant": "gguf", "gguf": "Q4_0"}), kind="pair")
    path, claimed = queue.claim("w", "gpu-host-1")
    calls: list[RunSpec] = []

    def fake_execute(spec: RunSpec, _cfg, _services, on_created=None) -> str:
        run_id = f"run-{spec.gguf}-{spec.thinking}"
        if on_created:
            on_created(run_id)
        calls.append(spec)
        if spec.gguf == "BF16":
            raise RunIncomplete(run_id, ["swebench"])
        return run_id

    with pytest.raises(SmaError, match="these failed: swebench"):
        _run_job(default_config(), lambda _cfg: None, queue, path, claimed, fake_execute, RunSpec)
    # The reference with thinking off, then the quant with thinking off and on, each against the reference.
    assert [(spec.gguf, spec.thinking, spec.reference_run_id) for spec in calls] == [
        ("BF16", "off", None),
        ("Q4_0", "off", "run-BF16-off"),
        ("Q4_0", "on", "run-BF16-off"),
    ]
    failed = json.loads(next((tmp_path / "q" / "failed").glob("*.json")).read_text())
    assert [leg["status"] for leg in failed["legs"]] == ["incomplete", "done", "done"]


def test_pair_commands_parse_their_repo_flags(tmp_path: Path, capsys) -> None:
    # F1 (model-list.txt): every run-pair and queue add-pair failed with "Pass one of --bf16 ..." because
    # the repo names in --bf16 and --nvfp4 were read as true-or-false flags.
    from sma.cli import main

    config = tmp_path / "sma.yaml"
    config.write_text(f"runs_dir: {tmp_path / 'runs'}\nqueue:\n  dir: {tmp_path / 'q'}\n")
    assert main(["queue", "add-pair", "--bf16", "org/ref", "--nvfp4", "org/fp4", "--config", str(config)]) == 0
    assert main(["queue", "add-pair", "--gguf", "org/tiny-GGUF", "BF16", "Q4_0", "--config", str(config)]) == 0
    jobs = JobQueue(tmp_path / "q").list_jobs()
    legs = [[(leg["repo"], leg["gguf"] or leg["quant"], leg["thinking"]) for leg in job["legs"]] for job in jobs]
    assert [("org/ref", "bf16", "off"), ("org/fp4", "nvfp4", "off"), ("org/fp4", "nvfp4", "on")] in legs
    assert [("org/tiny-GGUF", "BF16", "off"), ("org/tiny-GGUF", "Q4_0", "off"), ("org/tiny-GGUF", "Q4_0", "on")] in legs
    assert next(job for job in jobs if job["legs"][0]["quant"] == "gguf")["engine"] == "llamacpp"


def test_a_pending_job_can_be_removed_and_a_started_one_cannot(tmp_path: Path) -> None:
    # Owner, 2026-10-29: two queued LFM3 GGUF pairs were dropped from the plan; the queue had no way to do it.
    queue = JobQueue(tmp_path / "q")
    job = queue.add([{"repo": "org/ref", "quant": "bf16"}], kind="single")
    started = queue.add([{"repo": "org/other", "quant": "bf16"}], kind="single")
    path, claimed = queue.claim("w", "gpu-host-1")
    with pytest.raises(SmaError, match="is claimed. Only a pending job can be removed"):
        queue.remove(claimed["job_id"])
    queue.release_to_pending(path, {**claimed, "legs": [{**claimed["legs"][0], "run_id": "run-1", "status": "failed"}]})
    with pytest.raises(SmaError, match="already has run run-1"):
        queue.remove(job["job_id"] if claimed["job_id"] == job["job_id"] else started["job_id"])
    other = started["job_id"] if claimed["job_id"] == job["job_id"] else job["job_id"]
    queue.remove(other)
    assert [j["job_id"] for j in queue.list_jobs()] == [claimed["job_id"]]
    assert (tmp_path / "q" / "removed" / f"{other}.json").is_file()
    with pytest.raises(SmaError, match="No queue job"):
        queue.remove("job-missing")


def test_reclaim_can_take_a_shorter_age_when_the_worker_is_known_gone(tmp_path: Path) -> None:
    from sma.cli import main

    config = tmp_path / "sma.yaml"
    config.write_text(f"runs_dir: {tmp_path / 'runs'}\nqueue:\n  dir: {tmp_path / 'q'}\n")
    queue = JobQueue(tmp_path / "q")
    queue.add([{"repo": "org/ref", "quant": "bf16"}], kind="single")
    queue.claim("w", "gpu-host-1")
    assert main(["queue", "reclaim", "--config", str(config)]) == 0  # an hour by default: nothing moves
    assert [job["state"] for job in queue.list_jobs()] == ["claimed"]
    assert main(["queue", "reclaim", "--older-than", "0", "--config", str(config)]) == 0
    assert [job["state"] for job in queue.list_jobs()] == ["pending"]
