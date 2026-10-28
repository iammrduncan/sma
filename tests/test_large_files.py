"""Run records are committed; the files too big for git are not (owner decision 2026-10-27).

Each manifest keeps the sha256 and size of those files, and SMA refuses a copy that does not match."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.catalog import Catalog
from sma.compare import compare_run_dirs
from sma.errors import CompareRefused
from sma.hashing import sha256_file
from sma.pipeline import RunSpec, execute
from sma.records import LARGE_FILES
from sma.schemautil import validate_instance
from tests.fakes import FixtureFetcher, write_gguf_snapshot
from tests.harness import make_config, make_services

ROOT = Path(__file__).resolve().parents[1]


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_gitignore_keeps_run_records_and_leaves_out_only_the_large_files_and_machine_names() -> None:
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "runs/" not in lines and "runs" not in lines
    assert "runs/.env" in lines
    for rel in LARGE_FILES:
        assert f"runs/**/{rel}" in lines


def test_a_fresh_clone_lists_its_committed_runs_without_the_ignored_catalog(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    runs = tmp_path / "runs"
    for derived in ("catalog.sqlite", "index.json"):
        (runs / derived).unlink()
    assert [row["run_id"] for row in Catalog(runs).list_runs()] == [run_id]


def test_the_logprob_dump_hash_is_recorded_and_a_changed_dump_is_refused(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_dir = _run_dir(tmp_path, execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services))
    manifest = _manifest(run_dir)
    validate_instance(manifest, "manifest.schema.json")
    dump = run_dir / "logprobs" / "dump.json"
    assert manifest["large_files"] == {
        "logprobs/dump.json": {"sha256": f"sha256:{sha256_file(dump)}", "bytes": dump.stat().st_size}
    }

    compare_run_dirs(run_dir, run_dir)
    dump.write_text(dump.read_text(encoding="utf-8").replace('"windows"', '"windows" ', 1), encoding="utf-8")
    with pytest.raises(CompareRefused, match="does not match the sha256 in its manifest"):
        compare_run_dirs(run_dir, run_dir)


def test_a_reference_whose_logits_file_changed_gives_no_kl(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    services.fetcher = FixtureFetcher(write_gguf_snapshot(tmp_path / "gguf"), repo="example/tiny-GGUF", quant_flag="gguf")
    reference = execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="BF16"), cfg, services)
    ref_dir = _run_dir(tmp_path, reference)
    logits = ref_dir / "logprobs" / "kld-base.bin"
    assert _manifest(ref_dir)["large_files"] == {
        "logprobs/kld-base.bin": {"sha256": f"sha256:{sha256_file(logits)}", "bytes": logits.stat().st_size}
    }

    logits.write_bytes(b"other logits")
    candidate = execute(
        RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="UD-Q4_K_XL", reference_run_id=reference), cfg, services
    )
    run_dir = _run_dir(tmp_path, candidate)
    assert any("does not match the sha256 in its manifest" in warning for warning in _manifest(run_dir)["warnings"])
    assert not (run_dir / "logprobs" / "kl_vs_reference.json").exists()
