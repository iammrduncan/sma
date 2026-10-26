"""SQLite catalog plus an atomic index.json snapshot of every run."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from sma.atomic import atomic_write_json
from sma.errors import SmaError
from sma.scrub import scrub_obj


class _CatalogLock:
    """Blocking lock so two processes can update the catalog without corrupting it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> "_CatalogLock":
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise SmaError(f"Timed out waiting for the catalog lock at {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc: object) -> None:
        import fcntl

        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


class Catalog:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self.db_path = self.runs_dir / "catalog.sqlite"
        self.index_path = self.runs_dir / "index.json"
        self._lock = _CatalogLock(self.runs_dir / ".catalog.lock")

    def upsert(self, manifest: dict[str, Any], run_dir: Path) -> None:
        row = _row(manifest, run_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with _connect(self.db_path) as conn:
                _ensure(conn)
                conn.execute(
                    """
                    INSERT INTO runs (
                      run_id, model_repo, model_slug, quant, revision_sha, status,
                      host_alias, run_dir, created_at, updated_at, reference_run_id, pair_job_id
                    ) VALUES (
                      :run_id, :model_repo, :model_slug, :quant, :revision_sha, :status,
                      :host_alias, :run_dir, :created_at, :updated_at, :reference_run_id, :pair_job_id
                    )
                    ON CONFLICT(run_id) DO UPDATE SET
                      model_repo=excluded.model_repo,
                      model_slug=excluded.model_slug,
                      quant=excluded.quant,
                      revision_sha=excluded.revision_sha,
                      status=excluded.status,
                      host_alias=excluded.host_alias,
                      run_dir=excluded.run_dir,
                      updated_at=excluded.updated_at,
                      reference_run_id=excluded.reference_run_id,
                      pair_job_id=excluded.pair_job_id
                    """,
                    row,
                )
                conn.commit()
                rows = _all(conn)
            atomic_write_json(self.index_path, {"schema_version": 1, "runs": rows})

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return self._index_rows()
        with self._lock:
            with _connect(self.db_path) as conn:
                _ensure(conn)
                return _all(conn)

    def find(self, run_id: str) -> Path | None:
        for row in self.list_runs():
            if row["run_id"] == run_id:
                path = Path(row["run_dir"])
                if (path / "manifest.json").is_file():
                    return path
        scanned = self._scan_manifest(run_id)
        return scanned

    def rebuild(self) -> int:
        rows: list[dict[str, Any]] = []
        if self.runs_dir.is_dir():
            for manifest_path in sorted(self.runs_dir.glob("*/*/*/manifest.json")):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(manifest, dict) or "run_id" not in manifest:
                    continue
                rows.append(_row(manifest, manifest_path.parent))
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with _connect(self.db_path) as conn:
                _ensure(conn)
                conn.execute("DELETE FROM runs")
                for row in rows:
                    conn.execute(
                        """
                        INSERT INTO runs (
                          run_id, model_repo, model_slug, quant, revision_sha, status,
                          host_alias, run_dir, created_at, updated_at, reference_run_id, pair_job_id
                        ) VALUES (
                          :run_id, :model_repo, :model_slug, :quant, :revision_sha, :status,
                          :host_alias, :run_dir, :created_at, :updated_at, :reference_run_id, :pair_job_id
                        )
                        """,
                        row,
                    )
                conn.commit()
                stored = _all(conn)
            atomic_write_json(self.index_path, {"schema_version": 1, "runs": stored})
        return len(stored)

    def _index_rows(self) -> list[dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        runs = payload.get("runs") if isinstance(payload, dict) else None
        if not isinstance(runs, list):
            return []
        return [row for row in runs if isinstance(row, dict)]

    def _scan_manifest(self, run_id: str) -> Path | None:
        if not self.runs_dir.is_dir():
            return None
        for manifest_path in self.runs_dir.glob("*/*/*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(manifest, dict) and manifest.get("run_id") == run_id:
                return manifest_path.parent
        return None


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
          run_id TEXT PRIMARY KEY,
          model_repo TEXT NOT NULL,
          model_slug TEXT NOT NULL,
          quant TEXT NOT NULL,
          revision_sha TEXT,
          status TEXT NOT NULL,
          host_alias TEXT,
          run_dir TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          reference_run_id TEXT,
          pair_job_id TEXT
        )
        """
    )


def _all(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def _row(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
    created = timestamps.get("created") or ""
    updated = timestamps.get("finished") or timestamps.get("started") or created
    return scrub_obj(
        {
            "run_id": manifest["run_id"],
            "model_repo": manifest["model_repo"],
            "model_slug": manifest["model_slug"],
            "quant": manifest["quant_flag"],
            "revision_sha": manifest.get("revision_sha"),
            "status": _status(manifest),
            "host_alias": manifest.get("host_alias"),
            "run_dir": str(run_dir),
            "created_at": created,
            "updated_at": updated or created,
            "reference_run_id": manifest.get("reference_run_id"),
            "pair_job_id": manifest.get("pair_job_id"),
        }
    )


def _status(manifest: dict[str, Any]) -> str:
    timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
    if timestamps.get("finished") and not _failed(manifest):
        return "done"
    if _failed(manifest):
        return "failed"
    if timestamps.get("started"):
        return "running"
    return "pending"


def _failed(manifest: dict[str, Any]) -> bool:
    steps = manifest.get("steps")
    if not isinstance(steps, dict):
        return False
    return any(isinstance(step, dict) and step.get("status") == "failed" for step in steps.values())
