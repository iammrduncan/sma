from pathlib import Path

import pytest

from sma.fetch import download_snapshot
from tests.harness import make_config


def _fake_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    token: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    allow_patterns: list[str] | None = None,
) -> str:
    """Behave like huggingface_hub: a cache snapshot is links into blobs, a local_dir is files."""
    del token, allow_patterns
    if local_dir is not None:
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(root)
    repo = Path(cache_dir or "") / ("models--" + repo_id.replace("/", "--"))
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    (repo / "blobs" / "b1").write_text("{}", encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").symlink_to("../../blobs/b1")
    return str(snapshot)


def test_the_snapshot_is_a_folder_of_files_the_container_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Info:
        sha = "abc123"

    class _Api:
        def __init__(self, token: str | None = None) -> None:
            del token

        def model_info(self, repo_id: str, revision: str | None = None) -> _Info:
            del repo_id, revision
            return _Info()

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)
    cfg = make_config(tmp_path)

    path, sha = download_snapshot("org/model", None, cfg)

    # Only this folder is mounted at /model. A link that leaves it is broken in the container.
    assert sha == "abc123"
    assert path.name == "abc123"
    assert path.parent.parent == cfg.cache_path()
    files = [item for item in path.rglob("*") if ".cache" not in item.relative_to(path).parts]
    assert files
    assert not any(item.is_symlink() for item in files)
