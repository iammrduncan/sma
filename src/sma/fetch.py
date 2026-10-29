"""Download a pinned Hugging Face snapshot and describe the checkpoint.

The download itself uses huggingface_hub (resumable). Tests call
`inspect_snapshot` on a local directory and never touch the network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sma.config import Config, hf_token
from sma.errors import QuantMismatch, SmaError
from sma.hashing import sha256_bytes, sha256_file
from sma.logsetup import log_event
from sma.memplan import KvLayout, kv_layout_from_hf_config
from sma.params import active_params, billions, count_shards, size_class
from sma.quant import DetectedQuant, assert_flag_matches, detect_quant
from sma.retry import call_with_retry
from sma.scrub import scrub_text

_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
_SKIP_DIRS = {".git", ".cache"}


@dataclass
class FetchResult:
    repo: str
    revision_requested: str | None
    revision_sha: str
    snapshot_path: str
    quant_flag: str
    quant_label: str
    detected: DetectedQuant
    file_hashes: dict[str, str]
    file_sizes: dict[str, int]
    params_total: int | None
    params_active: int | None
    params_total_b: float | None
    params_active_b: float | None
    params_method: str
    size_class: str | None
    tokenizer_fingerprint: str | None
    config: dict[str, Any]
    warnings: list[str]
    # Bytes of the weight files the engine loads. The memory plan counts these.
    weights_bytes: int = 0
    # The KV layout, from config.json or the GGUF metadata. None when it could not be read.
    layout: KvLayout | None = None
    # GGUF only: the files, the one the engine loads, and what their own metadata says.
    gguf: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        detected = self.detected
        return {
            "repo": self.repo,
            "revision_requested": self.revision_requested,
            "revision_sha": self.revision_sha,
            "snapshot_path": self.snapshot_path,
            "quant_flag": self.quant_flag,
            "quant_label": self.quant_label,
            "quant_detected": detected.format,
            "quant_detail": detected.detail,
            "quant_producer": detected.producer,
            "architecture": detected.model_type,
            "architectures": detected.architectures,
            "is_moe": detected.is_moe,
            "num_experts": detected.num_experts,
            "experts_per_tok": detected.experts_per_tok,
            "batch_invariant_supported": detected.batch_invariant_supported,
            "batch_invariant_reason": detected.batch_invariant_reason,
            "dtype": detected.dtype,
            "file_hashes": self.file_hashes,
            "file_sizes": self.file_sizes,
            "params_total": self.params_total,
            "params_active": self.params_active,
            "params_total_b": self.params_total_b,
            "params_active_b": self.params_active_b,
            "params_method": self.params_method,
            "size_class": self.size_class,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "warnings": self.warnings,
            "sources": detected.sources,
            "weights_bytes": self.weights_bytes,
            "gguf": self.gguf,
        }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SmaError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SmaError(f"{path.name} must be a JSON object")
    return loaded


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return sorted(files)


def inspect_snapshot(
    snapshot: Path,
    *,
    repo: str,
    revision_sha: str,
    quant_flag: str,
    revision_requested: str | None = None,
) -> FetchResult:
    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise SmaError(f"Snapshot directory does not exist: {snapshot}")
    config = _read_json(snapshot / "config.json")
    if config is None:
        raise SmaError(
            f"{repo} has no config.json. SMA cannot verify the quant flag without it."
        )
    hf_quant = _read_json(snapshot / "hf_quant_config.json")
    try:
        detected = detect_quant(config, hf_quant)
    except QuantMismatch:
        raise
    quant_label = assert_flag_matches(quant_flag, detected)

    warnings: list[str] = []
    if not detected.batch_invariant_supported and detected.batch_invariant_reason:
        warnings.append(detected.batch_invariant_reason)

    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    weight_files: list[Path] = []
    for path in _iter_files(snapshot):
        rel = path.relative_to(snapshot).as_posix()
        sizes[rel] = path.stat().st_size
        if path.suffix.lower() in _WEIGHT_SUFFIXES or path.name in {
            "config.json",
            "hf_quant_config.json",
        }:
            hashes[rel] = "sha256:" + sha256_file(path)
            if path.suffix.lower() in _WEIGHT_SUFFIXES:
                weight_files.append(path)
    if not any(path.suffix.lower() == ".safetensors" for path in weight_files):
        warnings.append(
            "No safetensors weights were found. Parameter counts are left empty. "
            "Pickled .bin files are hashed but never loaded."
        )

    shards = [path for path in weight_files if path.suffix.lower() == ".safetensors"]
    if shards:
        total, expert, other = count_shards(shards)
        active, method = active_params(
            total=total,
            expert=expert,
            other=other,
            num_experts=detected.num_experts,
            experts_per_tok=detected.experts_per_tok,
            is_moe=detected.is_moe,
        )
        total_b = billions(total)
        active_b = billions(active) if active is not None else None
        klass = size_class(total)
        if active is None and detected.is_moe:
            warnings.append(method)
    else:
        total = active = None
        total_b = active_b = None
        klass = None
        method = "unavailable: no safetensors"

    fingerprint = None
    tokenizer = snapshot / "tokenizer.json"
    if tokenizer.is_file():
        fingerprint = "sha256:" + sha256_bytes(tokenizer.read_bytes())
    else:
        warnings.append(
            "tokenizer.json is missing. The logprob pass needs a fast tokenizer file "
            "and will fail until one is present. Chat templates are not used for logprobs."
        )

    return FetchResult(
        repo=repo,
        revision_requested=revision_requested,
        revision_sha=revision_sha,
        snapshot_path=str(snapshot),
        quant_flag=quant_flag,
        quant_label=quant_label,
        detected=detected,
        file_hashes=hashes,
        file_sizes=sizes,
        params_total=total,
        params_active=active,
        params_total_b=total_b,
        params_active_b=active_b,
        params_method=method,
        size_class=klass,
        tokenizer_fingerprint=fingerprint,
        config=config,
        warnings=warnings,
        weights_bytes=sum(path.stat().st_size for path in shards),
        layout=kv_layout_from_hf_config(config),
    )


def inspect_gguf(
    snapshot: Path,
    *,
    repo: str,
    revision_sha: str,
    selector: str,
    revision_requested: str | None = None,
) -> FetchResult:
    """Describe one GGUF quant from its files' own metadata. SMA never converts or relabels it."""
    from sma.gguf import match_gguf_files, read_header_file, summarize

    snapshot = Path(snapshot)
    if not snapshot.is_dir():
        raise SmaError(f"Snapshot directory does not exist: {snapshot}")
    listing = [path.relative_to(snapshot).as_posix() for path in _iter_files(snapshot)]
    files = match_gguf_files(listing, selector)
    headers = [read_header_file(snapshot / name) for name in files]
    info = summarize(headers[0], *headers[1:])
    sizes = {name: (snapshot / name).stat().st_size for name in files}
    hashes = {name: "sha256:" + sha256_file(snapshot / name) for name in files}
    composition = {"tensor_types": dict(info.tensor_types), "tensor_bytes": dict(info.tensor_bytes)}
    detected = DetectedQuant(
        format="gguf",
        detail=f"GGUF {info.file_type or 'with no file type'}; tensors "
        + ", ".join(f"{count} {name}" for name, count in sorted(info.tensor_types.items())),
        dtype=None,
        model_type=info.architecture,
        architectures=[info.architecture],
        is_moe=info.is_moe,
        num_experts=info.expert_count,
        experts_per_tok=info.expert_used_count,
        batch_invariant_supported=False,
        batch_invariant_reason="llama.cpp has no batch-invariant mode. The determinism probe still runs and records what it sees.",
        producer=None,
        sources={"gguf_file_type": info.file_type, "quant_method": "gguf", "kv_cache_quant_algo": None},
        composition=composition,
    )
    total = info.params_total
    expert = sum(tensor.n_elements for header in headers for tensor in header.tensors if "_exps" in tensor.name)
    if total is not None:
        active, method = active_params(
            total=total,
            expert=expert,
            other=total - expert,
            num_experts=info.expert_count,
            experts_per_tok=info.expert_used_count,
            is_moe=info.is_moe,
        )
    else:
        active, method = None, "unavailable: not every split was read"
    warnings = list(info.notes)
    return FetchResult(
        repo=repo,
        revision_requested=revision_requested,
        revision_sha=revision_sha,
        snapshot_path=str(snapshot),
        quant_flag="gguf",
        quant_label=selector,
        detected=detected,
        file_hashes=hashes,
        file_sizes=sizes,
        params_total=total,
        params_active=active,
        params_total_b=billions(total) if total is not None else None,
        params_active_b=billions(active) if active is not None else None,
        params_method=f"GGUF tensor shapes; {method}",
        size_class=size_class(total) if total is not None else None,
        tokenizer_fingerprint=info.tokenizer_fingerprint,
        config={},
        warnings=warnings,
        weights_bytes=sum(sizes.values()),
        layout=info.layout,
        gguf={
            "files": files,
            "model_file": files[0],
            "file_type": info.file_type,
            "file_type_id": info.file_type_id,
            "architecture": info.architecture,
            "name": info.name,
            "split_count": info.split_count,
            "native_context": info.native_context,
            "notes": list(info.notes),
        },
    )


def read_remote_gguf_header(repo: str, sha: str, filename: str, token: str | None) -> Any:
    """A GGUF header read with ranged requests, before the weights are downloaded."""
    from huggingface_hub import HfFileSystem

    from sma.gguf import read_header

    fs = HfFileSystem(token=token)
    with fs.open(f"{repo}@{sha}/{filename}", "rb", block_size=8 * 1024 * 1024) as handle:
        return read_header(handle.read)


# A tokenizer's files, for a bench that needs one apart from the checkpoint (sma.pipeline._bench_tokenizer).
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"]


def snapshot_dir(cfg: Config, repo: str, sha: str) -> Path:
    """Where one commit of a repo is downloaded: <hf.cache_dir>/<model slug>/<sha>."""
    from sma.records import model_slug

    return cfg.cache_path() / model_slug(repo) / sha


def download_snapshot(
    repo: str, revision: str | None, cfg: Config, *, allow_patterns: list[str] | None = None
) -> tuple[Path, str]:
    """Resolve a revision to a commit SHA and download that snapshot (only `allow_patterns`, when set)."""
    token = hf_token(cfg)

    def _once() -> tuple[Path, str]:
        # Local import so `sma --help` works even if the hub package is broken.
        from huggingface_hub import HfApi, snapshot_download

        previous = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(cfg.hf.download_timeout_s)
        try:
            api = HfApi(token=token)
            info = api.model_info(repo_id=repo, revision=revision)
            sha = info.sha
            if not sha:
                raise SmaError(f"Hugging Face did not return a commit SHA for {repo}")
            # A plain folder of real files per commit. The engine container mounts only this
            # folder at /model. A cache snapshot is links into blobs outside it, which break there.
            path = snapshot_download(
                repo_id=repo,
                revision=sha,
                local_dir=str(snapshot_dir(cfg, repo, sha)),
                token=token,
                allow_patterns=allow_patterns,
            )
        finally:
            if previous is None:
                os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
            else:
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = previous
        log_event("fetched", repo=repo, revision_sha=sha)
        return Path(path), sha

    try:
        return call_with_retry(
            _once,
            attempts=cfg.retries.attempts,
            backoff_s=cfg.retries.backoff_s,
            factor=cfg.retries.backoff_factor,
            what=f"download {repo}",
        )
    except SmaError:
        raise
    except Exception as exc:
        raise SmaError(f"download {repo} failed: {scrub_text(str(exc))}") from exc
