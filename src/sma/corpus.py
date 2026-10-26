"""Versioned natural-text corpus. No chat template is applied.

sma-corpus-v3 is a vendored public-domain slice of Pride and Prejudice
(Project Gutenberg ebook 1342). It starts at the novel's first sentence.
The illustrated edition's list of illustrations and page numbers was removed
from the previous slice, and the sha256 changed. Commercial use is allowed
because the text is public domain. The default run scores 48 windows of
2048 tokens (98304) when the tokenizer yields that many.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

CORPUS_VERSION = "sma-corpus-v3"
_DATA = Path(__file__).resolve().parent / "data"


def corpus_paths(version: str) -> tuple[Path, Path]:
    if version != CORPUS_VERSION:
        raise ValueError(
            f"Unknown corpus version {version!r}. This build ships {CORPUS_VERSION}."
        )
    return _DATA / "sma-corpus-v3.txt", _DATA / "sma-corpus-v3.json"


def load_corpus(version: str) -> tuple[str, dict[str, Any]]:
    text_path, meta_path = corpus_paths(version)
    text = text_path.read_text(encoding="utf-8")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != meta.get("sha256"):
        raise ValueError(
            f"Corpus {version} hash is {digest}, metadata says {meta.get('sha256')}. "
            "Refusing to score a corpus whose bytes changed."
        )
    return text, meta


def windows_from_ids(
    token_ids: list[int],
    *,
    window_tokens: int,
    num_windows: int,
) -> tuple[list[list[int]], int]:
    if window_tokens < 2:
        raise ValueError("window_tokens must be >= 2")
    windows: list[list[int]] = []
    cursor = 0
    while len(windows) < num_windows and cursor + window_tokens <= len(token_ids):
        windows.append(list(token_ids[cursor : cursor + window_tokens]))
        cursor += window_tokens
    dropped = max(0, len(token_ids) - cursor)
    return windows, dropped


def build_windows(
    encode: Callable[[str], list[int]],
    *,
    version: str,
    window_tokens: int,
    num_windows: int,
) -> tuple[list[list[int]], str, dict[str, Any]]:
    """Cut the pinned corpus into exact token windows. No chat template."""
    text, meta = load_corpus(version)
    ids = list(encode(text))
    windows, _dropped = windows_from_ids(
        ids, window_tokens=window_tokens, num_windows=num_windows
    )
    if len(windows) < num_windows:
        raise ValueError(
            f"Corpus {version} produced {len(ids)} tokens, "
            f"which is fewer than {num_windows} windows of {window_tokens}."
        )
    return windows, text, meta
