"""Versioned corpus of token-id windows. No chat template is applied."""

from __future__ import annotations

from collections.abc import Callable

CORPUS_VERSION = "sma-corpus-v1"

# A small fixed vocabulary. The generator, not English quality, is the contract.
_WORDS = (
    "the", "model", "reads", "a", "fixed", "window", "of", "tokens", "and",
    "scores", "the", "next", "one", "without", "a", "chat", "template",
    "spark", "memory", "holds", "weights", "and", "cache", "together",
    "prefill", "is", "compute", "bound", "decode", "is", "bandwidth", "bound",
    "north", "south", "east", "west", "river", "stone", "cedar", "signal",
    "alpha", "beta", "gamma", "delta", "epsilon", "lambda", "sigma", "omega",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "return", "value", "function", "table", "row", "column", "sum", "mean",
)


def _lcg(seed: int) -> Callable[[], int]:
    state = seed & 0x7FFFFFFF

    def _next() -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state

    return _next


def generate_text(version: str, min_chars: int) -> str:
    if version != CORPUS_VERSION:
        raise ValueError(
            f"Unknown corpus version {version!r}. This build knows {CORPUS_VERSION}."
        )
    # The seed is the version string itself, not Python's salted hash().
    seed = int.from_bytes(version.encode("utf-8"), "little") & 0x7FFFFFFF
    draw = _lcg(seed or 1)
    parts: list[str] = []
    produced = 0
    while produced < min_chars:
        length = 8 + (draw() % 9)
        words = [_WORDS[draw() % len(_WORDS)] for _ in range(length)]
        sentence = " ".join(words).capitalize() + "."
        parts.append(sentence)
        produced += len(sentence) + 1
    return " ".join(parts)


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
) -> tuple[list[list[int]], str]:
    """Grow the versioned text until `num_windows` exact windows exist."""
    chars = max(256, window_tokens * num_windows)
    text = ""
    ids: list[int] = []
    windows: list[list[int]] = []
    for _ in range(12):
        text = generate_text(version, chars)
        ids = list(encode(text))
        windows, _dropped = windows_from_ids(
            ids, window_tokens=window_tokens, num_windows=num_windows
        )
        if len(windows) >= num_windows:
            return windows, text
        chars *= 2
    raise ValueError(
        f"Corpus {version} produced {len(ids)} tokens, "
        f"which is fewer than {num_windows} windows of {window_tokens}."
    )
