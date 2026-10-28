"""Remove hostnames and secrets before anything is stored in a run record."""

from __future__ import annotations

import os
import re
import socket
from typing import Any

from sma.errors import SmaError

# Match credential fields only. "tokens", "tokenizer", "token_count", and "*_per_token" are data.
_SECRET_KEY = re.compile(
    r"(?:^|_)(?:hf_token|access_token|refresh_token|id_token|auth_token|(?<!per_)token|secret|password|authorization|credential)$"
    r"|api[_-]?key",
    re.IGNORECASE,
)


# Lab machines' names and addresses, from the root .env (.env.example lists them). A record names each
# one by its placeholder, as the baseline READMEs under runs/ do.
LAB_ENV = (
    "SMA_HOST",
    "KERNELBENCH_URL",
    "KERNELBENCH_HOST",
    "KERNELBENCH_GPU_PCI",
    "SWE_RUNNER_HOST",
    "SWE_RUNNER_USER",
    "SWE_RUNNER_SSH_KEY",
    "TAILNET_DOMAIN",
    "SMA_JUDGE_URL",
)


def host_names() -> list[str]:
    found: list[str] = []
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            value = getter()
        except OSError:
            continue
        if not value:
            continue
        lowered = value.lower()
        if lowered in {"localhost", "localhost.localdomain"}:
            continue
        found.append(value)
    # Longest first so an FQDN is replaced before its short name.
    return sorted(set(found), key=len, reverse=True)


def scrub_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    for name in host_names():
        if len(name) < 3:
            continue
        # A period after the name ends a sentence ("/home/name."). Only a period followed by a
        # letter or digit continues a longer domain name, which belongs to another machine.
        pattern = re.compile(
            rf"(?<![A-Za-z0-9._-]){re.escape(name)}(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9])",
            re.IGNORECASE,
        )
        redacted = pattern.sub("<host>", redacted)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "SMA_JUDGE_API_KEY"):
        value = os.environ.get(key)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, "<redacted>")
    # Longest value first, so a URL is replaced before the host inside it. Short values (a user name
    # like "sma") are left alone: they identify nothing and would match ordinary words.
    lab = sorted(((os.environ.get(key) or "", key) for key in LAB_ENV), key=lambda item: len(item[0]), reverse=True)
    for value, key in lab:
        if len(value) >= 4:
            redacted = redacted.replace(value.rstrip("/"), "${" + key + "}")
    return redacted


def assert_text_clean(text: str, what: str) -> str:
    """Fail closed when text contains a hostname or a known token."""
    cleaned = scrub_text(text)
    if cleaned != text:
        raise SmaError(
            f"The {what} contains this machine's hostname or a secret. "
            "SMA does not write those into run records. "
            "Set hf.cache_dir in sma.yaml to a path that does not include the hostname."
        )
    return text


def scrub_obj(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY.search(str(key)):
                cleaned[str(key)] = "<redacted>"
            else:
                cleaned[str(key)] = scrub_obj(item)
        return cleaned
    if isinstance(value, list):
        return [scrub_obj(item) for item in value]
    if isinstance(value, str):
        return scrub_text(value)
    return value
