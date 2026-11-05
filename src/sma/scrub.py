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
    "V100_HOST",
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


# Secrets in .env whose values never reach a record. browsecomp.judge_key_env must name one of them.
TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "SMA_JUDGE_API_KEY", "OPENROUTER_KEY")

# Keys by their shape, wherever they came from. sma publish-check refuses them; scrub_text cannot know them.
_KEY_SHAPES = (
    ("a Hugging Face token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("a GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("an OpenRouter key", re.compile(r"sk-or-v1-[0-9a-f]{32,}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY")),
)


def _host_patterns() -> list[re.Pattern[str]]:
    # A period after the name ends a sentence ("/home/name."). Only a period followed by a
    # letter or digit continues a longer domain name, which belongs to another machine.
    return [
        re.compile(rf"(?<![A-Za-z0-9._-]){re.escape(name)}(?![A-Za-z0-9_-])(?!\.[A-Za-z0-9])", re.IGNORECASE)
        for name in host_names()
        if len(name) >= 3
    ]


def scrub_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pattern in _host_patterns():
        redacted = pattern.sub("<host>", redacted)
    for key in TOKEN_ENV:
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


def publish_deny() -> list[str]:
    """Private names (repos, machines, projects) that are never published: SMA_PUBLISH_DENY in the root .env,
    comma-separated. The list stays in .env, so publishing it cannot leak it."""
    return [term.strip() for term in (os.environ.get("SMA_PUBLISH_DENY") or "").split(",") if len(term.strip()) >= 3]


def find_leaks(text: str) -> list[str]:
    """What text would publish, named and never quoted: this machine's hostname, a token, a lab name from .env,
    a name from SMA_PUBLISH_DENY, or something shaped like a key."""
    found = []
    if any(pattern.search(text) for pattern in _host_patterns()):
        found.append("this machine's hostname")
    for key in TOKEN_ENV + LAB_ENV:
        value = (os.environ.get(key) or "").rstrip("/")
        if len(value) >= 4 and value in text:
            found.append(key)
    lowered = text.lower()
    if any(term.lower() in lowered for term in publish_deny()):
        found.append("a name in SMA_PUBLISH_DENY")
    found += [label for label, shape in _KEY_SHAPES if shape.search(text)]
    return found


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
