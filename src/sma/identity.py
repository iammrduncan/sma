"""Host aliases. Real hostnames are never written into a run record."""

from __future__ import annotations

import re

from sma.errors import SmaError
from sma.scrub import host_names

_ALIAS = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


def assert_alias_safe(alias: str) -> None:
    if alias and "." in alias:
        raise SmaError(
            "host_alias contains a dot, so it looks like a hostname. "
            "Use an alias such as gpu-host-1."
        )
    if not _ALIAS.fullmatch(alias or ""):
        raise SmaError(
            "host_alias must look like gpu-host-1: lowercase letters, digits, and hyphens, "
            "starting with a letter or digit, at most 41 characters."
        )
    for name in host_names():
        if alias.lower() == name.lower():
            raise SmaError(
                "host_alias is this machine's hostname. "
                "Pick an alias such as gpu-host-1. The hostname is not stored."
            )
