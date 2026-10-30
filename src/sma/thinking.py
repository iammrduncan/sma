"""The model's own thinking mode, from its chat template.

The quality server turns thinking off for every request (quality.chat_template_kwargs), because BFCL
cannot send it per request. KernelBench and SWE-bench run the model as shipped (owner decision
2026-10-27), so each of their requests sends the value that reproduces the template's own default.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sma.errors import SmaError


def chat_template(snapshot: Path, model_file: str | None = None) -> str | None:
    """The checkpoint's chat template: a GGUF's metadata, chat_template.jinja, or tokenizer_config.json."""
    if model_file:
        from sma.gguf import read_header_file

        value = read_header_file(snapshot / model_file).metadata.get("tokenizer.chat_template")
        return value if isinstance(value, str) else None
    jinja = snapshot / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    config = snapshot / "tokenizer_config.json"
    if not config.is_file():
        return None
    value = json.loads(config.read_text(encoding="utf-8")).get("chat_template")
    if isinstance(value, list):
        named = {item.get("name"): item.get("template") for item in value if isinstance(item, dict)}
        value = named.get("default")
    return value if isinstance(value, str) else None


def thinking_default(template: str | None) -> dict[str, Any]:
    """{"enable_thinking": True, False, or None, "how": why}. None means send nothing: the template
    ignores enable_thinking, or SMA could not tell its default, and the server's setting applies."""
    if not template:
        return {"enable_thinking": None, "how": "the checkpoint has no chat template"}
    try:
        base, on, off = (_render(template, extra) for extra in ({}, {"enable_thinking": True}, {"enable_thinking": False}))
    except Exception as exc:  # A template may call anything; SMA records that it could not tell.
        return {"enable_thinking": None, "how": f"the chat template did not render here ({type(exc).__name__})"}
    if on == off:
        return {"enable_thinking": None, "how": "the chat template ignores enable_thinking"}
    if base == on:
        return {"enable_thinking": True, "how": "the chat template thinks unless told not to"}
    if base == off:
        return {"enable_thinking": False, "how": "the chat template thinks only when asked"}
    return {"enable_thinking": None, "how": "the chat template's default matches neither setting"}


def _render(template: str, extra: dict[str, Any]) -> str:
    from jinja2 import nodes
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(message: str) -> None:
        raise SmaError(message)

    class Generation(Extension):
        """{% generation %}...{% endgeneration %}, which transformers' templates use to mark the assistant's
        tokens (LFM3's does). It renders its body; SMA needs no mask. Without it LFM3's template did not
        render, and its agent suites ran with the server's thinking setting (off) instead of its default (on)."""

        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

        def _body(self, caller):
            return caller()

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", Generation])
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    messages = [{"role": "user", "content": "Hello."}]
    return env.from_string(template).render(messages=messages, add_generation_prompt=True, bos_token="", eos_token="", **extra)
