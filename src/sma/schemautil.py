"""Load the packaged JSON Schemas and validate run records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sma.errors import SmaError


def schema_path(name: str) -> Path:
    packaged = Path(__file__).resolve().parent / "schemas" / name
    if packaged.is_file():
        return packaged
    root = Path(__file__).resolve().parents[2] / "schemas" / name
    if root.is_file():
        return root
    raise SmaError(f"JSON Schema {name} is not available")


def load_schema(name: str) -> dict[str, Any]:
    return json.loads(schema_path(name).read_text(encoding="utf-8"))


def validate_instance(instance: dict[str, Any], name: str) -> None:
    import jsonschema

    schema = load_schema(name)
    try:
        jsonschema.validate(instance, schema)
    except jsonschema.ValidationError as exc:
        path = "/".join(str(part) for part in exc.absolute_path)
        where = f" at {path}" if path else ""
        raise SmaError(f"{name} schema validation failed{where}: {exc.message}") from exc
