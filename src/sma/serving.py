"""Serving rates averaged over recent windows, from the engine counters each run samples into
engine_metrics.jsonl (telemetry.engine_metrics_interval_s).

Counters only grow while one server runs. A drop means the server restarted, and the new value counts
from zero. A pair of samples further apart than max_gap_s spans a stop, so it is skipped. Rates come
from each engine's backend: nothing is computed for a counter the engine does not report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sma.backends import get_backend
from sma.telemetry import ENGINE_METRICS_FILE

WINDOWS = (("5 min", 300), ("1 hour", 3600), ("24 hours", 86400))


@dataclass
class Series:
    """One run's engine samples, oldest first: (unix time, {name: value})."""

    engine: str
    samples: list[tuple[float, dict[str, float]]]


@dataclass
class Window:
    """Counter changes and gauge means over one window, for one engine."""

    engine: str
    change: dict[str, float] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    def rates(self) -> dict[str, float | None]:
        mean = {name: self.totals[name] / self.counts[name] for name in self.totals if self.counts.get(name)}
        return get_backend(self.engine).serving_rates(self.change, mean, self.seconds)


def recent_series(runs_dir: Path, *, now: float, span_s: float) -> list[Series]:
    """Every run's samples newer than now - span_s. Files not written in that span are not read."""
    found = []
    for path in sorted(Path(runs_dir).glob(f"*/*/*/*/{ENGINE_METRICS_FILE}")):
        try:
            if path.stat().st_mtime < now - span_s:
                continue
            manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        engine = (manifest.get("engine") or {}).get("name") if isinstance(manifest, dict) else None
        samples = [sample for sample in _read_samples(path) if sample[0] >= now - span_s]
        if engine and samples:
            found.append(Series(str(engine), samples))
    return found


def window(series: list[Series], *, start: float, max_gap_s: float) -> list[Window]:
    """One Window per engine, from the sample pairs that end after start."""
    by_engine: dict[str, Window] = {}
    for one in series:
        for (before_t, before), (after_t, after) in zip(one.samples, one.samples[1:]):
            if after_t <= start or after_t - before_t > max_gap_s:
                continue
            totals = by_engine.setdefault(one.engine, Window(one.engine))
            totals.seconds += after_t - before_t
            for name, value in after.items():
                previous = before.get(name)
                if previous is not None:
                    totals.change[name] = totals.change.get(name, 0.0) + (value - previous if value >= previous else value)
                totals.totals[name] = totals.totals.get(name, 0.0) + value
                totals.counts[name] = totals.counts.get(name, 0) + 1
    return list(by_engine.values())


def _read_samples(path: Path) -> list[tuple[float, dict[str, float]]]:
    samples = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row: Any = json.loads(line)
            stamp, values = float(row["ts"]), row["values"]
        except (ValueError, KeyError, TypeError):
            continue  # A row cut short by a stop mid-write.
        if isinstance(values, dict):
            samples.append((stamp, {str(k): float(v) for k, v in values.items() if isinstance(v, (int, float))}))
    samples.sort(key=lambda sample: sample[0])
    return samples
