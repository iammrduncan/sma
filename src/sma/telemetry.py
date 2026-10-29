"""1 Hz GPU samples and throttle detection from the CSV log, and the engine's /metrics counters."""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from sma.atomic import atomic_write_text
from sma.commands import nvidia_smi_query, nvidia_smi_query_legacy
from sma.config import Config
from sma.errors import SmaError
from sma.httpapi import scrape_metrics
from sma.logsetup import log_event
from sma.proc import run_command
from sma.prom import metric_name, parse_prometheus

# One JSON row per engine scrape: {"ts", "iso", "phase", "values": {name: number}}.
ENGINE_METRICS_FILE = "engine_metrics.jsonl"

_HEADER = (
    "ts_unix,iso,sm_clock_mhz,temp_c,power_w,"
    "hw_thermal,sw_thermal,hw_power,sw_power,throttle,flags_unverified,phase\n"
)


def _flag(text: str) -> int | None:
    folded = text.strip().lower().strip("[]")
    if folded in {"", "n/a", "na", "unknown", "none"}:
        return None
    if folded in {"1", "active", "yes", "true"}:
        return 1
    if folded in {"0", "not active", "inactive", "no", "false"}:
        return 0
    return None


def _number(text: str) -> str:
    cleaned = text.strip().split()[0] if text.strip() else ""
    try:
        return str(float(cleaned))
    except ValueError:
        return ""


@dataclass
class Sample:
    ts_unix: float
    temp_c: float | None
    power_w: float | None
    throttle: bool
    phase: str
    hw_thermal: int | None = 0
    sw_thermal: int | None = 0
    hw_power: int | None = 0
    sw_power: int | None = 0
    flags_unverified: bool = False


def parse_nvidia_line(line: str, *, now: float, phase: str) -> Sample | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 7:
        return None
    temp = power = None
    try:
        temp = float(parts[1])
    except ValueError:
        temp = None
    try:
        power = float(parts[2])
    except ValueError:
        power = None
    flags = [_flag(part) for part in parts[3:7]]
    while len(flags) < 4:
        flags.append(None)
    unverified = any(flag is None for flag in flags)
    known = [flag for flag in flags if flag is not None]
    throttled = any(flag == 1 for flag in known)
    return Sample(
        ts_unix=now,
        temp_c=temp,
        power_w=power,
        throttle=throttled,
        phase=phase,
        hw_thermal=flags[0],
        sw_thermal=flags[1],
        hw_power=flags[2],
        sw_power=flags[3],
        flags_unverified=unverified,
    )


def _flag_cell(flag: int | None) -> str:
    if flag is None:
        return "na"
    return str(flag)


def format_sample_row(parsed: Sample, raw_clock: str, flags: list[int | None]) -> str:
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(parsed.ts_unix))
    temp = "" if parsed.temp_c is None else f"{parsed.temp_c:.3f}"
    power = "" if parsed.power_w is None else f"{parsed.power_w:.3f}"
    padded = list(flags) + [None] * (4 - len(flags))
    cells = [
        f"{parsed.ts_unix:.6f}",
        iso,
        raw_clock,
        temp,
        power,
        _flag_cell(padded[0]),
        _flag_cell(padded[1]),
        _flag_cell(padded[2]),
        _flag_cell(padded[3]),
        "1" if parsed.throttle else "0",
        "1" if parsed.flags_unverified else "0",
        parsed.phase,
    ]
    return ",".join(cells) + "\n"


def read_samples(path: Path) -> list[Sample]:
    if not path.is_file():
        return []
    rows: list[Sample] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                ts = float(row["ts_unix"])
            except (KeyError, TypeError, ValueError):
                continue
            temp = _maybe_float(row.get("temp_c"))
            power = _maybe_float(row.get("power_w"))
            rows.append(
                Sample(
                    ts_unix=ts,
                    temp_c=temp,
                    power_w=power,
                    throttle=row.get("throttle") == "1",
                    phase=row.get("phase") or "",
                    hw_thermal=_stored_flag(row.get("hw_thermal")),
                    sw_thermal=_stored_flag(row.get("sw_thermal")),
                    hw_power=_stored_flag(row.get("hw_power")),
                    sw_power=_stored_flag(row.get("sw_power")),
                    flags_unverified=row.get("flags_unverified") == "1",
                )
            )
    return rows


def _stored_flag(value: str | None) -> int | None:
    if value is None or value == "" or value == "na":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def throttled_between(path: Path, start: float, end: float) -> list[Sample]:
    return [
        sample
        for sample in read_samples(path)
        if sample.throttle and start <= sample.ts_unix <= end
    ]


def integrate_joules(samples: list[Sample], *, max_gap_s: float = 2.5) -> float | None:
    """Trapezoid of GPU power inside one continuous window.

    Gaps longer than max_gap_s are not bridged. Callers that have several
    windows must integrate each window and add the results.
    """
    points = [(sample.ts_unix, sample.power_w) for sample in samples if sample.power_w is not None]
    points.sort()
    if len(points) < 2:
        return None
    joules = 0.0
    used = False
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > max_gap_s or p0 is None or p1 is None:
            continue
        joules += (p0 + p1) / 2.0 * dt
        used = True
    return joules if used else None


def coverage_problem(
    samples: list[Sample],
    *,
    start: float,
    end: float,
    interval_s: float,
    min_ratio: float,
    max_gap_s: float,
) -> str | None:
    """Unverified unless the window has about duration/interval samples and no large gap.

    The recorder's failure counter is not an input. A later successful sample
    resets that counter, so it cannot prove the window was covered.
    """
    duration = max(0.0, end - start)
    if interval_s <= 0:
        return "telemetry interval is not positive"
    expected = duration / interval_s
    needed = expected * min_ratio
    if len(samples) < needed and not (expected < 1.0 and len(samples) >= 1 and duration <= max_gap_s):
        return (
            f"telemetry covered {len(samples)} sample(s); "
            f"this window needs about {expected:.2f} ({min_ratio:g} of duration/interval)"
        )
    times = sorted(sample.ts_unix for sample in samples)
    gaps: list[float] = []
    if not times:
        gaps.append(duration)
    else:
        gaps.append(max(0.0, times[0] - start))
        gaps.extend(times[index + 1] - times[index] for index in range(len(times) - 1))
        gaps.append(max(0.0, end - times[-1]))
    largest = max(gaps) if gaps else duration
    if largest > max_gap_s:
        return f"telemetry gap of {largest:.3f}s is larger than max_sample_gap_s ({max_gap_s:g})"
    return None


_MEMORY_HEADER = "ts_unix,mem_available_bytes,mem_total_bytes,phase\n"


def meminfo() -> tuple[int, int] | None:
    """(MemAvailable, MemTotal) in bytes from /proc/meminfo. On a GB10 this memory is the GPU's too."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if name in {"MemAvailable", "MemTotal"} and parts and parts[0].isdigit():
            values[name] = int(parts[0]) * 1024
    if "MemAvailable" not in values or "MemTotal" not in values:
        return None
    return values["MemAvailable"], values["MemTotal"]


def read_memory(path: Path) -> list[tuple[float, int, int]]:
    """(ts_unix, available, total) rows from memory.csv. Empty when it was not recorded."""
    if not path.is_file():
        return []
    rows: list[tuple[float, int, int]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((float(row["ts_unix"]), int(row["mem_available_bytes"]), int(row["mem_total_bytes"])))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def peak_memory_bytes(rows: list[tuple[float, int, int]]) -> int | None:
    """How far available memory fell below the run's first sample, at its lowest. Other processes on the
    host move it too, so it is the run's footprint only while the run is the only large job."""
    if len(rows) < 2:
        return None
    return max(0, rows[0][1] - min(row[1] for row in rows))


def invalidating_throttle(sample: Sample, *, sw_power_cap_invalidates: bool) -> bool:
    if sample.hw_thermal == 1 or sample.sw_thermal == 1 or sample.hw_power == 1:
        return True
    return bool(sw_power_cap_invalidates and sample.sw_power == 1)


class TelemetryRecorder:
    """Writes telemetry.csv. A scripted feeder can append rows in tests."""

    def __init__(self, path: Path, cfg: Config) -> None:
        self.path = path
        self.cfg = cfg
        self.phase = "startup"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: IO[str] | None = None
        self.start_temp: float | None = None
        self.end_temp: float | None = None
        self._query = nvidia_smi_query(cfg.doctor.gpu_index)
        self.failures = 0
        self._sampling = False
        # (base URL, metrics path, names) of the engine whose counters are sampled, once known.
        self._engine: tuple[str, str, tuple[str, ...]] | None = None
        self._engine_thread: threading.Thread | None = None

    def start(self, sampler: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", encoding="utf-8")
        if new_file:
            self._handle.write(_HEADER)
            self._handle.flush()
        self._stop.clear()
        if not sampler:
            return
        self._sampling = True
        self._thread = threading.Thread(target=self._loop, name="sma-telemetry", daemon=True)
        self._thread.start()

    def watch_engine(self, base_url: str, path: str, names: tuple[str, ...]) -> None:
        """Sample the engine's counters into engine_metrics.jsonl every telemetry.engine_metrics_interval_s
        while a server answers. Its own thread, so a slow scrape never delays the 1 Hz GPU samples."""
        self._engine = (base_url, path, names)
        if self._sampling and self._engine_thread is None:
            self._engine_thread = threading.Thread(target=self._engine_loop, name="sma-engine-metrics", daemon=True)
            self._engine_thread.start()

    def sample_engine(self, *, now: float | None = None, text: str | None = None) -> bool:
        """Append one engine_metrics.jsonl row from a scrape, or from text a test passes. False, and no
        row, when no server answers or it reports none of the names."""
        if self._engine is None:
            return False
        base_url, path, names = self._engine
        if text is None:
            try:
                text = scrape_metrics(base_url, self.cfg.timeouts.metrics_s, path=path)
            except SmaError:
                return False
        totals = engine_values(text, names)
        if not totals:
            return False
        stamp = time.time() if now is None else now
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stamp))
        row = {"ts": round(stamp, 3), "iso": iso, "phase": self.phase, "values": totals}
        with self.path.with_name(ENGINE_METRICS_FILE).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return True

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.telemetry.interval_s + 5)
        if self._engine_thread is not None:
            self._engine_thread.join(timeout=self.cfg.timeouts.metrics_s + 5)
            self._engine_thread = None
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
        samples = read_samples(self.path)
        temps = [sample.temp_c for sample in samples if sample.temp_c is not None]
        if temps:
            if self.start_temp is None:
                self.start_temp = temps[0]
            self.end_temp = temps[-1]

    def append_parsed(self, line: str, *, now: float | None = None, phase: str | None = None) -> None:
        stamp = time.time() if now is None else now
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            return
        flags = [_flag(part) for part in parts[3:7]]
        sample = parse_nvidia_line(line, now=stamp, phase=phase or self.phase)
        if sample is None:
            return
        if self.start_temp is None and sample.temp_c is not None:
            self.start_temp = sample.temp_c
        if sample.temp_c is not None:
            self.end_temp = sample.temp_c
        row = format_sample_row(sample, _number(parts[0]), flags)
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                if self.path.stat().st_size == 0:
                    handle.write(_HEADER)
                handle.write(row)
            return
        self._handle.write(row)
        self._handle.flush()

    def append_memory(self, available: int, total: int, *, now: float | None = None) -> None:
        """One memory.csv row next to telemetry.csv. A test can feed it directly."""
        path = self.path.with_name("memory.csv")
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8") as handle:
            if new_file:
                handle.write(_MEMORY_HEADER)
            stamp = time.time() if now is None else now
            handle.write(f"{stamp:.6f},{available},{total},{self.phase}\n")

    def latest_temp(self) -> float | None:
        samples = read_samples(self.path)
        for sample in reversed(samples):
            if sample.temp_c is not None:
                return sample.temp_c
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.time()
            try:
                result = run_command(self._query, timeout_s=10, what="nvidia-smi")
                if result.returncode != 0:
                    legacy = run_command(
                        nvidia_smi_query_legacy(self.cfg.doctor.gpu_index),
                        timeout_s=10,
                        what="nvidia-smi legacy",
                    )
                    result = legacy if legacy.returncode == 0 else result
                memory = meminfo()
                if memory is not None:
                    self.append_memory(*memory)
                if result.returncode == 0 and result.stdout.strip():
                    self.append_parsed(result.stdout.strip().splitlines()[0])
                    self.failures = 0
                else:
                    self.failures += 1
                    if self.failures == 1 or self.failures % 30 == 0:
                        log_event("telemetry_error", failures=self.failures, stderr=result.stderr[-200:])
            except Exception as exc:
                self.failures += 1
                log_event("telemetry_error", error=str(exc))
            elapsed = time.time() - started
            wait = max(0.0, self.cfg.telemetry.interval_s - elapsed)
            self._stop.wait(wait)


    def _engine_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_engine()
            except Exception as exc:  # A sampler error must never stop the run.
                log_event("engine_metrics_error", error=str(exc)[-200:])
            self._stop.wait(self.cfg.telemetry.engine_metrics_interval_s)


def engine_values(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """The named series from a /metrics scrape, labels stripped and summed. Names it lacks are left out."""
    totals: dict[str, float] = {}
    for key, value in parse_prometheus(text).items():
        name = metric_name(key)
        if name in names:
            totals[name] = totals.get(name, 0.0) + value
    return totals


def write_note(path: Path, payload: dict[str, Any]) -> None:
    from sma.atomic import atomic_write_json

    atomic_write_json(path, payload)


def ensure_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        atomic_write_text(path, _HEADER)
