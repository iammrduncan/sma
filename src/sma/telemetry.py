"""1 Hz GPU samples and throttle detection from the CSV log."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from sma.atomic import atomic_write_text
from sma.commands import nvidia_smi_query, nvidia_smi_query_legacy
from sma.config import Config
from sma.logsetup import log_event
from sma.proc import run_command

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
        self._thread = threading.Thread(target=self._loop, name="sma-telemetry", daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.telemetry.interval_s + 5)
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


def write_note(path: Path, payload: dict[str, Any]) -> None:
    from sma.atomic import atomic_write_json

    atomic_write_json(path, payload)


def ensure_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        atomic_write_text(path, _HEADER)
