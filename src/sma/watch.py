"""`sma watch`: a live view of this machine's work. Health first (problems and warnings on top), then each slot
or running run on one line with the selected one's list beneath, the queue, KernelBench's queue, serving rates,
and finished runs.

This module is the model: it reads run folders, the queue, the dispatcher's state, /proc/meminfo, the SMA
containers that are up, and the services' status endpoints, and builds a Board of painted lines. sma.tui lays
the Board out on a terminal and follows keys; `--once` prints it as text. It changes nothing. The screen is
scrubbed like a run record (no hostname, lab machine, or key), so it can be shared.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from sma.backends import get_backend
from sma.catalog import Catalog
from sma.config import Config
from sma.dispatcher import state_path, unit_alive
from sma.errors import SmaError
from sma.locks import lock_held
from sma.machines import Machine
from sma.queue import JobQueue
from sma.records import STEP_NAMES
from sma.scrub import scrub_text
from sma.serving import WINDOWS, Series, combined_peaks, combined_rates, recent_series, window
from sma.telemetry import ENGINE_METRICS_FILE, engine_values
from sma.tui import Board, Item, Panel, pad, plain, plain_text

# Steps that send work elsewhere and report their own progress in the run folder.
_AGENT_SUITES = {"kernelbench": "kernelbench_l1", "swebench": "swebench_verified_mini", "browsecomp_plus": "browsecomp_plus"}
_MARK = {"done": "✓", "running": "▶", "failed": "✗", "pending": "·", "queued": "·", "stopped": "◼"}
_COLOR = {
    "done": "32", "running": "33", "failed": "31", "pending": "2", "queued": "2", "stopped": "33", "head": "1;36",
    "warn": "33",
    # SERVING cells: the window's average in blue, its peak in purple (owner, 2026-10-30).
    "avg": "38;5;39", "max": "38;5;135",
}  # fmt: skip
# Finished runs `--once` prints; the TUI scrolls through all of them.
_FINISHED_SHOWN = 3
# A run's progress on its slot line is cut to this many characters; its window below has it whole.
_PROGRESS_WIDTH = 50
# The machines are asked again this often: whether each answers can change, what it is does not.
_MACHINES_EVERY_S = 60.0
# The KernelBench service's queue is asked again this often.
_KERNELBENCH_EVERY_S = 15.0
# A running run that writes none of these for this long is flagged: its process may be gone. Telemetry and
# memory are sampled every second and engine counters every 15 s while a run works.
_QUIET_S = 600.0
_WRITES = ("manifest.json", ENGINE_METRICS_FILE, "telemetry.csv", "memory.csv", "logs/sma.log")
# A memory-guard stop stays an alert this long.
_GUARD_ALERT_S = 3600.0
# The dispatcher's systemd unit, where services/kernelbench/host-setup.sh installs it.
_DISPATCHER_UNIT = Path("/etc/systemd/system/sma-dispatcher.service")
# Free memory under this many GiB is a problem (red); under _MEMORY_LOW_GIB, yellow.
_MEMORY_CRITICAL_GIB, _MEMORY_LOW_GIB = 16, 32
_MACHINE_NAMES = {"model": "this machine", "kernel": "the KernelBench service", "swe": "the x86_64 runner", "judge": "the judge"}


@dataclass
class Live:
    """sma watch's own reads of the current run's server. The latest read gives the since-start row. For
    a run that records no engine_metrics.jsonl (one started before SMA recorded them), the reads also
    stand in for its samples while sma watch is open."""

    run_id: str | None = None
    engine: str | None = None
    latest: dict[str, float] | None = None
    # Seconds between reads (sma watch --interval), so a pair of reads is not taken for a server stop.
    interval_s: float = 0.0
    # When the server container started: its counters run from then, so all-stream rates since start
    # divide by this uptime. None when docker does not say.
    started: datetime | None = None
    samples: list[tuple[float, dict[str, float]]] = field(default_factory=list)


@dataclass
class View:
    """What one screen shows. Gathered from the outside world, then rendered with no further reads of it."""

    containers: list[str]
    memory: tuple[int, int] | None
    now: datetime
    color: bool = False
    live: Live | None = None
    # Terminal columns. Longer lines are clipped so nothing wraps. None: no clipping.
    width: int | None = None
    # The machines SMA uses (sma.machines.gather). None: only this one, from what the view has.
    machines: list[Machine] | None = None
    interval_s: float | None = None
    # SWE-bench progress read from the runner for the current run (read_swe), or None.
    swe: dict[str, Any] | None = None
    # The KernelBench service's latest jobs (GET /v1/jobs), anyone's; None when it is not set or does not answer.
    kernelbench: list[dict[str, Any]] | None = None


def render(cfg: Config, view: View, *, plan: Path | None = None, scoped: bool = True) -> str:
    """One screen of text: the TUI's main screen with every panel in full (`sma watch --once`)."""
    return plain(board(cfg, view, plan=plan, scoped=scoped), view.width)


def board(cfg: Config, view: View, *, plan: Path | None = None, open_item: tuple[str, str] | None = None,
          scoped: bool = True) -> Board:  # fmt: skip
    """Everything one screen shows, read now: the top line, the alerts, each panel, and the open item's detail
    (open_item: ("run", run id) or ("job", job id)). Scoped (`sma watch --config`): what this config serves, the
    serving machine's slots or this Spark. Not scoped (plain `sma watch`): every serving machine's slots in the queue,
    this Spark's runs beside them, and every finished run."""
    paint = _painter(view.color)
    now = view.now
    runs = _runs(cfg)
    by_id = {row["run_id"]: row for row in runs}
    # FINISHED: the runs served where this config serves (the serving machine's GPUs, or this Spark's), or every run.
    served = cfg.remote.hardware if cfg.remote.enabled else None
    here = [row for row in runs if _served_on(row) == served] if scoped else runs
    # A run's own config names its suites; a total added to sma.yaml since then still counts for it.
    totals = {suite.name: suite.questions for suite in cfg.quality.suites if suite.questions}
    usual = _usual_durations(runs)
    jobs = [job for job in JobQueue(cfg.queue_path()).list_jobs() if isinstance(job.get("legs"), list)]
    # In the order they were queued, which is the order they start (a job's id begins with its creation time).
    queued = sorted((job for job in jobs if job.get("state") in ("pending", "claimed")), key=lambda job: str(job["job_id"]))
    series = recent_series(cfg.runs_path(), now=now.timestamp(), span_s=WINDOWS[0][1])
    states = _dispatcher_states(cfg, scoped)
    if not states and scoped and cfg.remote.slots:
        # Slots and no dispatcher state yet: the slots, idle, and a dispatcher that is not running.
        states = [{"slots": [{"name": slot.name, "gpus": slot.gpus, "state": "idle"} for slot in cfg.remote.slots], "alive": False}]
    # This Spark's runs at work: a config's own when scoped, else beside the slots. Only runs served on the Spark: a
    # slot's run has a container here only while a grader runs here (BFCL's), so it would come and go.
    pool = here if scoped else [row for row in runs if _served_on(row) is None]
    spark = [] if scoped and states else _running_runs(pool, view.containers, now, lock_held(cfg.lock_file()))
    alerts = _machine_alerts(view) + _queue_alerts(cfg, queued, now)
    current: dict[str, Any] | None = None
    # (label, run) for each run at work: the alerts and the serving rows name them by it.
    active: list[tuple[str, dict[str, Any]]] = []
    state = None
    if states:
        state = states[0] if len(states) == 1 else {"slots": [slot for one in states for slot in one["slots"]],
                                                    "alive": all(one.get("alive") for one in states)}  # fmt: skip
        # A dispatcher runs one run per slot of the serving machine: each slot's run, compact.
        active_panel = _slots_panel(cfg, state, by_id, now, paint, totals, usual, series, spark)
        for slot in state["slots"]:
            row = by_id.get(str(slot.get("run_id") or "")) if slot.get("state") == "busy" else None
            if row is not None:
                active.append((f"{slot.get('name')} {_run_name(row)}", row))
        active += [(f"spark {_run_name(row)}", row) for row in spark]
        for one in states:
            alerts += _dispatcher_alerts(cfg, one, by_id, queued, now)
        busy, total, down = _slot_counts(state["slots"])
        activity = f"{busy} of {total} slots busy" + (f", {down} down" if down else "")
        activity += f", {_count(len(spark), 'run')} on this Spark" if spark else ""
    else:
        running = spark
        active = [(_run_name(row), row) for row in running]
        if len(running) > 1:
            active_panel = _running_panel(cfg, running, now, paint, totals, usual, series)
        else:
            current = running[0] if running else None
            swe = view.swe if current and view.swe and view.swe.get("run_id") == current["run_id"] else None
            active_panel = _current_panel(current, now, paint, totals, usual, swe)
        activity = f"{len(running)} run{'' if len(running) == 1 else 's'}" if running else "no run"
        # Scoped to this Spark while legs run on a serving machine's slots (the same queue): one line says so.
        slotted = sum(1 for job in queued for leg in job["legs"] if leg.get("status") == "running" and leg.get("slot"))
        if slotted and scoped and not cfg.remote.enabled:
            active_panel.notes.append(f"  {_count(slotted, 'run')} on the serving machine's GPU slots; plain sma watch shows every run")
            activity += f" on this Spark, {slotted} on the serving machine"
    for label, row in active:
        alerts += [(level, f"{label}: {text}") for level, text in _run_problems(row, now, usual)]
    # What runs, what is next, then the rest.
    panels = [_machines_panel(view, paint), active_panel, _queue_panel(runs, current, plan, paint, queued, jobs)]
    if view.kernelbench is not None:
        panels.append(_kernelbench_panel(view.kernelbench, by_id, now, paint))
    if not active:
        lines = [paint("SERVING", "head") + "  nothing is serving now"]
    elif len(active) > 1 or state is not None:
        # Short labels: the running panel above names each run in full.
        slot_of = {str(slot.get("run_id")): slot.get("name") for slot in (state or {}).get("slots") or [] if slot.get("state") == "busy"}
        rows = [(f"{slot_of.get(row['run_id']) or row['model_repo'].split('/')[-1]} {row['quant']}"
                 + (" thinking" if _thinking(row) else ""), row["run_id"]) for _name, row in active]  # fmt: skip
        lines = _serving_rows(cfg, now, paint, rows)
    else:
        lines = _serving_lines(cfg, view, paint, current["run_id"] if current else None)
    # On a short screen the table gives way to one line, never a heading with no numbers.
    short = paint("SERVING", "head") + paint(f"  the table needs {len(lines)} rows; a run's detail (Enter) has its own", "pending")
    panels.append(Panel("serving", lines[0], notes=lines[1:], compact=short if len(lines) > 1 else None))
    active_ids = {row["run_id"] for _name, row in active}
    where = None if not scoped else f"served on {cfg.remote.hardware}" if cfg.remote.enabled else "on this Spark"
    panels.append(_finished_panel([row for row in here if row["run_id"] not in active_ids], current, paint, where))
    detail = None
    if open_item is not None:
        detail = _detail(cfg, view, open_item, by_id, jobs, state, active_ids, usual, totals, paint)
    summary = _summary(view, paint, alerts, activity, queued)
    painted = [paint(("✗ " if level == "failed" else "! ") + text, level) for level, text in alerts]
    return _scrubbed(Board(summary, painted, panels, detail))


def _scrubbed(screen: Board) -> Board:
    """Every line scrubbed like a run record. Painted parts were scrubbed before their color codes."""
    for panel in screen.panels:
        panel.title = scrub_text(panel.title)
        panel.notes = [scrub_text(line) for line in panel.notes]
        panel.compact = scrub_text(panel.compact) if panel.compact else None
        for item in panel.items:
            item.lines = [scrub_text(line) for line in item.lines]
    screen.summary = scrub_text(screen.summary)
    screen.alerts = [scrub_text(line) for line in screen.alerts]
    screen.detail = [scrub_text(line) for line in screen.detail] if screen.detail is not None else None
    return screen


class Sources:
    """The outside world for one screen, through the CLI's readers. The machines and SWE-bench progress are asked
    at most once a minute and the KernelBench service every 15 s; the rest on every read."""

    def __init__(
        self,
        cfg: Config,
        *,
        interval_s: float,
        containers: Callable[[], list[str]],
        memory: Callable[[], tuple[int, int] | None],
        scrape: Callable[[str, str], str],
        machines: Callable[[], list[Machine]] | None = None,
        started_at: Callable[[str], datetime | None] | None = None,
        swe_progress: Callable[[str], dict[str, Any] | None] | None = None,
        kernelbench: Callable[[], list[dict[str, Any]] | None] | None = None,
    ) -> None:
        self.cfg, self.containers, self.memory, self.scrape = cfg, containers, memory, scrape
        self.machines, self.started_at, self.swe_progress, self.kernelbench = machines, started_at, swe_progress, kernelbench
        self.live = Live(interval_s=interval_s)
        self.known: list[Machine] | None = None
        self.swe: dict[str, Any] | None = None
        self.kb_jobs: list[dict[str, Any]] | None = None
        self.asked = self.swe_asked = self.kb_asked = -1e9

    def read(self, *, color: bool, width: int | None, interval_s: float | None) -> View:
        up, now = self.containers(), datetime.now(timezone.utc)
        read_live(self.cfg, self.live, up, now, self.scrape, self.started_at)
        if self.machines is not None and time.monotonic() - self.asked >= _MACHINES_EVERY_S:
            self.known, self.asked = self.machines(), time.monotonic()
        # SWE-bench's progress lives on the runner: read it at most once a minute, while the step runs.
        if self.swe_progress is not None and self.live.run_id and time.monotonic() - self.swe_asked >= _MACHINES_EVERY_S:
            self.swe_asked = time.monotonic()
            self.swe = self.swe_progress(self.live.run_id) if _swe_running(self.cfg, self.live.run_id) else None
            if self.swe is not None:
                self.swe["run_id"] = self.live.run_id
        if self.kernelbench is not None and time.monotonic() - self.kb_asked >= _KERNELBENCH_EVERY_S:
            self.kb_jobs, self.kb_asked = self.kernelbench(), time.monotonic()
        return View(up, self.memory(), now, color, self.live, width, self.known, interval_s, self.swe, self.kb_jobs)


def watch(
    cfg: Config,
    *,
    plan: Path | None,
    interval_s: float,
    once: bool,
    containers: Callable[[], list[str]],
    memory: Callable[[], tuple[int, int] | None],
    scrape: Callable[[str, str], str],
    machines: Callable[[], list[Machine]] | None = None,
    started_at: Callable[[str], datetime | None] | None = None,
    swe_progress: Callable[[str], dict[str, Any] | None] | None = None,
    kernelbench: Callable[[], list[dict[str, Any]] | None] | None = None,
    out: TextIO | None = None,
    scoped: bool = True,
) -> None:
    """Print the screen once (--once); on a terminal, run the TUI until q or Ctrl-C; otherwise (piped, or curses
    cannot start) print a screen every interval_s seconds until Ctrl-C. scrape(base_url, path) returns the
    engine's /metrics text, or "" when it does not answer."""
    out = out or sys.stdout
    sources = Sources(cfg, interval_s=interval_s, containers=containers, memory=memory, scrape=scrape, machines=machines,
                      started_at=started_at, swe_progress=swe_progress, kernelbench=kernelbench)  # fmt: skip
    if not once and out.isatty() and sys.stdin.isatty():
        from sma.tui import run

        def read(open_item: tuple[str, str] | None) -> Board:
            return board(cfg, sources.read(color=True, width=None, interval_s=None), plan=plan, open_item=open_item, scoped=scoped)

        if run(read, interval_s):
            return
    color = out.isatty() and not os.environ.get("NO_COLOR")
    while True:
        width = shutil.get_terminal_size().columns if out.isatty() else None
        text = render(cfg, sources.read(color=color, width=width, interval_s=None if once else interval_s), plan=plan, scoped=scoped)
        if once:
            out.write(text + "\n")
            return
        # A terminal is redrawn in place; a pipe gets one screen after another.
        out.write(("\033[H\033[2J" if out.isatty() else "\n") + text + "\n")
        out.flush()
        time.sleep(interval_s)


def _swe_running(cfg: Config, run_id: str) -> bool:
    row = next((row for row in _runs(cfg) if row["run_id"] == run_id), None)
    return bool(row) and ((row["manifest"].get("steps") or {}).get("swebench") or {}).get("status") == "running"


def read_live(
    cfg: Config,
    live: Live,
    containers: list[str],
    now: datetime,
    scrape: Callable[[str, str], str],
    started_at: Callable[[str], datetime | None] | None = None,
) -> None:
    """Read the current run's server counters into live. Nothing when no run has a container up."""
    current = _current_run(_runs(cfg), containers)
    mine = [name for name in containers if current is not None and name.startswith(f"sma-{current['run_id']}-")]
    if current is None or not mine:
        live.latest = None
        return
    if current["run_id"] != live.run_id:
        live.run_id, live.engine, live.latest, live.samples = current["run_id"], current["engine"], None, []
    # The serving container (quality or speed), not a short-lived one such as the logprobs run.
    server = next((name for name in mine if name.endswith(("-quality", "-speed"))), None)
    live.started = started_at(server) if server and started_at else None
    backend = get_backend(str(live.engine))
    values = engine_values(scrape(f"http://{cfg.engine.bind_host}:{cfg.engine.port}", backend.metrics_path()), backend.serving_metrics())
    live.latest = values or None
    if values and not (Path(current["run_dir"]) / ENGINE_METRICS_FILE).is_file():
        live.samples.append((now.timestamp(), values))
        live.samples = [sample for sample in live.samples if sample[0] >= now.timestamp() - WINDOWS[-1][1]]


def _painter(color: bool) -> Callable[[str, str], str]:
    def paint(text: str, style: str) -> str:
        # Scrubbed before the color codes, which would hide a name from scrub_text's word match.
        code = _score_color(float(style[6:])) if style.startswith("score:") else _COLOR.get(style)
        return f"\033[{code}m{scrub_text(text)}\033[0m" if color and code else text

    return paint


# 256-color steps from red through orange and yellow to green, one per tenth of a score.
_GRADIENT = (196, 202, 208, 214, 220, 226, 190, 154, 118, 46)


def _score_color(value: float) -> str:
    return f"38;5;{_GRADIENT[min(max(int(value * 10), 0), 9)]}"


def _percent(value: float, paint: Callable[[str, str], str]) -> str:
    """A score from 0 to 1 as a percentage, colored red (0) through green (1)."""
    return paint(f"{value * 100:.1f}%", f"score:{value}")


def _identity(text: str, _style: str) -> str:
    return text


# The top line and the alerts.


def _summary(view: View, paint, alerts: list[tuple[str, str]], activity: str, queued: list[dict[str, Any]]) -> str:
    """Time, health, what runs, the queue, KernelBench's queue, free memory."""
    title = f"{paint('SMA watch', 'head')}  {view.now.astimezone().strftime('%Y-%m-%d %H:%M:%S')}  "
    problems = sum(1 for level, _text in alerts if level == "failed")
    warnings = len(alerts) - problems
    if not alerts:
        health = paint("✓ healthy", "done")
    else:
        counts = [_count(problems, "problem") if problems else "", _count(warnings, "warning") if warnings else ""]
        health = paint(("✗ " if problems else "! ") + ", ".join(part for part in counts if part), "failed" if problems else "warn")
    parts = [health, activity, f"queue {_count(len(queued), 'job')}"]
    if view.kernelbench is not None:
        timing = sum(1 for job in view.kernelbench if job.get("status") == "running")
        waiting = sum(1 for job in view.kernelbench if job.get("status") == "queued")
        parts.append(f"KernelBench {timing} timing" + (f", {waiting} queued" if waiting else ""))
    if view.memory is not None:
        free = view.memory[0] / 2**30
        parts.append(paint(f"{free:.1f} GiB free", _memory_style(free)))
    line = title + " · ".join(parts)
    if view.interval_s:
        line += paint(f"  ·  redraws every {view.interval_s:g} s, Ctrl-C quits", "pending")
    return line


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}{'' if number == 1 else 's'}"


def _memory_style(free_gib: float) -> str:
    return "failed" if free_gib < _MEMORY_CRITICAL_GIB else "running" if free_gib < _MEMORY_LOW_GIB else "done"


def _machine_alerts(view: View) -> list[tuple[str, str]]:
    alerts = [("failed", f"{_MACHINE_NAMES.get(m.role, m.role)} does not answer") for m in view.machines or [] if m.up is False]
    if view.memory is not None and view.memory[0] / 2**30 < _MEMORY_CRITICAL_GIB:
        alerts.append(("failed", f"memory is low on this machine: {view.memory[0] / 2**30:.1f} GiB free"))
    return alerts


def _queue_alerts(cfg: Config, queued: list[dict[str, Any]], now: datetime) -> list[tuple[str, str]]:
    """A claim whose heartbeat stopped (its worker may be gone), and legs that failed for good."""
    alerts = []
    stale_s = max(300.0, 3 * max(cfg.queue.heartbeat_s, cfg.queue.poll_s))
    for job in queued:
        repo = (job["legs"][0] if job["legs"] else {}).get("repo")
        beat = _time(job.get("heartbeat"))
        if job.get("state") == "claimed" and beat is not None and (now - beat).total_seconds() > stale_s:
            alerts.append(("warn", f"{repo}: the queue claim's heartbeat is {_elapsed(beat, None, now)} old; is its worker still running? (sma queue reclaim)"))
        for leg in job["legs"]:
            if leg.get("dead_lettered"):
                error = f": {_one_line(leg['error'])[:100]}" if leg.get("error") else ""
                alerts.append(("failed", f"{repo} {_leg_name(leg)} failed for good{error}"))
    return alerts


def _dispatcher_alerts(cfg: Config, state: dict[str, Any], by_id: dict[str, dict], queued: list[dict[str, Any]], now: datetime) -> list[tuple[str, str]]:
    """A dead or silent dispatcher, down slots, and the memory guard's last stop."""
    alerts = []
    busy = any(slot.get("state") == "busy" for slot in state["slots"])
    if not state.get("alive") and (busy or queued):
        start = "sudo systemctl start sma-dispatcher.service" if _DISPATCHER_UNIT.is_file() else "sma worker --slots --config <yaml>"
        alerts.append(("failed", f"the dispatcher is not running: {_count(len(queued), 'queued job')} wait{'s' if len(queued) == 1 else ''}, and the memory guard is off. Start it: {start}"))
    updated = _time(state.get("updated"))
    silent_s = max(120.0, 3 * cfg.queue.poll_s)
    if state.get("alive") and updated is not None and (now - updated).total_seconds() > silent_s:
        alerts.append(("warn", f"the dispatcher has not written its state for {_elapsed(updated, None, now)} (it writes every {cfg.queue.poll_s:g} s)"))
    for slot in state["slots"]:
        if slot.get("state") == "down":
            alerts.append(("failed", f"slot {slot.get('name')} is down: {slot.get('reason') or 'no reason given'}"))
    guard = state.get("memory_guard") or {}
    at = _time(guard.get("at"))
    if at is not None and (now - at).total_seconds() < _GUARD_ALERT_S:
        row = by_id.get(str(guard.get("run_id") or ""))
        alerts.append(("warn", f"the memory guard stopped {_run_name(row) if row else guard.get('run_id')} on {guard.get('slot')} "
                               f"{_elapsed(at, None, now)} ago ({guard.get('source')}); it resumes later"))  # fmt: skip
    return alerts


def _run_problems(row: dict[str, Any], now: datetime, usual: dict[str | None, dict[str, float]]) -> list[tuple[str, str]]:
    """A running run's failed steps, steps far past their usual time, and silence in its folder."""
    problems = []
    quiet = _quiet_for(row, now)
    for name, step in _ordered_steps(row["manifest"]):
        if step.get("status") == "failed":
            problems.append(("failed", f"{name} failed: {_one_line(step.get('error') or 'no error recorded')[:120]}"))
        # A silent run's step is late because nothing runs it; the silence is the one warning.
        late = None if quiet else _overdue(name, step, usual.get(_served_on(row)) or {}, now)
        if late:
            problems.append(("warn", f"{name} {late}"))
    if quiet:
        problems.append(("warn", f"nothing written to its run folder for {quiet}; is its process still running?"))
    return problems


def _usual_durations(runs: list[dict[str, Any]]) -> dict[str | None, dict[str, float]]:
    """The median seconds each step took on its first attempt, over this folder's runs, for steps seen 3 times: per
    hardware the runs were served on (None: a Spark), so a V100 step is not late by a Spark's times."""
    took: dict[str | None, dict[str, list[float]]] = {}
    for row in runs:
        mine = took.setdefault(_served_on(row), {})
        for name, step in _ordered_steps(row["manifest"]):
            start, end = _time(step.get("started")), _time(step.get("finished"))
            if step.get("status") == "done" and int(step.get("attempts") or 0) <= 1 and start and end:
                mine.setdefault(name, []).append((end - start).total_seconds())
    return {hardware: {name: statistics.median(values) for name, values in steps.items() if len(values) >= 3}
            for hardware, steps in took.items()}  # fmt: skip


def _overdue(name: str, step: dict[str, Any], usual: dict[str, float], now: datetime) -> str | None:
    """"has run 2h10m, usually 40m" for a first attempt far past its usual time: twice it, and 15 minutes over."""
    started = _time(step.get("started"))
    if step.get("status") != "running" or int(step.get("attempts") or 0) > 1 or started is None or name not in usual:
        return None
    seconds, typical = (now - started).total_seconds(), usual[name]
    if seconds <= max(2 * typical, typical + 900):
        return None
    return f"has run {_elapsed(started, None, now)}, usually {_elapsed(now - timedelta(seconds=typical), None, now)}"


def _quiet_for(row: dict[str, Any], now: datetime) -> str | None:
    """How long the run's folder has had no write, when that is over _QUIET_S."""
    times = [moment for moment in (_mtime(Path(row["run_dir"]) / name) for name in _WRITES) if moment]
    latest = max(times, default=None)
    if latest is None or (now - latest).total_seconds() <= _QUIET_S:
        return None
    return _elapsed(latest, None, now)


def _ordered_steps(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    steps = {name: step for name, step in (manifest.get("steps") or {}).items() if isinstance(step, dict)}
    return sorted(steps.items(), key=lambda item: STEP_NAMES.index(item[0]) if item[0] in STEP_NAMES else len(STEP_NAMES))


def _one_line(text: Any) -> str:
    return re.sub(r"\s*\n\s*", " ", str(text)).strip()


def _run_name(row: dict[str, Any]) -> str:
    return f"{row['model_repo']} {row['quant']}{_thinking(row)}"


def _served_on(row: dict[str, Any]) -> str | None:
    """The serving machine's hardware a run was served on (v100), or None for a Spark's own GPU. The engine folder
    says it too (llamacpp-v100 beside llamacpp), for a run that stopped before its record named its hardware."""
    hardware = ((row.get("manifest") or {}).get("serving") or {}).get("hardware")
    if hardware:
        return str(hardware)
    folder, engine = Path(row["run_dir"]).parent.parent.name, str(row.get("engine") or "")
    return folder[len(engine) + 1 :] if engine and folder.startswith(engine + "-") else None


def _leg_name(leg: dict[str, Any]) -> str:
    return f"{leg.get('gguf') or leg.get('quant')}" + (" thinking" if leg.get("thinking") == "on" else "")


# The panels.


def _machines_panel(view: View, paint) -> Panel:
    if view.memory is None:
        memory = free_only = "memory unknown"
    else:
        free, total = view.memory[0] / 2**30, view.memory[1] / 2**30
        style = _memory_style(free)
        memory = f"memory {paint(f'{free:.1f}', style)} of {total:.1f} GiB free"
        free_only = f"{paint(f'{free:.1f}', style)} free"
    lines, marks = [], []
    for machine in view.machines or [Machine("model", True)]:
        if machine.up is None:
            mark, text = paint("○", "pending"), paint("not set in .env", "pending")
        elif machine.up is False:
            mark, text = paint("●", "failed"), paint("does not answer", "failed")
        else:
            mark, text = paint("●", "done"), machine.summary
        if machine.role == "model" and view.memory is None:
            text = text or memory
        elif machine.role == "model":
            # The summary ends with the memory total; add what is free to it.
            text = f"{text}, {free_only}" if text.endswith(" GiB") else f"{text} · {memory}" if text else memory
        lines.append(f"  {mark} {machine.role:7} {text}")
        marks.append(f"{mark} {machine.role}")
    title = paint("MACHINES", "head")
    return Panel("machines", title, notes=lines, compact=f"{title}  " + "  ".join(marks))


def _runs(cfg: Config) -> list[dict[str, Any]]:
    """Catalog rows whose run folder still has a manifest, each with the manifest under "manifest".

    The catalog stores each run's folder scrubbed, so on a machine whose hostname is part of the path
    (a user name equal to the hostname) the stored folder does not exist. Folders are found by run id.
    """
    root = Path(cfg.runs_path())
    folders = {path.parent.name: path.parent for pattern in ("*/*/*/*/manifest.json", "*/*/*/manifest.json") for path in root.glob(pattern)}
    rows = []
    for row in Catalog(root).list_runs():
        run_dir = folders.get(row["run_id"], Path(row["run_dir"]))
        manifest = _read_json(run_dir / "manifest.json")
        if manifest:
            rows.append({**row, "run_dir": str(run_dir), "manifest": manifest})
    return rows


def _current_run(runs: list[dict[str, Any]], containers: list[str]) -> dict[str, Any] | None:
    """The run with a container up, whatever its record says: a resumed run keeps the failed step it is
    redoing until that step runs again. Without one (between steps, or none), the newest run still
    running. A run whose machine went down mid-step stays "running" in its record, with no container."""
    live = [row for row in runs if any(name.startswith(f"sma-{row['run_id']}-") for name in containers)]
    pool = live or [row for row in runs if row["status"] == "running"]
    return max(pool, key=lambda row: row["run_id"]) if pool else None


def _running_runs(runs: list[dict[str, Any]], containers: list[str], now: datetime, held: bool | None = None) -> list[dict[str, Any]]:
    """The runs at work when no dispatcher writes its state: each with a container up or a write to its folder in
    the last 10 minutes. With none, the newest run still "running" in its record, but only while a process holds
    the GPU lock (held, from /proc/locks; None: not known): a record a crash left "running" is not a run."""
    live = [row for row in runs if any(name.startswith(f"sma-{row['run_id']}-") for name in containers)]
    pool = live or [row for row in runs if row["status"] == "running"]
    working = [row for row in pool if row in live or not _quiet_for(row, now)]
    if working:
        return sorted(working, key=lambda row: row["run_id"])
    return [max(pool, key=lambda row: row["run_id"])] if pool and held is not False else []


def _queue_panel(runs, current, plan: Path | None, paint, queued: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> Panel:
    """One item per job: each leg with its mark (a pair: the reference, then the 4-bit quant with thinking off and
    on). A leg whose run had failed steps shows as failed; its record says which. Then the plan's lines."""
    failed = sum(1 for job in jobs if job.get("state") == "failed")
    title = paint("QUEUE", "head") + (f"  {_count(len(queued), 'job')}, in the order they start" if queued else "  empty")
    title += f" · {failed} failed (sma queue list)" if failed else ""
    by_id = {row["run_id"]: row for row in runs}
    items = []
    rows = []
    for job in queued:
        legs = []
        for leg in job["legs"]:
            status = {"done": "done", "running": "running", "failed": "failed", "incomplete": "failed"}.get(leg.get("status"), "queued")
            name = _leg_name(leg)
            if status == "running" and leg.get("slot"):
                name += f" on {leg['slot']}"
                row = by_id.get(leg.get("run_id") or "")
                if row and (Path(row["run_dir"]) / "logprobs" / "waiting.json").is_file():
                    name += ", waiting for the reference's logits"
            elif status == "queued" and int(leg.get("guard_trips") or 0):
                name += " (stopped by the memory guard, resumes)"
            legs.append(f"{paint(_MARK[status], status)} {name}")
        rows.append((job, legs))
    # Columns line up: the repo, then each leg in its place (a long one, a running leg's note, only pushes its own line).
    repo_width = max((len(job["legs"][0]["repo"]) for job in queued), default=0)
    leg_widths = [max((len(plain_text(legs[n])) for _job, legs in rows if n < len(legs) and len(plain_text(legs[n])) <= 30), default=0)
                  for n in range(max((len(legs) for _job, legs in rows), default=0))]  # fmt: skip
    for position, (job, legs) in enumerate(rows, 1):
        state, word = ("running", "started") if job["state"] == "claimed" else ("queued", "waiting")
        cells = "  ".join(leg + " " * max(0, leg_widths[n] - len(plain_text(leg))) for n, leg in enumerate(legs))
        # Each line starts with the engine the job runs on (owner, 2026-10-30).
        line = f"  {engine_tag(job.get('engine')):4} {position:>2}. {job['legs'][0]['repo']:<{repo_width}}  {paint(f'{word:7}', state)}  {cells}"
        items.append(Item([line.rstrip()], ("job", str(job["job_id"]))))
    if plan is not None:
        plan_title, plan_items = _plan_items(plan, runs, current, paint)
        items += [Item([plan_title])] + plan_items
    return Panel("queue", title, items)


def _dispatcher_states(cfg: Config, scoped: bool) -> list[dict[str, Any]]:
    """The dispatchers' states (sma worker --slots): this config's, or every one in the queue when not scoped."""
    paths = [state_path(cfg)] if scoped else sorted(state_path(cfg).parent.glob("*.json"))
    return [state for state in map(_dispatcher_state, paths) if state is not None]


def _dispatcher_state(path: Path) -> dict[str, Any] | None:
    """One dispatcher's state file, with "alive" from its pid. None when there is none."""
    state = _read_json(path)
    if not state or not isinstance(state.get("slots"), list):
        return None
    pid = state.get("pid")
    try:
        os.kill(int(pid), 0)
        state["alive"] = True
    except (ProcessLookupError, TypeError, ValueError):
        state["alive"] = False
    except PermissionError:
        state["alive"] = True
    for slot in state["slots"]:
        # A slot whose unit process is gone runs nothing, whatever a stopped dispatcher last wrote.
        if slot.get("state") == "busy" and slot.get("pid") and not unit_alive(slot["pid"]):
            slot["state"] = "idle"
            slot.pop("unit_slot", None)
    # A run across GPUs shows once, on its first slot, with every GPU it holds.
    gpus_of = {slot.get("name"): str(slot.get("gpus")) for slot in state["slots"]}
    merged = []
    for slot in state["slots"]:
        parts = str(slot.get("unit_slot") or "").split("+") if slot.get("state") == "busy" and slot.get("unit_slot") else []
        if parts and slot.get("name") != parts[0]:
            continue
        if parts:
            slot = {**slot, "gpus": ",".join(gpus_of.get(part, "?") for part in parts), "slots": len(parts)}
        merged.append(slot)
    state["slots"] = merged
    return state


def _slot_counts(slots: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Busy, all, and down slots; a run across GPUs counts every slot it holds."""
    count = lambda rows: sum(int(row.get("slots") or 1) for row in rows)  # noqa: E731
    return (count(row for row in slots if row.get("state") == "busy"), count(slots),
            count(row for row in slots if row.get("state") == "down"))  # fmt: skip


def _decode_rates(cfg: Config, series: list[Series], run_id: str, now: datetime) -> tuple[dict, dict] | None:
    """The run's averages and peaks over the last 5 minutes, from its own counters; None without samples."""
    mine = [one for one in series if one.run_id == run_id]
    windows = window(mine, start=now.timestamp() - WINDOWS[0][1], max_gap_s=3 * cfg.telemetry.engine_metrics_interval_s)
    return (combined_rates(windows), combined_peaks(windows)) if windows else None


def _decode(cfg: Config, series: list[Series], run_id: str, now: datetime) -> str:
    """The run's decode rate over the last 5 minutes, average/peak per stream, and over all streams."""
    found = _decode_rates(cfg, series, run_id, now)
    if found is None:
        return ""
    rates, peaks = found
    return f"  decode {_number(rates.get('decode_tok_s'))}/{_number(peaks.get('decode_tok_s'))} tok/s per stream, {_number(rates.get('generated_tok_s_all'))} all"


def _flags(row: dict[str, Any], now: datetime, usual: dict[str | None, dict[str, float]], paint) -> str:
    """The run's warnings, in place on its first line."""
    warnings = [text for level, text in _run_problems(row, now, usual) if level == "warn"]
    return "".join(paint(f"  ! {text.split(';')[0]}", "warn") for text in warnings)


def _started(row: dict[str, Any] | None, now: datetime) -> str:
    started = _time(((row or {}).get("manifest") or {}).get("timestamps", {}).get("started")) if row else None
    return _elapsed(started, None, now) if started else "starting"


@dataclass
class _Work:
    """What one run at work is doing: its list (_run_rows), the line it is on, its time so far, its decode rate."""

    row: dict[str, Any] | None
    rows: list[tuple[str, str, str]] = field(default_factory=list)
    on: str = ""
    progress: str = "starting"
    elapsed: str = "starting"
    rate: str = ""
    rates: str = ""
    flags: str = ""


def _work(cfg: Config, row: dict[str, Any] | None, now: datetime, paint, totals: dict[str, int], usual: dict[str | None, dict[str, float]],
          series: list[Series]) -> _Work:  # fmt: skip
    if row is None:
        return _Work(None)
    rows = _run_rows(row, now, paint, totals)
    on = next(((name, detail) for name, status, detail in rows if status == "running"), None)
    # A step at work with no row of its own yet: quality before its first suite starts.
    on = on or next(((name, "starting") for name, step in _ordered_steps(row["manifest"]) if step.get("status") == "running"), None)
    rate = _decode_rates(cfg, series, row["run_id"], now)
    return _Work(
        row, rows, on[0] if on else "", on[1] if on else "between steps", _started(row, now),
        f"{_number(rate[0].get('decode_tok_s'))} tok/s" if rate else "", _decode(cfg, series, row["run_id"], now),
        _flags(row, now, usual, paint),
    )


def _work_lines(leads: list[str], works: list[_Work]) -> list[str]:
    """One line per run at work, its columns aligned: the lead (mark, slot, model), its time so far, what it is on
    and how far, its decode rate per stream over the last 5 minutes, and its warnings."""
    lead_width = max((len(plain_text(lead)) for lead in leads), default=0)
    on_width = max((len(work.on) for work in works), default=0)
    progress_width = min(max((len(plain_text(work.progress)) for work in works), default=0), _PROGRESS_WIDTH)
    return [
        f"{pad(lead, lead_width)}  {work.elapsed:>8}  {work.on:<{on_width}}  {pad(work.progress, progress_width)}  {work.rate:>11}".rstrip()
        + work.flags
        for lead, work in zip(leads, works)
    ]


def _window(row: dict[str, Any] | None, work: _Work, paint, slot: str | None = None) -> list[str]:
    """A run's window, shown under the slots or runs for the selected one: its title, then its whole list, as
    CURRENT RUN shows a run alone."""
    if row is None:
        return [paint(slot or "", "head") + "  starting"]
    name = f"{row['model_repo']} {row['quant']}"
    title = (paint(slot, "head") + f"  {name}" if slot else paint(name, "head")) + f" on {row['engine']}, thinking "
    title += f"{row['manifest'].get('thinking') or 'off'}  {work.elapsed}{work.rates}"
    return [title, *_list_lines(work.rows, paint)]


def _short_name(repo: str, quant: str, thinking: bool) -> str:
    """The model without its organization or a -GGUF suffix, then the quant: LFM3-1.7B QAD-Q4_0."""
    name = repo.split("/")[-1]
    return f"{name.removesuffix('-GGUF')} {quant}" + (" thinking" if thinking else "")


def _slots_panel(cfg: Config, state: dict[str, Any], by_id: dict[str, dict], now: datetime, paint, totals: dict[str, int],
                 usual: dict[str | None, dict[str, float]], series: list[Series], spark: list[dict[str, Any]]) -> Panel:  # fmt: skip
    """One line per slot: its GPU, its model, its time so far, what it is on and how far, and its decode rate over
    the last 5 minutes (from the run's own counters); then this Spark's runs at work (plain `sma watch`), the same way.
    The selected run shows whole beneath (its window)."""
    slots = state["slots"]
    busy, total, _down = _slot_counts(slots)
    title = paint("SLOTS", "head") + f"  {busy} of {total} busy on the serving machine"
    title += f", {_count(len(spark), 'run')} on this Spark" if spark else ""
    if not state.get("alive"):
        title += paint("  ·  the dispatcher is not running", "failed")
    elif state.get("stopping"):
        title += paint("  ·  the dispatcher is stopping", "warn")
    items, at_work, names, leads, works = [], [], [], [], []
    for slot in slots:
        name, gpus = slot.get("name"), slot.get("gpus")
        if slot.get("state") == "busy":
            short = _short_name(str(slot.get("repo")), str(slot.get("gguf")), slot.get("thinking") == "on")
            at_work.append(len(items))
            names.append(str(name))
            leads.append(f"  {paint(_MARK['running'], 'running')} {name:8} GPU {gpus}  {short}")
            works.append(_work(cfg, by_id.get(str(slot.get("run_id") or "")), now, paint, totals, usual, series))
            items.append(Item([], ("run", str(slot["run_id"])) if slot.get("run_id") else None))
        elif slot.get("state") == "down":
            items.append(Item([f"  {paint(_MARK['failed'], 'failed')} {name:8} GPU {gpus}  {paint('down: ' + str(slot.get('reason') or ''), 'failed')}"]))
        else:
            items.append(Item([f"  {paint(_MARK['pending'], 'pending')} {name:8} GPU {gpus}  {paint('idle', 'pending')}"]))
    for row in spark:
        at_work.append(len(items))
        names.append("spark")
        leads.append(f"  {paint(_MARK['running'], 'running')} {'spark':8} GPU 0  {_short_name(row['model_repo'], row['quant'], bool(_thinking(row)))}")
        works.append(_work(cfg, row, now, paint, totals, usual, series))
        items.append(Item([], ("run", row["run_id"])))
    # The runs' lines are made together, so their columns line up.
    for index, name, line, work in zip(at_work, names, _work_lines(leads, works), works):
        items[index].lines, items[index].window = [line], _window(work.row, work, paint, name)
    guard = state.get("memory_guard") or {}
    notes = [paint(f"  memory guard: stopped {guard.get('run_id')} on {guard.get('slot')} at {guard.get('at')} ({guard.get('source')})", "stopped")] if guard else []
    return Panel("active", title, items, notes)


def _running_panel(cfg: Config, running: list[dict[str, Any]], now: datetime, paint, totals: dict[str, int],
                   usual: dict[str | None, dict[str, float]], series: list[Series]) -> Panel:  # fmt: skip
    """Several runs at once with no dispatcher state: one line per run, the selected one whole beneath."""
    works = [_work(cfg, row, now, paint, totals, usual, series) for row in running]
    leads = [f"  {paint(_MARK['running'], 'running')} {_run_name(row)} on {row['engine']}" for row in running]
    items = [Item([line], ("run", row["run_id"]), _window(row, work, paint)) for line, row, work in zip(_work_lines(leads, works), running, works)]
    return Panel("active", paint("RUNNING", "head") + f"  {len(running)} runs", items)


def _current_panel(current: dict[str, Any] | None, now: datetime, paint, totals: dict[str, int], usual: dict[str | None, dict[str, float]],
                   swe: dict[str, Any] | None) -> Panel:  # fmt: skip
    """The one current run: its title, then one line per benchmark (as before the slots)."""
    if current is None:
        return Panel("active", paint("CURRENT RUN", "head") + "  none")
    lines = _run_lines(current, now, paint, totals, swe)
    return Panel("active", lines[0] + _flags(current, now, usual, paint), [Item(lines[1:], ("run", current["run_id"]))])


def _kernelbench_panel(jobs: list[dict[str, Any]], by_id: dict[str, dict], now: datetime, paint) -> Panel:
    """The KernelBench service's FIFO: the job it is timing, then the queued jobs in the order it takes them."""

    def label(job: dict[str, Any]) -> str:
        row = by_id.get(str(job.get("name") or ""))
        return _run_name(row) if row else str(job.get("name") or job.get("id"))

    def key(job: dict[str, Any]) -> tuple[str, str] | None:
        return ("run", str(job["name"])) if str(job.get("name") or "") in by_id else None

    running = [job for job in jobs if job.get("status") == "running"]
    queued = sorted((job for job in jobs if job.get("status") == "queued"), key=lambda job: str(job.get("created_at") or ""))
    title = paint("KERNELBENCH", "head") + f"  {len(running)} timing, {len(queued)} queued (one job at a time, oldest first)"
    items = []
    for job in running:
        progress = job.get("progress") or {}
        line = f"  {paint(_MARK['running'], 'running')} {label(job)}  {progress.get('done')}/{progress.get('total')} kernels  {_elapsed(_time(job.get('started_at')), None, now)}"
        items.append(Item([line], key(job)))
    for position, job in enumerate(queued, 1):
        line = f"  {paint(_MARK['queued'], 'queued')} {position}. {label(job)}  queued {_elapsed(_time(job.get('created_at')), None, now)}"
        items.append(Item([line], key(job)))
    return Panel("kernelbench", title, items)


def _finished_panel(runs: list[dict[str, Any]], current: dict[str, Any] | None, paint, where: str | None) -> Panel:
    """Every run here that is not running, newest first, with its scores. `--once` shows the latest few. where names
    the machine for the title; None (every machine) marks each run served off this Spark with its hardware instead."""
    done = [row for row in runs if row is not current and _label(row, current) in ("done", "failed", "stopped")]
    done.sort(key=lambda row: row["run_id"], reverse=True)
    items = []
    for row in done:
        summary = _read_json(Path(row["run_dir"]) / "summary.json") or {}
        parts = []
        if isinstance(summary.get("perplexity"), (int, float)):
            parts.append(f"ppl {summary['perplexity']:.2f}")
        if isinstance(summary.get("kl_mean"), (int, float)):
            parts.append(f"KL {summary['kl_mean']:.4f}")
        for suite, item in (summary.get("quality") or {}).items():
            if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
                parts.append(f"{suite} {_percent(item['score'], paint)}")
        state = _label(row, current)
        scores = "  " + " · ".join(parts) if parts else ""
        note = paint(" (stopped, resumable)", "stopped") if state == "stopped" else ""
        hardware = _served_on(row) if where is None else None
        served = paint(f" on {hardware}", "pending") if hardware else ""
        items.append(Item([f"  {paint(_MARK[state], state)} {_run_name(row)}{served}{note}{scores}"], ("run", row["run_id"])))
    scope = f" {where}" if where else ""
    title = paint("FINISHED", "head") + (f"  {_count(len(done), 'run')}{scope}, newest first" if done else f"  none{scope} yet")
    return Panel("finished", title, items, once_items=_FINISHED_SHOWN)


# The detail view.


def _detail(cfg: Config, view: View, open_item: tuple[str, str], by_id: dict[str, dict], jobs: list[dict[str, Any]],
            state: dict[str, Any] | None, active_ids: set[str], usual: dict[str | None, dict[str, float]], totals: dict[str, int], paint) -> list[str] | None:  # fmt: skip
    kind, key = open_item
    if kind == "job":
        job = next((job for job in jobs if job.get("job_id") == key), None)
        return _job_detail(job, view.now, paint) if job else None
    row = by_id.get(key)
    if row is None:
        return None
    slot = next((slot for slot in (state or {}).get("slots") or [] if slot.get("run_id") == key and slot.get("state") == "busy"), None)
    swe = view.swe if view.swe and view.swe.get("run_id") == key else None
    return _run_detail(cfg, view, row, slot, key in active_ids, usual, totals, swe, paint)


def _run_detail(cfg: Config, view: View, row: dict[str, Any], slot: dict[str, Any] | None, running: bool, usual: dict[str | None, dict[str, float]],
                totals: dict[str, int], swe: dict[str, Any] | None, paint) -> list[str]:  # fmt: skip
    """Everything about one run: where it runs, its list as the main screen shows it (with whole errors), its
    serving rates, and its recent events."""
    now, manifest, run_dir = view.now, row["manifest"], Path(row["run_dir"])
    stamps = manifest.get("timestamps") or {}
    started, finished = _time(stamps.get("started")), _time(stamps.get("finished"))
    lines = [paint(f"RUN  {row['run_id']}", "head") + f"  {row['model_repo']} {row['quant']} on {row['engine']}, thinking {manifest.get('thinking') or 'off'}"
             + (_flags(row, now, usual, paint) if running else "")]  # fmt: skip
    serving = manifest.get("serving") if isinstance(manifest.get("serving"), dict) else {}
    facts = ["running" if running else _label(row, None)]
    if slot:
        facts.append(f"slot {slot.get('name')}, GPU {slot.get('gpus')}")
    elif serving.get("slot"):
        facts.append(f"slot {serving['slot']}, GPU {serving.get('gpus')}")
    if serving.get("hardware"):
        facts.append(f"served on {serving['hardware']}" + (f" ({', '.join(map(str, serving['gpu_names']))})" if serving.get("gpu_names") else ""))
    if started:
        facts.append(f"started {_clock(started)}, " + (f"took {_elapsed(started, finished, now)}" if finished else f"{_elapsed(started, None, now)} ago"))
    lines += ["  " + " · ".join(facts), f"  folder {run_dir}", ""]
    lines += _list_lines(_run_rows(row, now, paint, totals, swe, active=running, cut=None), paint)
    lines += [""] + _serving_lines(cfg, view, paint, row["run_id"])
    events = _events(run_dir)
    lines += ["", paint("EVENTS", "head") + f"  the last {len(events)} of logs/sma.log"] + [f"  {event}" for event in events]
    return lines


def _job_detail(job: dict[str, Any], now: datetime, paint) -> list[str]:
    """One queue job: its claim and heartbeat, then each leg with its status, slot, run, attempts, and notes."""
    created = _time(job.get("created"))
    lines = [paint(f"JOB  {job['job_id']}", "head") + f"  {job.get('kind')} on {engine_tag(job.get('engine'))}, {job.get('state')}"
             + (f", queued {_elapsed(created, None, now)} ago" if created else "")]  # fmt: skip
    claim = job.get("claim") if isinstance(job.get("claim"), dict) else {}
    if claim:
        beat = _time(job.get("heartbeat"))
        heartbeat = f"; heartbeat {_elapsed(beat, None, now)} ago" if beat else "; no heartbeat"
        lines.append(f"  claimed by {claim.get('worker_id')} on {claim.get('host_alias')} at {_clock(_time(claim.get('when')))}{heartbeat}")
    if job.get("attempts"):
        lines.append(f"  attempts {job['attempts']}")
    if job.get("error"):
        lines.append("  " + paint(_one_line(job["error"]), "failed"))
    lines += ["", paint("LEGS", "head")]
    for index, leg in enumerate(job["legs"], 1):
        status = str(leg.get("status") or "pending")
        style = {"done": "done", "running": "running", "failed": "failed", "incomplete": "failed"}.get(status, "queued")
        lines.append(f"  {paint(_MARK[style], style)} {index}. {leg.get('repo')} {leg.get('gguf') or leg.get('quant')}, thinking {leg.get('thinking') or 'off'}  {status}")
        facts = [
            f"slot {leg['slot']}" if leg.get("slot") else "",
            f"run {leg['run_id']}" if leg.get("run_id") else "",
            f"attempts {leg['attempts']}" if leg.get("attempts") else "",
            f"waits until {leg['not_before']}" if leg.get("not_before") else "",
            f"stopped by the memory guard {_count(int(leg['guard_trips']), 'time')}" if leg.get("guard_trips") else "",
        ]
        if any(facts):
            lines.append("       " + " · ".join(fact for fact in facts if fact))
        for note in leg.get("guard_notes") or []:
            text = f"{note.get('at')} {note.get('note')}" if isinstance(note, dict) else str(note)
            lines.append("       " + paint(f"memory guard: {_one_line(text)}", "stopped"))
        if leg.get("error"):
            lines.append("       " + paint(_one_line(leg["error"]), "failed"))
        if leg.get("dead_lettered"):
            lines.append("       " + paint("failed for good: the queue keeps its record in failed/", "failed"))
    return lines


def _events(run_dir: Path, count: int = 12) -> list[str]:
    """The last events of the run's logs/sma.log, one line each: the time, then the event's message."""
    path, chunk = run_dir / "logs" / "sma.log", 32768
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - chunk))
            rows = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    events = []
    for raw in rows[1 if size > chunk else 0 :]:
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if isinstance(event, dict):
            moment = _time(event.get("ts"))
            events.append(f"{_clock(moment) if moment else '':8}  {_one_line(event.get('message') or event.get('event') or '')}")
    return events[-count:]


def _clock(moment: datetime | None) -> str:
    """A time of day on this machine's clock."""
    return moment.astimezone().strftime("%H:%M:%S") if moment else ""


def engine_tag(engine: str | None) -> str:
    """The short engine name a queue line starts with."""
    return {"sglang": "SGL", "llamacpp": "CPP", "vllm": "VLLM"}.get(engine or "", "?")


def _plan_items(plan: Path, runs, current, paint) -> tuple[str, list[Item]]:
    """A plan file lists the runs a script will start, one "REPO QUANT" per line, in order; "REPO QUANT
    thinking" is the thinking-on run. A line "since TIME" (ISO time, or a run id) ignores runs created
    before it. Each line shows the newest run of that repo, quant, and thinking mode since then, or queued.
    Returns the PLAN title line and one item per line, which opens its run."""
    try:
        text = plan.read_text(encoding="utf-8")
    except OSError as exc:
        raise SmaError(f"Could not read the plan file {plan}: {exc}") from exc
    since, items = "", []
    for raw in text.splitlines():
        words = raw.split("#", 1)[0].split()
        if len(words) == 2 and words[0] == "since":
            since = _run_id_time(words[1])
        elif len(words) == 2 or (len(words) == 3 and words[2] == "thinking"):
            items.append((words[0], words[1], len(words) == 3))
        elif words:
            raise SmaError(f"Plan file {plan}: '{raw.strip()}' is not 'REPO QUANT', 'REPO QUANT thinking', or 'since TIME'.")
    title = paint("PLAN", "head") + f"  {plan}" + (f"  (runs since {since})" if since else "")
    lines = []
    for repo, quant, thinking in items:
        matches = [
            row for row in runs
            if row["model_repo"] == repo and str(row["quant"]).lower() == quant.lower() and row["run_id"] >= since
            and bool(_thinking(row)) == thinking
        ]
        row = max(matches, key=lambda r: r["run_id"]) if matches else None
        state = "queued" if row is None else _label(row, current)
        name = f"{repo} {quant}" + (" thinking" if thinking else "")
        line = f"  {paint(_MARK.get(state, '?'), state)} {name:44} {paint(state, state):8}  {row['run_id'] if row else ''}"
        lines.append(Item([line], ("run", row["run_id"]) if row else None))
    return title, lines


def _run_id_time(value: str) -> str:
    """A since value as a run-id prefix (20261029T020500Z), which sorts like the run ids."""
    if re.fullmatch(r"\d{8}T\d{6}Z.*", value):
        return value[:16]
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SmaError(f"'since {value}' is not an ISO time or a run id.") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_lines(
    row: dict[str, Any], now: datetime, paint, totals: dict[str, int] | None = None, swe: dict[str, Any] | None = None
) -> list[str]:
    """The current run's title, then its list (_run_rows)."""
    manifest = row["manifest"]
    started = _time((manifest.get("timestamps") or {}).get("started"))
    title = (
        paint(f"CURRENT RUN  {row['run_id']}", "head")
        + f"  {row['model_repo']} {row['quant']} on {row['engine']}, thinking {manifest.get('thinking') or 'off'}  "
        + _elapsed(started, None, now)
    )
    return [title, *_list_lines(_run_rows(row, now, paint, totals, swe), paint)]


def _run_rows(
    row: dict[str, Any], now: datetime, paint, totals: dict[str, int] | None = None, swe: dict[str, Any] | None = None,
    *, active: bool = True, cut: int | None = 100,
) -> list[tuple[str, str, str]]:
    """A run's list as (name, state, detail): one row per benchmark (its score, its progress, or queued), in the
    order the run does them, after the setup and bookkeeping steps (fetch, servers, results, teardown) that are
    running or failed. Errors are cut to cut characters (None: whole). In a run no longer at work (active False),
    what its record left running is shown stopped."""
    manifest, run_dir = row["manifest"], Path(row["run_dir"])
    steps = dict(_ordered_steps(manifest))
    rows: list[tuple[str, str, str]] = []
    for name, step in steps.items():
        status = step.get("status")
        if name not in _BENCHMARK_STEPS and status in ("running", "failed"):
            error = _one_line(step.get("error") or "")[:cut]
            rows.append((name, status, f"{status}  {paint(error, 'failed')}" if error else status))
    waiting = _read_json(run_dir / "logprobs" / "waiting.json")
    for name, status, detail in _benchmarks(manifest, run_dir, steps, now, totals or {}, paint, swe, cut=cut):
        if name == "perplexity" and status == "running" and waiting:
            detail = f"waiting for the reference's logits ({waiting.get('reference_run_id')})"
        rows.append((name, status, _one_line(detail)))  # An error's traceback would break the one-line layout.
    if not active:
        rows = [(name, "stopped", _stopped(detail)) if status == "running" else (name, status, detail) for name, status, detail in rows]
    return rows


def _stopped(detail: str) -> str:
    """A running row's detail in a run no longer at work: stopped, how far it got, but not the time since."""
    left = re.sub(r"(^|\s+)\d+(h\d+m|m\d+s|s)$", "", detail)
    return "stopped" if left in ("", "running") else f"stopped  {left}"


def _list_lines(rows: list[tuple[str, str, str]], paint) -> list[str]:
    return [f"  {paint(_MARK.get(status, '?'), status)} {name:16} {paint('queued', 'queued') if status == 'queued' else detail}" for name, status, detail in rows]


# The steps that measure something. Each gets a line; the others only while running or failed.
_BENCHMARK_STEPS = ("determinism", "logprobs", "quality", "kernelbench", "swebench", "browsecomp_plus", "speed")


def _benchmarks(
    manifest, run_dir: Path, steps: dict[str, dict], now: datetime, totals: dict[str, int], paint=_identity,
    swe: dict[str, Any] | None = None, *, cut: int | None = 100,
) -> list[tuple[str, str, str]]:
    """(name, state, detail) for each benchmark, in the order the run does them; a failed step's error cut to cut
    characters (None: whole)."""

    def state(step: str) -> str:
        status = (steps.get(step) or {}).get("status") or "pending"
        return "queued" if status == "pending" else status

    def took(step: str) -> str:
        info = steps.get(step) or {}
        if info.get("status") == "failed":
            return str(info.get("error") or "failed")[:cut]
        if info.get("status") == "running" and int(info.get("attempts") or 0) > 1:
            # A redone step keeps its first attempt's start and finish, so no elapsed time is shown.
            return f"attempt {info['attempts']}"
        return _elapsed(_time(info.get("started")), _time(info.get("finished")), now) if info.get("started") else ""

    rows: list[tuple[str, str, str]] = []
    rows.append(_determinism_row(run_dir, state("determinism"), took("determinism")))
    perplexity = (_read_json(run_dir / "logprobs" / "perplexity.json") or {}).get("perplexity")
    kl = (_read_json(run_dir / "logprobs" / "kl_vs_reference.json") or {}).get("kl_mean")
    detail = " · ".join(part for part in (
        f"{perplexity:.3f}" if isinstance(perplexity, (int, float)) else "",
        f"KL {kl:.4f}" if isinstance(kl, (int, float)) else "",
    ) if part)
    rows.append(("perplexity", state("logprobs"), detail or took("logprobs")))
    suites = [
        _suite_row(run_dir, {"questions": totals.get(suite["name"]), **{k: v for k, v in suite.items() if v is not None}}, now, paint)
        for suite in _suites(manifest)
    ]
    # Suites run one at a time. An unfinished attempt older than the newest one, or any while quality is not
    # running, was cut short by a stop (a resume starts it again), so it is not shown as running.
    started = {name: _suite_started(run_dir / "quality" / name) for name, status, _ in suites if status == "running"}
    current = max(started, key=lambda name: started[name]) if started and state("quality") == "running" else None
    for name, status, detail in suites:
        if status == "running" and name != current:
            answered = detail.rsplit("  ", 1)[0]  # Without the time since it began.
            status, detail = "stopped", "stopped" + (f"  {answered}" if answered != "running" else "")
        rows.append((name, status, detail))
    rows += _agent_rows(manifest, run_dir, state, took, paint, now, swe)
    speed = _read_json(run_dir / "speed" / "summary.json") or {}
    parts = [
        f"prefill {_number(speed.get('prefill_tok_s'))} tok/s" if isinstance(speed.get("prefill_tok_s"), (int, float)) else "",
        f"decode {_number(speed.get('decode_tok_s'))} tok/s" if isinstance(speed.get("decode_tok_s"), (int, float)) else "",
        f"8 streams {_number(speed.get('output_tok_s_c8'))} tok/s" if isinstance(speed.get("output_tok_s_c8"), (int, float)) else "",
    ]
    rows.append(("speed", state("speed"), " · ".join(part for part in parts if part) or took("speed")))
    return rows


def _suites(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = (manifest.get("config_snapshot") or {}).get("quality") or {}
    return [suite for suite in snapshot.get("suites") or [] if isinstance(suite, dict) and suite.get("name")]


def _determinism_row(run_dir: Path, status: str, took: str) -> tuple[str, str, str]:
    """Passed, or failed naming the batch sizes whose output differs from one request alone."""
    if status != "done":
        return "determinism", status, took
    result = _read_json(run_dir / "determinism.json") or {}
    if result.get("passed"):
        return "determinism", "done", "passed"
    differ = [str(row.get("batch_size")) for row in result.get("results") or [] if isinstance(row, dict) and row.get("matches_batch_1") is False]
    reason = f": batch {', '.join(differ)} differ from batch 1" if differ else ""
    return "determinism", "failed", f"failed{reason}"


def _suite_row(run_dir: Path, suite: dict[str, Any], now: datetime, paint=_identity) -> tuple[str, str, str]:
    name = suite["name"]
    folder = run_dir / "quality" / name
    scores = _read_json(folder / "scores.json")
    if scores and (folder / ".done").is_file():
        if scores.get("primary_score") is None:
            return name, "failed", f"null  {str(scores.get('error') or '')[:90]}"
        detail = f"{_percent(scores['primary_score'], paint)} {scores.get('primary_metric')}  n={scores.get('n')}"
        # IFEval's headline is strict by prompt; the loose and by-instruction measures follow it.
        measures = scores.get("measures") if isinstance(scores.get("measures"), dict) else {}
        shown = [
            f"{label} {_percent(measures[key], paint)}"
            for key, label in (
                ("prompt_level_loose_acc", "loose"),
                ("inst_level_strict_acc", "instr strict"),
                ("inst_level_loose_acc", "instr loose"),
            )
            if isinstance(measures.get(key), (int, float))
        ]
        detail += f"  ({' · '.join(shown)})" if shown else ""
        # Answers that ran out of room: the output limit, or the end of the stream's context.
        cut = scores.get("answers_cut_off")
        return name, "done", detail + (f"  {paint(f'{cut} cut off', 'failed')}" if isinstance(cut, int) and cut else "")
    attempts = sorted(folder.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0)
    # A suite that failed writes scores.json with its error and no .done marker; a retry starts a new attempt.
    if scores and scores.get("error") and (not attempts or (folder / "scores.json").stat().st_mtime >= attempts[-1].stat().st_mtime):
        return name, "failed", str(scores["error"])[:90]
    if not attempts:
        return name, "queued", ""
    total = suite.get("limit") or suite.get("questions")
    limit = f"/{total:,}" if total else ""
    logs = list(attempts[-1].glob("*/inference_log.jsonl"))
    # A suite in NVIDIA's eval-factory container (BFCL) writes one JSON line per answered question into
    # a file per category. Its total lives in the container's dataset: the suite's `questions` says it.
    results = sorted(attempts[-1].glob("result/*/*_result.json")) if suite.get("image") else []
    # LiveCodeBench's runner keeps its answers in cache/<file>.json, one key per problem, saved only when a
    # batch of 10 is done. Its proxy counts each response as it comes, one request per problem and sample.
    cache = sorted(attempts[-1].glob("cache/*.json")) if suite.get("image") else []
    stats = _response_stats(attempts[-1]) if "codegeneration" in str(suite.get("task") or "") else None
    if logs:
        answered = f"{sum(_count_answers(log) for log in logs):,}{limit or '/all'} answered"
    elif stats is not None:
        samples = int(((suite.get("params") or {}).get("extra") or {}).get("n_samples") or 1)
        cut = int((stats.get("finish_reason") or {}).get("length") or 0)
        answered = f"{int(stats.get('successful_count') or 0) // samples:,}{limit} answered"
        answered += f" · {paint(f'{cut} cut off', 'failed')}" if cut else ""
        # The runner writes its answers file when every problem has one, then runs their code against the
        # hidden tests, which can take an hour, and writes *_eval_all.json when that is done.
        written = [path for path in attempts[-1].glob("*_Scenario.*.json") if "_eval" not in path.name]
        answered += " · running the tests" if written and not list(attempts[-1].glob("*_eval_all.json")) else ""
    elif cache and isinstance(_read_json(cache[-1]), dict):
        answered = f"{len(_read_json(cache[-1])):,}{limit} answered"
    elif results:
        answered = f"{sum(_count_lines(path) for path in results):,}{limit} answered · {len(results)} categor{'y' if len(results) == 1 else 'ies'}"
    else:
        answered = "running"
    return name, "running", f"{answered}  {_elapsed(_suite_started(folder), None, now)}"


def _response_stats(attempt: Path) -> dict[str, Any] | None:
    """The eval-factory proxy's running totals (its response-stats interceptor): responses, how each
    ended, status codes. Read only while the container has the database open: a WAL database whose -wal
    and -shm files are gone would have them made again by a reader, and sma watch writes nothing."""
    import sqlite3
    from contextlib import closing

    db = attempt / "response_stats_cache" / "cache.db"
    if not all(Path(f"{db}{suffix}").is_file() for suffix in ("", "-wal", "-shm")):
        return None
    try:
        with closing(sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)) as connection:
            row = connection.execute("SELECT value FROM Cache WHERE key = 'interceptor_state'").fetchone()
        state = json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError):
        return None
    stats = state.get("aggregated_stats") if isinstance(state, dict) else None
    return stats if isinstance(stats, dict) else None


def _suite_started(folder: Path) -> datetime:
    """When the suite's latest attempt began: command.json is written when it starts."""
    attempts = sorted(folder.glob("attempt-*"), key=lambda p: p.stat().st_mtime)
    marker = folder / "command.json" if (folder / "command.json").is_file() else (attempts[-1] if attempts else folder)
    return datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc)


def _agent_rows(
    manifest, run_dir: Path, state, took, paint=_identity, now: datetime | None = None, swe: dict[str, Any] | None = None
) -> list[tuple[str, str, str]]:
    """KernelBench, SWE-bench, and BrowseComp-Plus. KernelBench and BrowseComp-Plus are scored by their own
    steps after BrowseComp-Plus, before speed."""
    rows = []
    # A suite the run's own config turned off gets no line (SWE-bench, off since 2026-10-29).
    snapshot = manifest.get("config_snapshot") or {}
    for step, label, section in (
        ("kernelbench", "kernelbench", "kernelbench"),
        ("swebench", "swebench", "swebench"),
        ("browsecomp_plus", "browsecomp", "browsecomp"),
    ):
        if (snapshot.get(section) or {}).get("enabled", True) is False:
            continue
        folder = run_dir / "quality" / _AGENT_SUITES[step]
        status = state(step)
        # A step being redone keeps its last attempt's scores.json until it writes a new one.
        scores = None if status == "running" else _read_json(folder / "scores.json")
        if scores and scores.get("primary_score") is not None:
            detail = f"{_percent(scores['primary_score'], paint)} {scores.get('primary_metric')}  n={scores.get('n')}"
            counts = scores.get("kernelbench") if isinstance(scores.get("kernelbench"), dict) else {}
            if counts.get("total"):
                # fast_1: correct and faster than eager PyTorch. Most kernels fall before that.
                detail += f"  ({counts.get('correct')} of {counts['total']} correct, {counts.get('compiled')} compiled)"
            rows.append((label, "done", detail))
            continue
        if scores and scores.get("error"):
            rows.append((label, "failed", str(scores["error"])[:90]))
            continue
        detail = took(step)
        if status == "running" and now is not None:
            # Each attempt writes settings.json as it starts: the time since then, and the attempt when redone.
            begun = _mtime(folder / "settings.json")
            since = _elapsed(begun, None, now) if begun else ""
            detail = " · ".join(part for part in (since, detail if detail.startswith("attempt") else "") if part)
        if step == "kernelbench":
            written = sum(1 for _ in (folder / "generations").glob("*")) if (folder / "generations").is_dir() else 0
            detail = "kernels sent; scored after BrowseComp-Plus" if (folder / "job.json").is_file() else f"{written}/100 kernels written  {detail}"
        elif step == "browsecomp_plus":
            limit = ((manifest.get("config_snapshot") or {}).get("browsecomp") or {}).get("limit")
            answered = sum(1 for _ in (folder / "queries").glob("*.json")) if (folder / "queries").is_dir() else 0
            detail = f"{answered}/{limit} answered" + ("; judging next" if status == "done" else f"  {detail}")
        elif status == "running":
            detail = f"{_swe_progress(swe)}  {detail}"
        if status == "failed":
            detail = took(step)
        elif status == "done":
            status = "running"  # Measured, not yet scored.
        rows.append((label, status, detail))
    return rows


def _swe_progress(swe: dict[str, Any] | None) -> str:
    """Finished tasks of the total, by how each ended, as the runner last reported them."""
    if not swe or not swe.get("total"):
        return "agents on the runner"
    statuses = swe.get("statuses") or {}
    submitted, limited = statuses.get("Submitted", 0), statuses.get("LimitsExceeded", 0)
    other = int(swe.get("done") or 0) - submitted - limited
    parts = [f"{submitted} submitted", f"{limited} hit the step limit"] + ([f"{other} other"] if other else [])
    return f"{swe['done']}/{swe['total']} tasks ({', '.join(parts)})"


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return None


# The SERVING table's rows: (label, key in serving_rates). Columns are the windows.
# Per stream: what one request sees while sharing the server. All streams: the server's total over
# wall-clock time. Prefill has no all-streams row: prompts take a sliver of the server's time, so prompt
# tokens over wall-clock time measure how fast prompts arrive, not how fast they are processed.
_SERVING_ROWS = (
    ("prefill, per stream", "prefill_tok_s"),
    ("decode, per stream", "decode_tok_s"),
    ("decode, all streams", "generated_tok_s_all"),
    ("ITL ms, per stream", "itl_ms"),
)


def _serving_lines(cfg: Config, view: View, paint, run_id: str | None = None) -> list[str]:
    """The current run's serving rates: since its server started, and over the last 5 minutes, hour, and
    24 hours, from its own counters (engine_metrics.jsonl, or sma watch's own reads for a run without one).
    Then one more column for every run of the last 24 hours together, whatever engine served: each rate is
    tokens over seconds summed across them (owner, 2026-10-29)."""
    now, live = view.now, view.live
    stamp = now.timestamp()
    series = recent_series(cfg.runs_path(), now=stamp, span_s=WINDOWS[-1][1])
    own_reads = live is not None and bool(live.engine) and len(live.samples) > 1
    if own_reads:
        series.append(Series(live.engine, live.samples, live.run_id))
    mine = [one for one in series if run_id is not None and one.run_id == run_id]
    # label -> (averages, peaks). A peak is the highest rate over one pair of samples in the window.
    columns: dict[str, tuple[dict[str, float | None], dict[str, float]]] = {}
    max_gap_s = 3 * max(cfg.telemetry.engine_metrics_interval_s, live.interval_s if live else 0.0)
    if live is not None and live.latest and live.engine and live.run_id == run_id:
        # Counters run from the server's start, so the latest read is its average since then. All-stream
        # rates divide by the container's uptime; without it they are not shown.
        uptime = (now - live.started).total_seconds() if live.started else 0.0
        since = window(mine, start=live.started.timestamp(), max_gap_s=max_gap_s) if live.started else []
        columns["since start"] = (get_backend(live.engine).serving_rates(live.latest, live.latest, uptime), combined_peaks(since))
    for label, span in WINDOWS:
        windows = window(mine, start=stamp - span, max_gap_s=max_gap_s)
        if windows:
            columns[label] = (combined_rates(windows), combined_peaks(windows))
    everything = window(series, start=stamp - WINDOWS[-1][1], max_gap_s=max_gap_s)
    if everything:
        columns[_ALL_RUNS] = (combined_rates(everything), combined_peaks(everything))
    if not columns:
        return [
            paint("SERVING", "head") + "  no engine counters in the last 24 hours, and no server up.",
            f"  A run records them every {cfg.telemetry.engine_metrics_interval_s:g} s while its server is up.",
        ]
    labels = ["since start"] + [label for label, _span in WINDOWS] + [_ALL_RUNS]

    def cells(key: str) -> str:
        return "".join(_rate_cell(*columns.get(label, ({}, {})), key, paint) for label in labels)

    lines = [
        paint("SERVING", "head") + "  tok/s, ITL in ms, average/peak: this run, then every run of the last 24 hours",
        f"  {'':20}" + "".join(f"{label:>{_CELL}}" for label in labels),
        *(f"  {name:20}{cells(key)}" for name, key in _SERVING_ROWS),
    ]
    if own_reads:
        lines.append(paint("  this run's windows are sma watch's own reads (the run records no counters)", "pending"))
    return lines


def _serving_rows(cfg: Config, now: datetime, paint, runs: list[tuple[str, str]]) -> list[str]:
    """Several runs at once: one row per run (label, run id) over its last 5 minutes, then every run of the last
    24 hours together. Each run's own windows are in its detail view."""
    stamp = now.timestamp()
    series = recent_series(cfg.runs_path(), now=stamp, span_s=WINDOWS[-1][1])
    max_gap_s = 3 * cfg.telemetry.engine_metrics_interval_s
    rows = [(label, window([one for one in series if one.run_id == run_id], start=stamp - WINDOWS[0][1], max_gap_s=max_gap_s)) for label, run_id in runs]
    rows.append((_ALL_RUNS_ROW, window(series, start=stamp - WINDOWS[-1][1], max_gap_s=max_gap_s)))
    lines = [
        paint("SERVING", "head") + "  tok/s per stream unless all, ITL in ms, average/peak: each run's last 5 minutes, then all of 24 hours",
        f"  {'':28}" + "".join(f"{name:>{_CELL}}" for name in ("prefill", "decode", "decode, all", "ITL ms")),
    ]
    for label, windows in rows:
        averages, peaks = (combined_rates(windows), combined_peaks(windows)) if windows else ({}, {})
        lines.append(f"  {label[:27]:28}" + "".join(_rate_cell(averages, peaks, key, paint) for _name, key in _SERVING_ROWS))
    return lines


def _rate_cell(averages: dict[str, float | None], peaks: dict[str, float], key: str, paint) -> str:
    """average/peak, right-aligned in _CELL characters: the average blue, the peak purple."""
    mean, peak = averages.get(key), peaks.get(key)
    if mean is None and peak is None:
        return f"{'—':>{_CELL}}"
    text = f"{_number(mean)}/{_number(peak)}"
    return " " * max(0, _CELL - len(text)) + paint(_number(mean), "avg") + "/" + paint(_number(peak), "max")


# The last column: every run of the last 24 hours, whatever engine served.
_ALL_RUNS = "24 h, all"
_ALL_RUNS_ROW = "all runs, last 24 hours"
# A SERVING cell's width: "average/peak".
_CELL = 14


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.1f}"


def _label(row: dict[str, Any], current: dict[str, Any] | None) -> str:
    """running: it has the Spark. stopped: steps are left, and `sma resume` goes on from them. failed: every
    step ran and some failed. done: every step succeeded. The catalog calls a run failed as soon as one step
    is, and finished whenever its process ends, even when it was stopped on purpose to be resumed."""
    if current is not None and row["run_id"] == current["run_id"]:
        return "running"
    statuses = [step.get("status") for step in (row["manifest"].get("steps") or {}).values() if isinstance(step, dict)]
    if any(status in ("pending", "running") for status in statuses):
        return "stopped"
    if "failed" in statuses:
        return "failed"
    return "done" if statuses else str(row["status"])


def _thinking(row: dict[str, Any]) -> str:
    """" thinking on" for a thinking-on run; nothing for the default (off, or a run from before the setting)."""
    return " thinking on" if (row.get("manifest") or {}).get("thinking") == "on" else ""


def _elapsed(start: datetime | None, end: datetime | None, now: datetime) -> str:
    if start is None:
        return ""
    seconds = max(0, int(((end or now) - start).total_seconds()))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _count_answers(path: Path) -> int:
    """Answers in a NeMo Evaluator inference log. Its first line is a record of the config, not an answer."""
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if not (line.startswith(b"{") and _is_meta(line)))
    except OSError:
        return 0


def _is_meta(line: bytes) -> bool:
    if b'"_type"' not in line:
        return False
    try:
        return json.loads(line).get("_type") == "meta"
    except (ValueError, AttributeError):
        return False


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0
