"""`sma watch`: a live view of this Spark's work. The queue, the run in progress with each step and
benchmark, serving rates over the last 5 minutes, hour, and 24 hours, and the latest finished runs.

It reads run folders, the queue, /proc/meminfo, and the SMA containers that are up. It changes nothing.
The screen is scrubbed like a run record (no hostname, lab machine, or key), so it can be shared.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from sma.backends import get_backend
from sma.catalog import Catalog
from sma.config import Config
from sma.errors import SmaError
from sma.machines import Machine
from sma.queue import JobQueue
from sma.records import STEP_NAMES
from sma.scrub import scrub_text
from sma.serving import WINDOWS, Series, recent_series, window
from sma.telemetry import ENGINE_METRICS_FILE, engine_values

# Steps that send work elsewhere and report their own progress in the run folder.
_AGENT_SUITES = {"kernelbench": "kernelbench_l1", "swebench": "swebench_verified_mini", "browsecomp_plus": "browsecomp_plus"}
_MARK = {"done": "✓", "running": "▶", "failed": "✗", "pending": "·", "queued": "·"}
_COLOR = {"done": "32", "running": "33", "failed": "31", "pending": "2", "queued": "2", "head": "1;36"}
_FINISHED_SHOWN = 3
# The machines are asked again this often: whether each answers can change, what it is does not.
_MACHINES_EVERY_S = 60.0


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


def render(cfg: Config, view: View, *, plan: Path | None = None) -> str:
    """One screen of text."""
    paint = _painter(view.color)
    runs = _runs(cfg)
    current = _current_run(runs, view.containers)
    lines = _header(view, paint)
    lines += _queue_lines(cfg, runs, current, plan, paint)
    # A run's own config names its suites; a total added to sma.yaml since then still counts for it.
    totals = {suite.name: suite.questions for suite in cfg.quality.suites if suite.questions}
    lines += (_run_lines(current, view.now, paint, totals) if current else [paint("CURRENT RUN", "head") + "  none"]) + [""]
    lines += _serving_lines(cfg, view, paint) + [""]
    lines += _finished_lines(runs, current, paint)
    text = scrub_text("\n".join(lines))
    return "\n".join(_clip(line, view.width) for line in text.splitlines()) if view.width else text


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
    out: TextIO | None = None,
) -> None:
    """Print the screen once, or redraw it every interval_s seconds until Ctrl-C. scrape(base_url, path)
    returns the engine's /metrics text, or "" when it does not answer."""
    out = out or sys.stdout
    color = out.isatty() and not os.environ.get("NO_COLOR")
    live = Live(interval_s=interval_s)
    known: list[Machine] | None = None
    asked = 0.0
    while True:
        up, now = containers(), datetime.now(timezone.utc)
        read_live(cfg, live, up, now, scrape, started_at)
        if machines is not None and time.monotonic() - asked >= _MACHINES_EVERY_S:
            known, asked = machines(), time.monotonic()
        width = shutil.get_terminal_size().columns if out.isatty() else None
        view = View(up, memory(), now, color, live, width, known, None if once else interval_s)
        text = render(cfg, view, plan=plan)
        if once:
            out.write(text + "\n")
            return
        out.write("\033[H\033[2J" + text + "\n")
        out.flush()
        time.sleep(interval_s)


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
        code = _COLOR.get(style)
        return f"\033[{code}m{scrub_text(text)}\033[0m" if color and code else text

    return paint


def _header(view: View, paint) -> list[str]:
    title = f"{paint('SMA watch', 'head')}  {view.now.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
    if view.interval_s:
        title += paint(f"  ·  redraws every {view.interval_s:g} s, Ctrl-C quits", "pending")
    if view.memory is None:
        memory = "memory unknown"
    else:
        free, total = view.memory[0] / 2**30, view.memory[1] / 2**30
        style = "failed" if free < 16 else "running" if free < 32 else "done"
        memory = f"memory {paint(f'{free:.1f}', style)} of {total:.1f} GiB free"
        free_only = f"{paint(f'{free:.1f}', style)} free"
    lines = [title, "", paint("MACHINES", "head")]
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
    return lines


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


def _queue_lines(cfg, runs, current, plan: Path | None, paint) -> list[str]:
    jobs = [job for job in JobQueue(cfg.queue_path()).list_jobs() if job.get("state") in ("pending", "claimed")]
    lines = [paint("QUEUE", "head") + f"  {len(jobs)} job{'' if len(jobs) == 1 else 's'} in sma queue"]
    for job in jobs:
        legs = ", ".join(f"{leg['repo']} {leg.get('gguf') or leg['quant']}" for leg in job["legs"])
        state = "running" if job["state"] == "claimed" else "queued"
        lines.append(f"  {paint(_MARK[state], state)} {legs:44} {paint(job['state'], state)}  {job['job_id']}")
    if plan is not None:
        lines += _plan_lines(plan, runs, current, paint)
    return lines


def _plan_lines(plan: Path, runs, current, paint) -> list[str]:
    """A plan file lists the runs a script will start, one "REPO QUANT" per line, in order. A line
    "since TIME" (ISO time, or a run id) ignores runs created before it. Each line shows the newest run
    of that repo and quant since then, or queued."""
    try:
        text = plan.read_text(encoding="utf-8")
    except OSError as exc:
        raise SmaError(f"Could not read the plan file {plan}: {exc}") from exc
    since, items = "", []
    for raw in text.splitlines():
        words = raw.split("#", 1)[0].split()
        if len(words) == 2 and words[0] == "since":
            since = _run_id_time(words[1])
        elif len(words) == 2:
            items.append((words[0], words[1]))
        elif words:
            raise SmaError(f"Plan file {plan}: '{raw.strip()}' is not 'REPO QUANT' or 'since TIME'.")
    lines = [paint("PLAN", "head") + f"  {plan}" + (f"  (runs since {since})" if since else "")]
    for repo, quant in items:
        matches = [
            row for row in runs
            if row["model_repo"] == repo and str(row["quant"]).lower() == quant.lower() and row["run_id"] >= since
        ]
        row = max(matches, key=lambda r: r["run_id"]) if matches else None
        state = "queued" if row is None else row["status"]
        if row is not None and state == "running" and current is not None and row["run_id"] != current["run_id"]:
            state = "failed"  # Marked running, but another run has the Spark: it stopped mid-step.
        lines.append(f"  {paint(_MARK.get(state, '?'), state)} {repo + ' ' + quant:44} {paint(state, state):8}  {row['run_id'] if row else ''}")
    return lines


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


def _run_lines(row: dict[str, Any], now: datetime, paint, totals: dict[str, int] | None = None) -> list[str]:
    """The run's title, then one line per benchmark: its score, its progress, or queued. Setup and
    bookkeeping steps (fetch, servers, results, teardown) get a line only while running or failed."""
    manifest, run_dir = row["manifest"], Path(row["run_dir"])
    started = _time((manifest.get("timestamps") or {}).get("started"))
    lines = [
        paint(f"CURRENT RUN  {row['run_id']}", "head")
        + f"  {row['model_repo']} {row['quant']} on {row['engine']}  {_elapsed(started, None, now)}"
    ]
    steps = {name: step for name, step in (manifest.get("steps") or {}).items() if isinstance(step, dict)}
    for name in sorted(steps, key=lambda n: STEP_NAMES.index(n) if n in STEP_NAMES else len(STEP_NAMES)):
        status = steps[name].get("status")
        if name not in _BENCHMARK_STEPS and status in ("running", "failed"):
            error = paint(str(steps[name].get("error") or "")[:100], "failed")
            lines.append(f"  {paint(_MARK[status], status)} {name:16} {status}  {error}".rstrip())
    for name, status, detail in _benchmarks(manifest, run_dir, steps, now, totals or {}):
        detail = re.sub(r"\s*\n\s*", " ", detail)  # An error's traceback would break the one-line layout.
        lines.append(f"  {paint(_MARK.get(status, '?'), status)} {name:16} {detail if status != 'queued' else paint('queued', 'queued')}")
    return lines


# The steps that measure something. Each gets a line; the others only while running or failed.
_BENCHMARK_STEPS = ("determinism", "logprobs", "quality", "kernelbench", "swebench", "browsecomp_plus", "speed")


def _benchmarks(
    manifest, run_dir: Path, steps: dict[str, dict], now: datetime, totals: dict[str, int]
) -> list[tuple[str, str, str]]:
    """(name, state, detail) for each benchmark, in the order the run does them."""

    def state(step: str) -> str:
        status = (steps.get(step) or {}).get("status") or "pending"
        return "queued" if status == "pending" else status

    def took(step: str) -> str:
        info = steps.get(step) or {}
        if info.get("status") == "failed":
            return str(info.get("error") or "failed")[:100]
        if info.get("status") == "running" and int(info.get("attempts") or 0) > 1:
            # A redone step keeps its first attempt's start and finish, so no elapsed time is shown.
            return f"attempt {info['attempts']}"
        return _elapsed(_time(info.get("started")), _time(info.get("finished")), now) if info.get("started") else ""

    rows: list[tuple[str, str, str]] = []
    passed = (_read_json(run_dir / "determinism.json") or {}).get("passed")
    rows.append(("determinism", state("determinism"), f"passed: {'yes' if passed else 'no'}" if state("determinism") == "done" else took("determinism")))
    perplexity = (_read_json(run_dir / "logprobs" / "perplexity.json") or {}).get("perplexity")
    kl = (_read_json(run_dir / "logprobs" / "kl_vs_reference.json") or {}).get("kl_mean")
    detail = " · ".join(part for part in (
        f"{perplexity:.3f}" if isinstance(perplexity, (int, float)) else "",
        f"KL {kl:.4f}" if isinstance(kl, (int, float)) else "",
    ) if part)
    rows.append(("perplexity", state("logprobs"), detail or took("logprobs")))
    for suite in _suites(manifest):
        rows.append(_suite_row(run_dir, {"questions": totals.get(suite["name"]), **{k: v for k, v in suite.items() if v is not None}}, now))
    rows += _agent_rows(manifest, run_dir, state, took)
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


def _suite_row(run_dir: Path, suite: dict[str, Any], now: datetime) -> tuple[str, str, str]:
    name = suite["name"]
    folder = run_dir / "quality" / name
    scores = _read_json(folder / "scores.json")
    if scores and (folder / ".done").is_file():
        if scores.get("primary_score") is None:
            return name, "failed", f"null  {str(scores.get('error') or '')[:90]}"
        return name, "done", f"{scores['primary_score']:.3f} {scores.get('primary_metric')}  n={scores.get('n')}"
    attempts = sorted(folder.glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0)
    if not attempts:
        return name, "queued", ""
    total = suite.get("limit") or suite.get("questions")
    limit = f"/{total:,}" if total else ""
    logs = list(attempts[-1].glob("*/inference_log.jsonl"))
    # A suite in NVIDIA's eval-factory container (BFCL) writes one JSON line per answered question into
    # a file per category. Its total lives in the container's dataset: the suite's `questions` says it.
    results = sorted(attempts[-1].glob("result/*/*_result.json")) if suite.get("image") else []
    if logs:
        answered = f"{sum(_count_lines(log) for log in logs):,}{limit or '/all'} answered"
    elif results:
        answered = f"{sum(_count_lines(path) for path in results):,}{limit} answered · {len(results)} categor{'y' if len(results) == 1 else 'ies'}"
    else:
        answered = "running"
    # command.json is written when the suite starts.
    marker = folder / "command.json" if (folder / "command.json").is_file() else attempts[-1]
    begun = datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc)
    return name, "running", f"{answered}  {_elapsed(begun, None, now)}"


def _agent_rows(manifest, run_dir: Path, state, took) -> list[tuple[str, str, str]]:
    """KernelBench, SWE-bench, and BrowseComp-Plus. KernelBench and BrowseComp-Plus are scored after speed."""
    rows = []
    for step, label in (("kernelbench", "kernelbench"), ("swebench", "swebench"), ("browsecomp_plus", "browsecomp")):
        folder = run_dir / "quality" / _AGENT_SUITES[step]
        status = state(step)
        # A step being redone keeps its last attempt's scores.json until it writes a new one.
        scores = None if status == "running" else _read_json(folder / "scores.json")
        if scores and scores.get("primary_score") is not None:
            rows.append((label, "done", f"{scores['primary_score']:.3f} {scores.get('primary_metric')}"))
            continue
        if scores and scores.get("error"):
            rows.append((label, "failed", str(scores["error"])[:90]))
            continue
        detail = took(step)
        if step == "kernelbench":
            written = sum(1 for _ in (folder / "generations").glob("*")) if (folder / "generations").is_dir() else 0
            detail = "kernels sent; scored after speed" if (folder / "job.json").is_file() else f"{written}/100 kernels written  {detail}"
        elif step == "browsecomp_plus":
            limit = ((manifest.get("config_snapshot") or {}).get("browsecomp") or {}).get("limit")
            answered = sum(1 for _ in (folder / "queries").glob("*.json")) if (folder / "queries").is_dir() else 0
            detail = f"{answered}/{limit} answered" + ("; judged after speed" if status == "done" else f"  {detail}")
        elif status == "running":
            detail = f"agents on the runner; scored at the end  {detail}"
        if status == "failed":
            detail = took(step)
        elif status == "done":
            status = "running"  # Measured, not yet scored.
        rows.append((label, status, detail))
    return rows


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


def _serving_lines(cfg: Config, view: View, paint) -> list[str]:
    """The current server's averages since it started, then each window's averages from the engines'
    own counters (engine_metrics.jsonl in each run, or sma watch's own reads for a run without one).
    One table per engine: metrics down the side, windows across."""
    now, live = view.now, view.live
    stamp = now.timestamp()
    series = recent_series(cfg.runs_path(), now=stamp, span_s=WINDOWS[-1][1])
    if live is not None and live.engine and len(live.samples) > 1:
        series.append(Series(live.engine, live.samples))
    # engine -> column label -> (rates, covered)
    tables: dict[str, dict[str, tuple[dict[str, float | None], str]]] = {}
    if live is not None and live.latest and live.engine:
        # Counters run from the server's start, so the latest read is its average since then. All-stream
        # rates divide by the container's uptime; without it they are not shown.
        uptime = (now - live.started).total_seconds() if live.started else 0.0
        since = get_backend(live.engine).serving_rates(live.latest, live.latest, uptime)
        tables.setdefault(live.engine, {})["since start"] = (since, "")
    max_gap_s = 3 * max(cfg.telemetry.engine_metrics_interval_s, live.interval_s if live else 0.0)
    for label, span in WINDOWS:
        for one in window(series, start=stamp - span, max_gap_s=max_gap_s):
            tables.setdefault(one.engine, {})[label] = (one.rates(), _elapsed(now - timedelta(seconds=one.seconds), None, now))
    if not tables:
        return [
            paint("SERVING", "head") + "  no engine counters in the last 24 hours, and no server up.",
            f"  A run records them every {cfg.telemetry.engine_metrics_interval_s:g} s while its server is up.",
        ]
    labels = ["since start"] + [label for label, _span in WINDOWS]
    lines: list[str] = []
    for engine, columns in tables.items():

        def cells(key: str) -> str:
            return "".join(f"{_number(columns[label][0].get(key)) if label in columns else '—':>12}" for label in labels)

        lines += [
            paint("SERVING", "head") + f"  {engine} · tok/s, ITL in ms",
            f"  {'':20}" + "".join(f"{label:>12}" for label in labels),
            *(f"  {name:20}{cells(key)}" for name, key in _SERVING_ROWS),
        ]
        if live is not None and live.engine == engine and len(live.samples) > 1:
            lines.append(paint("  windows: sma watch's own reads (this run records no counters)", "pending"))
    return lines


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.1f}"


def _finished_lines(runs, current, paint) -> list[str]:
    done = [row for row in runs if row["status"] in ("done", "failed") and row is not current]
    done.sort(key=lambda row: row["run_id"], reverse=True)
    lines = [paint("FINISHED", "head") + f"  latest {min(len(done), _FINISHED_SHOWN)} of {len(done)}"]
    for row in done[:_FINISHED_SHOWN]:
        summary = _read_json(Path(row["run_dir"]) / "summary.json") or {}
        parts = []
        if isinstance(summary.get("perplexity"), (int, float)):
            parts.append(f"ppl {summary['perplexity']:.2f}")
        if isinstance(summary.get("kl_mean"), (int, float)):
            parts.append(f"KL {summary['kl_mean']:.4f}")
        for suite, item in (summary.get("quality") or {}).items():
            if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
                parts.append(f"{suite} {item['score']:.2f}")
        state = row["status"]
        scores = "  " + " · ".join(parts) if parts else ""
        lines.append(f"  {paint(_MARK[state], state)} {row['model_repo']} {row['quant']}{scores}")
    return lines


def _clip(line: str, width: int) -> str:
    """The line cut to width visible characters. Color codes are kept and closed."""
    visible, out, index, open_color = 0, [], 0, False
    while index < len(line):
        code = re.match(r"\033\[[0-9;]*[A-Za-z]", line[index:])
        if code:
            out.append(code.group(0))
            open_color = code.group(0) != "\033[0m"
            index += len(code.group(0))
            continue
        if visible == width:
            return "".join(out) + ("\033[0m" if open_color else "")
        out.append(line[index])
        visible += 1
        index += 1
    return line


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


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0
