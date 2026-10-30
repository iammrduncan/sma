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
from sma.serving import WINDOWS, Series, combined_peaks, combined_rates, recent_series, window
from sma.telemetry import ENGINE_METRICS_FILE, engine_values

# Steps that send work elsewhere and report their own progress in the run folder.
_AGENT_SUITES = {"kernelbench": "kernelbench_l1", "swebench": "swebench_verified_mini", "browsecomp_plus": "browsecomp_plus"}
_MARK = {"done": "✓", "running": "▶", "failed": "✗", "pending": "·", "queued": "·", "stopped": "◼"}
_COLOR = {
    "done": "32", "running": "33", "failed": "31", "pending": "2", "queued": "2", "stopped": "33", "head": "1;36",
    # SERVING cells: the window's average in blue, its peak in purple (owner, 2026-10-30).
    "avg": "38;5;39", "max": "38;5;135",
}  # fmt: skip
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
    # SWE-bench progress read from the runner for the current run (read_swe), or None.
    swe: dict[str, Any] | None = None


def render(cfg: Config, view: View, *, plan: Path | None = None) -> str:
    """One screen of text."""
    paint = _painter(view.color)
    runs = _runs(cfg)
    current = _current_run(runs, view.containers)
    lines = _header(view, paint)
    lines += _queue_lines(cfg, runs, current, plan, paint)
    # A run's own config names its suites; a total added to sma.yaml since then still counts for it.
    totals = {suite.name: suite.questions for suite in cfg.quality.suites if suite.questions}
    swe = view.swe if current and view.swe and view.swe.get("run_id") == current["run_id"] else None
    lines += (_run_lines(current, view.now, paint, totals, swe) if current else [paint("CURRENT RUN", "head") + "  none"]) + [""]
    lines += _serving_lines(cfg, view, paint, current["run_id"] if current else None) + [""]
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
    swe_progress: Callable[[str], dict[str, Any] | None] | None = None,
    out: TextIO | None = None,
) -> None:
    """Print the screen once, or redraw it every interval_s seconds until Ctrl-C. scrape(base_url, path)
    returns the engine's /metrics text, or "" when it does not answer."""
    out = out or sys.stdout
    color = out.isatty() and not os.environ.get("NO_COLOR")
    live = Live(interval_s=interval_s)
    known: list[Machine] | None = None
    asked = 0.0
    swe: dict[str, Any] | None = None
    swe_asked = 0.0
    while True:
        up, now = containers(), datetime.now(timezone.utc)
        read_live(cfg, live, up, now, scrape, started_at)
        if machines is not None and time.monotonic() - asked >= _MACHINES_EVERY_S:
            known, asked = machines(), time.monotonic()
        # SWE-bench's progress lives on the runner: read it at most once a minute, while the step runs.
        if swe_progress is not None and live.run_id and time.monotonic() - swe_asked >= _MACHINES_EVERY_S:
            swe_asked = time.monotonic()
            swe = swe_progress(live.run_id) if _swe_running(cfg, live.run_id) else None
            if swe is not None:
                swe["run_id"] = live.run_id
        width = shutil.get_terminal_size().columns if out.isatty() else None
        view = View(up, memory(), now, color, live, width, known, None if once else interval_s, swe)
        text = render(cfg, view, plan=plan)
        if once:
            out.write(text + "\n")
            return
        out.write("\033[H\033[2J" + text + "\n")
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
        state = "running" if job["state"] == "claimed" else "queued"
        # One line per job: each leg with its mark (a pair: the reference, then the 4-bit quant with thinking
        # off and on). A leg whose run had failed steps shows as failed; its record says which.
        legs = []
        for leg in job["legs"]:
            status = {"done": "done", "running": "running", "failed": "failed", "incomplete": "failed"}.get(leg.get("status"), "queued")
            name = f"{leg.get('gguf') or leg['quant']}" + (" thinking" if leg.get("thinking") == "on" else "")
            legs.append(f"{paint(_MARK[status], status)} {name}")
        lines.append(f"  {engine_tag(job.get('engine')):4} {paint(_MARK[state], state)} {job['legs'][0]['repo']}  {'  '.join(legs)}  {paint(job['state'], state)}")
    if plan is not None:
        lines += _plan_lines(plan, runs, current, paint)
    return lines


def engine_tag(engine: str | None) -> str:
    """The short engine name a queue line starts with."""
    return {"sglang": "SGL", "llamacpp": "CPP", "vllm": "VLLM"}.get(engine or "", "?")


def _plan_lines(plan: Path, runs, current, paint) -> list[str]:
    """A plan file lists the runs a script will start, one "REPO QUANT" per line, in order; "REPO QUANT
    thinking" is the thinking-on run. A line "since TIME" (ISO time, or a run id) ignores runs created
    before it. Each line shows the newest run of that repo, quant, and thinking mode since then, or queued."""
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
    lines = [paint("PLAN", "head") + f"  {plan}" + (f"  (runs since {since})" if since else "")]
    for repo, quant, thinking in items:
        matches = [
            row for row in runs
            if row["model_repo"] == repo and str(row["quant"]).lower() == quant.lower() and row["run_id"] >= since
            and bool(_thinking(row)) == thinking
        ]
        row = max(matches, key=lambda r: r["run_id"]) if matches else None
        state = "queued" if row is None else _label(row, current)
        name = f"{repo} {quant}" + (" thinking" if thinking else "")
        lines.append(f"  {paint(_MARK.get(state, '?'), state)} {name:44} {paint(state, state):8}  {row['run_id'] if row else ''}")
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


def _run_lines(
    row: dict[str, Any], now: datetime, paint, totals: dict[str, int] | None = None, swe: dict[str, Any] | None = None
) -> list[str]:
    """The run's title, then one line per benchmark: its score, its progress, or queued. Setup and
    bookkeeping steps (fetch, servers, results, teardown) get a line only while running or failed."""
    manifest, run_dir = row["manifest"], Path(row["run_dir"])
    started = _time((manifest.get("timestamps") or {}).get("started"))
    lines = [
        paint(f"CURRENT RUN  {row['run_id']}", "head")
        + f"  {row['model_repo']} {row['quant']} on {row['engine']}, thinking {manifest.get('thinking') or 'off'}  "
        + _elapsed(started, None, now)
    ]
    steps = {name: step for name, step in (manifest.get("steps") or {}).items() if isinstance(step, dict)}
    for name in sorted(steps, key=lambda n: STEP_NAMES.index(n) if n in STEP_NAMES else len(STEP_NAMES)):
        status = steps[name].get("status")
        if name not in _BENCHMARK_STEPS and status in ("running", "failed"):
            error = paint(str(steps[name].get("error") or "")[:100], "failed")
            lines.append(f"  {paint(_MARK[status], status)} {name:16} {status}  {error}".rstrip())
    for name, status, detail in _benchmarks(manifest, run_dir, steps, now, totals or {}, paint, swe):
        detail = re.sub(r"\s*\n\s*", " ", detail)  # An error's traceback would break the one-line layout.
        lines.append(f"  {paint(_MARK.get(status, '?'), status)} {name:16} {detail if status != 'queued' else paint('queued', 'queued')}")
    return lines


# The steps that measure something. Each gets a line; the others only while running or failed.
_BENCHMARK_STEPS = ("determinism", "logprobs", "quality", "kernelbench", "swebench", "browsecomp_plus", "speed")


def _benchmarks(
    manifest, run_dir: Path, steps: dict[str, dict], now: datetime, totals: dict[str, int], paint=_identity,
    swe: dict[str, Any] | None = None,
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
    rows.append(_determinism_row(run_dir, state("determinism"), took("determinism")))
    perplexity = (_read_json(run_dir / "logprobs" / "perplexity.json") or {}).get("perplexity")
    kl = (_read_json(run_dir / "logprobs" / "kl_vs_reference.json") or {}).get("kl_mean")
    detail = " · ".join(part for part in (
        f"{perplexity:.3f}" if isinstance(perplexity, (int, float)) else "",
        f"KL {kl:.4f}" if isinstance(kl, (int, float)) else "",
    ) if part)
    rows.append(("perplexity", state("logprobs"), detail or took("logprobs")))
    for suite in _suites(manifest):
        rows.append(_suite_row(run_dir, {"questions": totals.get(suite["name"]), **{k: v for k, v in suite.items() if v is not None}}, now, paint))
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
    # LiveCodeBench's runner keeps its answers in cache/<file>.json, one key per problem, saved every 10.
    cache = sorted(attempts[-1].glob("cache/*.json")) if suite.get("image") else []
    if logs:
        answered = f"{sum(_count_lines(log) for log in logs):,}{limit or '/all'} answered"
    elif cache and isinstance(_read_json(cache[-1]), dict):
        answered = f"{len(_read_json(cache[-1])):,}{limit} answered"
    elif results:
        answered = f"{sum(_count_lines(path) for path in results):,}{limit} answered · {len(results)} categor{'y' if len(results) == 1 else 'ies'}"
    else:
        answered = "running"
    # command.json is written when the suite starts.
    marker = folder / "command.json" if (folder / "command.json").is_file() else attempts[-1]
    begun = datetime.fromtimestamp(marker.stat().st_mtime, timezone.utc)
    return name, "running", f"{answered}  {_elapsed(begun, None, now)}"


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

    def cell(label: str, key: str) -> str:
        """average/peak, right-aligned in _CELL characters: the average blue, the peak purple."""
        averages, peaks = columns.get(label, ({}, {}))
        mean, peak = averages.get(key), peaks.get(key)
        if mean is None and peak is None:
            return f"{'—':>{_CELL}}"
        text = f"{_number(mean)}/{_number(peak)}"
        return " " * max(0, _CELL - len(text)) + paint(_number(mean), "avg") + "/" + paint(_number(peak), "max")

    def cells(key: str) -> str:
        return "".join(cell(label, key) for label in labels)

    lines = [
        paint("SERVING", "head") + "  tok/s, ITL in ms, average/peak: this run, then every run of the last 24 hours",
        f"  {'':20}" + "".join(f"{label:>{_CELL}}" for label in labels),
        *(f"  {name:20}{cells(key)}" for name, key in _SERVING_ROWS),
    ]
    if own_reads:
        lines.append(paint("  this run's windows are sma watch's own reads (the run records no counters)", "pending"))
    return lines


# The last column: every run of the last 24 hours, whatever engine served.
_ALL_RUNS = "24 h, all"
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


def _finished_lines(runs, current, paint) -> list[str]:
    done = [row for row in runs if row is not current and _label(row, current) in ("done", "failed", "stopped")]
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
                parts.append(f"{suite} {_percent(item['score'], paint)}")
        state = _label(row, current)
        scores = "  " + " · ".join(parts) if parts else ""
        note = paint(" (stopped, resumable)", "stopped") if state == "stopped" else ""
        lines.append(f"  {paint(_MARK[state], state)} {row['model_repo']} {row['quant']}{_thinking(row)}{note}{scores}")
    return lines


def _thinking(row: dict[str, Any]) -> str:
    """" thinking on" for a thinking-on run; nothing for the default (off, or a run from before the setting)."""
    return " thinking on" if (row.get("manifest") or {}).get("thinking") == "on" else ""


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
