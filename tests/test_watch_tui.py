"""`sma watch` as a TUI: the model (alerts, panels, detail views), the pure layout at a few terminal sizes, the
keys, and the curses loop on a fake screen."""

from __future__ import annotations

import curses
import io
import json
import os
import re
import time
from pathlib import Path

import pytest

from sma.catalog import Catalog
from sma.queue import JobQueue
from sma.records import STEP_NAMES
from sma.tui import Board, Frame, Item, Panel, Ui, attributes, clip, handle_key, layout, loop, segments, visible, wrap
from sma.watch import View, board, render, watch
from tests.harness import make_config
from tests.test_watch import NOW, _run

ANSI = re.compile(r"\033\[[0-9;]*m")
# A pid no process has: os.kill(pid, 0) says there is no such process.
DEAD_PID = 2**31 - 1


def _plain(lines: list[str]) -> list[str]:
    return [ANSI.sub("", line) for line in lines]


def _set_manifest(run_dir: Path, change) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    change(manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def _dispatcher(cfg, slots: list[dict], **extra) -> None:
    folder = cfg.queue_path() / "dispatcher"
    folder.mkdir(parents=True, exist_ok=True)
    state = {"pid": os.getpid(), "updated": "2026-10-29T02:59:58Z", "slots": slots, "memory_guard": None, "stopping": False, **extra}
    (folder / f"{cfg.host_alias}.json").write_text(json.dumps(state))


def _slots_fixture(cfg, **extra) -> tuple[str, str]:
    ref = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "F16", steps={"fetch": "done", "quality": "running"})
    cand = _run(cfg, "20261029T020540Z-bbbb", "LiquidAI/LFM3-1.7B-GGUF", "QAD-Q4_0", steps={"fetch": "done", "logprobs": "running"})
    _dispatcher(cfg, [
        {"name": "v100-0", "gpus": "0", "state": "busy", "run_id": ref.name, "repo": "LiquidAI/LFM3-1.7B-GGUF", "gguf": "F16", "thinking": "off"},
        {"name": "v100-1", "gpus": "1", "state": "busy", "run_id": cand.name, "repo": "LiquidAI/LFM3-1.7B-GGUF", "gguf": "QAD-Q4_0", "thinking": "off"},
        {"name": "v100-2", "gpus": "2", "state": "idle"},
        {"name": "v100-3", "gpus": "3", "state": "down", "reason": "its server failed to start 2 times in a row"},
    ], **extra)  # fmt: skip
    Catalog(cfg.runs_path()).rebuild()
    return ref.name, cand.name


def _synthetic_board() -> Board:
    """A busy screen: four slots of four lines, 30 queued jobs, 40 finished runs."""
    slots = [Item([f"  ▶ v100-{n}   GPU {n}  Org/Model-{n} F16  1h0{n}m", "      ▶ quality  3/5 answered", "      ✗ determinism failed",
                   "      ✓ perplexity 24.0 · ifeval 80.0%"], ("run", f"run-{n}")) for n in range(4)]  # fmt: skip
    return Board(
        summary="SMA watch  2026-10-29 03:00:00  ✗ 1 problem · 4 of 4 slots busy · queue 30 jobs",
        alerts=["✗ slot v100-3 is down: its server failed to start"],
        panels=[
            Panel("machines", "MACHINES", notes=[f"  ● {role}" for role in ("model", "kernel", "swe", "judge")], compact="MACHINES  ● model  ● kernel"),
            Panel("queue", "QUEUE  30 jobs in sma queue", [Item([f"  CPP  · Org/Queued-{n}  · F16  pending"], ("job", f"job-{n}")) for n in range(30)]),
            Panel("active", "SLOTS  4 of 4 busy on the serving machine", slots),
            Panel("kernelbench", "KERNELBENCH  1 timing, 0 queued", [Item(["  ▶ Org/Model-0 F16  41/100 kernels"], ("run", "run-0"))]),
            Panel("serving", "SERVING  tok/s", notes=["  " + "x" * 90] * 6),
            Panel("finished", "FINISHED  40 runs, newest first", [Item([f"  ✓ Org/Done-{n} F16  " + "score " * 40], ("run", f"done-{n}")) for n in range(40)],
                  once_items=3),  # fmt: skip
        ],
    )


def test_layout_fills_120x40_in_one_column_with_the_focused_panel_and_selection_marked() -> None:
    frame = layout(_synthetic_board(), Ui(), 120, 40)
    lines = _plain(frame.lines)

    assert len(lines) == 40 and max(len(line) for line in lines) <= 120
    assert lines[0].startswith("SMA watch") and lines[1].startswith("✗ slot v100-3 is down")
    assert lines[-1].startswith("Tab panel · ↑↓ select · Enter open")
    # Every slot fits; the queue and finished runs that do not are counted on their title line.
    assert all(f"v100-{n}" in "\n".join(lines) for n in range(4))
    queue_title = next(line for line in lines if line.startswith("QUEUE"))
    assert re.search(r"… \d+ below", queue_title)
    assert "│" not in "\n".join(lines)
    # The focused panel's name and its selected item's first line are drawn reversed.
    active_row = next(row for row, line in enumerate(lines) if line.startswith("SLOTS"))
    assert (active_row, 0, len("SLOTS")) in frame.highlight and (active_row + 1, 0, 120) in frame.highlight


def test_layout_uses_two_columns_at_200x60_and_marks_the_selection_in_the_right_one() -> None:
    ui = Ui()
    frame = layout(_synthetic_board(), ui, 200, 60)
    lines = _plain(frame.lines)

    assert len(lines) == 60 and max(len(line) for line in lines) <= 200
    body = lines[2:-1]
    # Left: the running panel and finished runs (wide lines). Right: machines, queue, KernelBench, serving.
    assert body[0].startswith("SLOTS") and "│ MACHINES" in body[0]
    left_width = 200 - 94 - 3
    assert all(line[left_width + 1] == "│" for line in body if len(line) > left_width + 1)
    assert any(line.startswith("FINISHED") for line in body)
    handle_key(ui, "tab", _synthetic_board(), frame)
    assert ui.focus == "queue"
    frame = layout(_synthetic_board(), ui, 200, 60)
    queue_row = next(row for row, line in enumerate(_plain(frame.lines)) if "│ QUEUE" in line)
    assert (queue_row, left_width + 3, left_width + 3 + len("QUEUE")) in frame.highlight
    assert (queue_row + 1, left_width + 3, left_width + 3 + 94) in frame.highlight


def test_a_small_terminal_folds_the_machines_and_still_shows_the_running_panel() -> None:
    frame = layout(_synthetic_board(), Ui(), 80, 16)
    lines = _plain(frame.lines)

    assert len(lines) == 16 and max(len(line) for line in lines) <= 80
    assert "MACHINES  ● model  ● kernel" in lines or not any(line.startswith("MACHINES") for line in lines)
    assert any(line.startswith("SLOTS") for line in lines) and any("v100-0" in line for line in lines)
    # A cut line ends with an ellipsis.
    assert any(line.endswith("…") for line in lines)


def test_keys_move_focus_and_selection_open_an_item_and_go_back() -> None:
    screen = _synthetic_board()
    ui = Ui()
    frame = layout(screen, ui, 120, 40)

    assert handle_key(ui, "tab", screen, frame) is None and ui.focus == "queue"
    handle_key(ui, "btab", screen, frame)
    assert ui.focus == "active"
    for _ in range(9):
        handle_key(ui, "down", screen, frame)
    assert ui.selected["active"] == 3  # Stops at the last slot.
    handle_key(ui, "home", screen, frame)
    assert ui.selected["active"] == 0
    handle_key(ui, "4", screen, frame)
    assert ui.focus == "finished"
    handle_key(ui, "end", screen, frame)
    assert ui.selected["finished"] == 39
    # Enter opens the selected item and asks for a read, so the detail comes with it.
    assert handle_key(ui, "enter", screen, frame) == "refresh" and ui.detail == ("run", "done-39")
    screen.detail = [f"line {n}" for n in range(100)]
    frame = layout(screen, ui, 120, 40)
    assert frame.detail_rows == 100 and _plain(frame.lines)[2] == "line 0"  # Under the top line and the alert.
    handle_key(ui, "pgdn", screen, frame)
    assert ui.scroll == frame.page
    handle_key(ui, "end", screen, frame)
    assert ui.scroll == 100 - frame.page
    handle_key(ui, "down", screen, frame)
    assert ui.scroll == 100 - frame.page  # Not past the end.
    assert _plain(layout(screen, ui, 120, 40).lines)[-2] == "line 99"
    handle_key(ui, "esc", screen, frame)
    assert ui.detail is None and ui.scroll == 0
    assert handle_key(ui, "?", screen, frame) is None and ui.help
    assert "KEYS" in _plain(layout(screen, ui, 120, 40).lines)
    handle_key(ui, "x", screen, frame)
    assert not ui.help
    assert handle_key(ui, "q", screen, frame) == "quit"


def test_enter_on_an_idle_slot_says_there_is_nothing_to_open_and_a_missing_panel_cannot_be_focused(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    _slots_fixture(cfg)
    screen = board(cfg, View([], None, NOW))
    ui = Ui()
    frame = layout(screen, ui, 120, 40)
    ui.selected["active"] = 2  # v100-2, idle.
    assert handle_key(ui, "enter", screen, frame) is None and ui.detail is None
    assert "Nothing to open here" in _plain(layout(screen, ui, 120, 40).lines)[-1]
    handle_key(ui, "3", screen, frame)  # No KernelBench panel: the service is not set.
    assert ui.focus == "active" and "nothing to show" in ui.message


def test_the_selected_slots_run_shows_whole_beneath_the_slots_and_follows_the_selection(tmp_path: Path) -> None:
    # Owner, 2026-11-01: a run is one list again, as CURRENT RUN was; the slots are one line each above it.
    cfg = make_config(tmp_path)
    _slots_fixture(cfg)
    screen = board(cfg, View([], None, NOW))
    ui = Ui()

    def body() -> list[str]:
        return _plain(layout(screen, ui, 120, 40).lines)

    lines = body()
    slots_row = next(row for row, line in enumerate(lines) if line.startswith("SLOTS"))
    assert all(lines[slots_row + 1 + n].startswith(f"  {mark} v100-{n}") for n, mark in enumerate("▶▶·✗"))
    window = lines[slots_row + 5 :]
    assert window[0].startswith("v100-0  LiquidAI/LFM3-1.7B-GGUF F16 on llamacpp")
    names = [line.split()[1] for line in window[1:] if line.startswith("  ") and line.split()[0] in "✓▶✗·◼"]
    assert names[:2] == ["determinism", "perplexity"] and "speed" in names and "fetch" not in names
    frame = layout(screen, ui, 120, 40)
    handle_key(ui, "down", screen, frame)
    assert body()[slots_row + 5].startswith("v100-1  LiquidAI/LFM3-1.7B-GGUF QAD-Q4_0")
    # The window stays while another panel has the focus.
    handle_key(ui, "tab", screen, frame)
    assert ui.focus == "queue" and any(line.startswith("v100-1  LiquidAI") for line in body())
    # An idle slot has no run to show.
    ui.focus, ui.selected["active"] = "active", 2
    assert not any(line.startswith("v100-") for line in body())
    # --once prints every busy slot's run.
    once = ANSI.sub("", render(cfg, View([], None, NOW)))
    assert "\nv100-0  LiquidAI/LFM3-1.7B-GGUF F16" in once and "\nv100-1  LiquidAI/LFM3-1.7B-GGUF QAD-Q4_0" in once


def test_a_runs_detail_is_one_list_with_whole_errors_and_a_stopped_runs_steps_stopped(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    error = "The model server did not answer within 600 s. " + "Its log ends with a CUDA error. " * 6
    row = _run(cfg, "20261029T020534Z-aaaa", "Org/Model", "F16", steps={"fetch": "done", "serve_quality": "failed", "logprobs": "running"})
    _set_manifest(row, lambda manifest: manifest["steps"]["serve_quality"].update(error=error))
    Catalog(cfg.runs_path()).rebuild()

    detail = _plain(board(cfg, View([], None, NOW), open_item=("run", row.name)).detail or [])

    assert not any(line.startswith(("STEPS", "BENCHMARKS")) for line in detail)
    assert any(line.startswith("  ✗ serve_quality") and line.endswith(error.strip()) for line in detail)
    assert "  ◼ perplexity       stopped" in detail  # Its record says running; no process runs it.
    assert not any(line.startswith("  ✓ fetch") for line in detail)


def test_the_filter_hides_items_that_do_not_contain_it_and_esc_clears_it() -> None:
    screen = _synthetic_board()
    ui = Ui(focus="queue")
    frame = layout(screen, ui, 120, 40)
    for key in ["/", "q", "u", "e", "u", "e", "d", "-", "1", "2", "enter"]:
        handle_key(ui, key, screen, frame)  # "q" while typing is text, not quit.
    assert ui.filter == "queued-12" and not ui.typing
    queue = next(panel for panel in screen.panels if panel.name == "queue")
    assert [item.key for item in visible(queue, ui)] == [("job", "job-12")]
    lines = _plain(layout(screen, ui, 120, 40).lines)
    assert "(29 hidden by the filter)" in "\n".join(lines) and "filter: queued-12" in lines[-1]
    handle_key(ui, "esc", screen, frame)
    assert ui.filter == ""


def test_the_detail_view_wraps_long_lines_with_their_colors() -> None:
    line = "      " + "\033[31m" + "error " * 30 + "\033[0m"
    rows = wrap(line, 60)
    assert len(rows) == 4 and all(len(ANSI.sub("", row)) <= 60 for row in rows)
    assert rows[1].startswith(" " * 8 + "\033[31m") and rows[0].endswith("\033[0m")
    assert clip("\033[32mabcdef\033[0m", 4) == "\033[32mabc…\033[0m"


def test_the_board_for_slots_flags_a_dead_dispatcher_a_down_slot_and_a_memory_guard_stop(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    guard = {"at": "2026-10-29T02:50:00Z", "slot": "v100-0", "run_id": "20261029T020534Z-aaaa", "source": "meminfo"}
    ref, _cand = _slots_fixture(cfg, pid=DEAD_PID, memory_guard=guard)
    screen = board(cfg, View([], (90 * 2**30, 121 * 2**30), NOW, color=True))
    alerts = _plain(screen.alerts)

    assert alerts[0].startswith("✗ the dispatcher is not running")
    assert "✗ slot v100-3 is down: its server failed to start 2 times in a row" in alerts
    assert "! the memory guard stopped LiquidAI/LFM3-1.7B-GGUF F16 on v100-0 10m00s ago (meminfo); it resumes later" in alerts
    # Problems are red and warnings yellow, in the alerts and the top line's count.
    assert screen.alerts[0].startswith("\033[31m✗") and any(line.startswith("\033[33m!") for line in screen.alerts)
    assert "✗ 2 problems, 1 warning" in ANSI.sub("", screen.summary) and "2 of 4 slots busy, 1 down" in screen.summary
    active = next(panel for panel in screen.panels if panel.name == "active")
    assert "the dispatcher is not running" in ANSI.sub("", active.title)
    assert [item.key for item in active.items] == [("run", ref), ("run", "20261029T020540Z-bbbb"), None, None]
    # Several runs at once: the serving panel has a row per slot's run, then every run of the day.
    serving = next(panel for panel in screen.panels if panel.name == "serving")
    labels = [line.split("  ")[1] for line in _plain(serving.notes)[1:]]
    assert labels == ["v100-0 F16", "v100-1 QAD-Q4_0", "all runs, last 24 hours"]


def test_a_silent_dispatcher_is_a_warning(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    _slots_fixture(cfg, updated="2026-10-29T02:50:00Z")
    alerts = _plain(board(cfg, View([], None, NOW)).alerts)
    assert "! the dispatcher has not written its state for 10m00s (it writes every 5 s)" in alerts


def test_a_step_far_past_its_usual_time_and_a_silent_run_are_flagged_in_place_and_on_top(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    for n in range(3):
        done = _run(cfg, f"20261028T0{n}0000Z-old{n}", "Org/Model", "F16", steps={name: "done" for name in STEP_NAMES}, finished=True)

        def took_ten_minutes(manifest):
            for step in manifest["steps"].values():
                step.update(started="2026-10-28T01:00:00Z", finished="2026-10-28T01:10:00Z", attempts=1)

        _set_manifest(done, took_ten_minutes)
    running = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "F16", steps={"fetch": "done", "quality": "running"})
    _set_manifest(running, lambda manifest: manifest["steps"]["quality"].update(started="2026-10-29T02:00:00Z", attempts=1))
    Catalog(cfg.runs_path()).rebuild()
    up = ["sma-20261029T020534Z-aaaa-quality"]

    screen = board(cfg, View(up, None, NOW))
    assert "! LiquidAI/LFM3-1.7B-GGUF F16: quality has run 1h00m, usually 10m00s" in _plain(screen.alerts)
    active = next(panel for panel in screen.panels if panel.name == "active")
    assert "! quality has run 1h00m, usually 10m00s" in ANSI.sub("", active.title)
    detail = _plain(board(cfg, View(up, None, NOW), open_item=("run", running.name)).detail or [])
    assert detail[0].startswith("RUN  20261029T020534Z-aaaa") and "! quality has run 1h00m, usually 10m00s" in detail[0]

    # Nothing written to its folder for 30 minutes: the silence is the warning, not the late step.
    os.utime(running / "manifest.json", (NOW.timestamp() - 1800,) * 2)
    alerts = _plain(board(cfg, View(up, None, NOW)).alerts)
    assert alerts == ["! LiquidAI/LFM3-1.7B-GGUF F16: nothing written to its run folder for 30m00s; is its process still running?"]


def test_a_stale_queue_claim_and_a_leg_failed_for_good_are_alerts(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    queue = JobQueue(cfg.queue_path())
    job = queue.add([{"repo": "Org/Model-GGUF", "quant": "gguf", "gguf": g, "engine": "llamacpp"} for g in ("F16", "Q4_0", "Q4_0")], kind="pair")
    queue.claim_job(job["job_id"], "worker-1", "gpu-host-1")

    def stale(j):
        j["heartbeat"] = "2026-10-29T01:00:00Z"
        j["legs"][2].update(status="failed", dead_lettered=True, error="The memory guard stopped this leg twice.")

    queue.update(job["job_id"], stale)
    screen = board(cfg, View([], None, NOW))
    alerts = _plain(screen.alerts)

    assert any(alert.startswith("! Org/Model-GGUF: the queue claim's heartbeat is 2h00m old") for alert in alerts)
    assert "✗ Org/Model-GGUF Q4_0 failed for good: The memory guard stopped this leg twice." in alerts
    # The job's detail: its claim, heartbeat, and each leg.
    detail = _plain(board(cfg, View([], None, NOW), open_item=("job", job["job_id"])).detail or [])
    assert "  claimed by worker-1 on gpu-host-1" in detail[1] and "heartbeat 2h00m ago" in detail[1]
    assert any("3. Org/Model-GGUF Q4_0, thinking off  failed" in line for line in detail)
    assert any("failed for good" in line for line in detail)


def test_several_running_runs_without_a_dispatcher_get_a_running_panel(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    _run(cfg, "20261029T020534Z-aaaa", "Org/One", "F16", steps={"quality": "running"})
    _run(cfg, "20261029T020600Z-bbbb", "Org/Two", "F16", steps={"logprobs": "running"})
    # A record a crash left running, silent for hours: not a third run at work.
    dead = _run(cfg, "20261028T010000Z-dead", "Org/Dead", "F16", steps={"serve_quality": "running"})
    os.utime(dead / "manifest.json", (NOW.timestamp() - 7200,) * 2)
    Catalog(cfg.runs_path()).rebuild()

    screen = render(cfg, View([], None, NOW))

    assert "RUNNING  2 runs" in screen and "CURRENT RUN" not in screen and "· 2 runs ·" in screen
    running = screen.split("RUNNING")[1].split("SERVING")[0]
    assert "▶ Org/One F16 on llamacpp" in running and "▶ Org/Two F16 on llamacpp" in running and "Dead" not in running
    assert "◼ Org/Dead F16 (stopped, resumable)" in screen.split("FINISHED")[1]


def _took(run_dir: Path, step: str, start: str, end: str) -> None:
    _set_manifest(run_dir, lambda manifest: manifest["steps"][step].update(status="done", started=start, finished=end, attempts=1))


def test_plain_watch_shows_every_slot_this_sparks_runs_and_every_finished_run(tmp_path: Path) -> None:
    # Owner, 2026-11-02: plain `sma watch` shows all runs, across every config; --config narrows it to one. It had shown
    # two of four V100 runs (the two whose BFCL grader container ran here), each "quality has run 4h49m, usually 19s".
    cfg = make_config(tmp_path)
    for n in range(3):
        _took(_run(cfg, f"20261028T0{n}0000Z-old{n}", "Org/Spark", "F16", steps={"quality": "done"}, finished=True),
              "quality", "2026-10-28T01:00:00Z", "2026-10-28T01:00:19Z")  # fmt: skip
    old = _run(cfg, "20261028T040000Z-v1d0", "Org/Volta", "F16", steps={"quality": "done"}, finished=True)
    _set_manifest(old, lambda manifest: manifest.update(serving={"hardware": "v100", "slot": "v100-0"}))
    slotted = []
    for n in range(2):
        run = _run(cfg, f"20261029T02000{n}Z-v10{n}", "Org/Volta", f"Q{n}", steps={"quality": "running"})
        _set_manifest(run, lambda manifest, n=n: manifest.update(serving={"hardware": "v100", "slot": f"v100-{n}"}))
        slotted.append(run.name)
    spark = _run(cfg, "20261029T020100Z-sprk", "Org/Spark", "BF16", steps={"quality": "running"})
    _dispatcher(cfg, [{"name": f"v100-{n}", "gpus": str(n), "state": "busy", "run_id": run_id, "repo": "Org/Volta", "gguf": f"Q{n}"}
                      for n, run_id in enumerate(slotted)])  # fmt: skip
    Catalog(cfg.runs_path()).rebuild()
    # A V100 run's grader container is up on this Spark: it does not make that run this Spark's.
    up = [f"sma-{slotted[0]}-bfcl"]

    screen = ANSI.sub("", render(cfg, View(up, None, NOW), scoped=False))

    assert "· 2 of 2 slots busy, 1 run on this Spark ·" in screen.splitlines()[0]
    slots = screen.split("SLOTS")[1].split("QUEUE")[0]
    assert slots.startswith("  2 of 2 busy on the serving machine, 1 run on this Spark")
    assert "▶ v100-0   GPU 0  Volta Q0" in slots and "▶ v100-1   GPU 1  Volta Q1" in slots and "▶ spark    GPU 0  Spark BF16" in slots
    # Each step's usual time is its own hardware's: the Spark's run is late by the Spark's 19 s, the V100 runs are not.
    assert "! spark Org/Spark BF16: quality has run 50m00s, usually 19s" in screen
    assert "Org/Volta Q0: quality" not in screen and "Org/Volta Q1: quality" not in screen
    finished = screen.split("FINISHED")[1]
    assert finished.startswith("  4 runs, newest first") and "Org/Volta F16 on v100" in finished and "Org/Spark F16 on" not in finished

    # --config narrows it: a Spark's config shows this Spark's run alone and says where the rest run.
    queue = JobQueue(cfg.queue_path())
    job = queue.add([{"repo": "Org/Volta", "quant": "gguf", "gguf": f"Q{n}", "engine": "llamacpp"} for n in range(2)], kind="pair")
    queue.claim_job(job["job_id"], "dispatcher-v100-pod", "v100-pod")
    queue.update(job["job_id"], lambda current: [leg.update(status="running", slot=f"v100-{n}") for n, leg in enumerate(current["legs"])])
    (cfg.queue_path() / "dispatcher" / f"{cfg.host_alias}.json").rename(cfg.queue_path() / "dispatcher" / "v100-pod.json")
    screen = ANSI.sub("", render(cfg, View(up, None, NOW)))
    assert "CURRENT RUN  20261029T020100Z-sprk" in screen and "SLOTS" not in screen
    assert "  2 runs on the serving machine's GPU slots; plain sma watch shows every run" in screen
    assert "· 1 run on this Spark, 2 on the serving machine ·" in screen.splitlines()[0]
    assert screen.split("FINISHED")[1].startswith("  3 runs on this Spark, newest first")


def test_the_spark_view_counts_only_this_sparks_runs_and_says_where_the_rest_run(tmp_path: Path) -> None:
    # Owner, 2026-11-02: with four runs on the V100 slots, `sma watch` showed two of them as this Spark's runs
    # (the two whose BFCL grader container ran here), each flagged "quality has run 4h49m, usually 19s".
    cfg = make_config(tmp_path)
    queue = JobQueue(cfg.queue_path())
    job = queue.add([{"repo": "Org/Model-GGUF", "quant": "gguf", "gguf": g, "engine": "llamacpp"} for g in ("F16", "Q4_0")], kind="pair")
    queue.claim_job(job["job_id"], "dispatcher-v100-pod", "v100-pod")
    runs = []
    for slot, (quant, run_id) in enumerate((("F16", "20261029T020534Z-aaaa"), ("Q4_0", "20261029T020600Z-bbbb"))):
        run = _run(cfg, run_id, "Org/Model-GGUF", quant, steps={"quality": "running"})
        _set_manifest(run, lambda manifest, slot=slot: manifest.update(serving={"hardware": "v100", "slot": f"v100-{slot}"}))
        runs.append(run)

    def start_legs(current: dict) -> None:
        for slot, leg in enumerate(current["legs"]):
            leg.update(status="running", slot=f"v100-{slot}")

    queue.update(job["job_id"], start_legs)
    Catalog(cfg.runs_path()).rebuild()

    # The F16 run's grader container is up on this Spark; the Q4_0 run has none here.
    screen = ANSI.sub("", render(cfg, View([f"sma-{runs[0].name}-bfcl"], None, NOW)))

    assert "CURRENT RUN  none" in screen and "RUNNING" not in screen
    assert "  2 runs on the serving machine's GPU slots; plain sma watch shows every run" in screen
    assert "· no run on this Spark, 2 on the serving machine ·" in screen.splitlines()[0]
    assert "Org/Model-GGUF F16 on llamacpp" not in screen.split("QUEUE")[0]


def test_once_prints_the_main_screen_with_three_finished_runs_and_counts_the_rest(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    for n in range(5):
        _run(cfg, f"20261028T0{n}0000Z-done", f"Org/Done-{n}", "F16", steps={name: "done" for name in STEP_NAMES}, finished=True)
    Catalog(cfg.runs_path()).rebuild()

    lines = render(cfg, View([], None, NOW)).splitlines()

    assert lines[0].startswith("SMA watch") and "✓ healthy · no run · queue 0 jobs" in lines[0]
    finished = lines[lines.index("FINISHED  5 runs on this Spark, newest first") + 1 :]
    assert finished[:4] == ["  ✓ Org/Done-4 F16", "  ✓ Org/Done-3 F16", "  ✓ Org/Done-2 F16", "  … 2 more"]


def test_piped_watch_prints_one_plain_screen_per_interval(tmp_path: Path, monkeypatch) -> None:
    cfg = make_config(tmp_path)
    Catalog(cfg.runs_path()).rebuild()
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("sma.watch.time.sleep", sleep)
    out = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        watch(cfg, plan=None, interval_s=7, once=False, containers=list, memory=lambda: None, scrape=lambda url, path: "", out=out)

    text = out.getvalue()
    assert text.count("SMA watch") == 2 and "\033[" not in text and sleeps == [7, 7]
    assert "redraws every 7 s, Ctrl-C quits" in text


class FakeScreen:
    """A curses window that records what is drawn and hands out scripted keys."""

    def __init__(self, keys: list, size: tuple[int, int] = (30, 100)) -> None:
        self.keys, self.size = keys, size
        self.rows: dict[int, str] = {}
        self.reversed: list[tuple[int, int, int]] = []
        self.frames = 0
        self.drawn: list[str] = []

    def getmaxyx(self) -> tuple[int, int]:
        return self.size

    def erase(self) -> None:
        self.rows, self.reversed = {}, []

    def addstr(self, row: int, column: int, text: str, attr: int) -> None:
        line = self.rows.get(row, "")
        self.rows[row] = line.ljust(column) + text

    def chgat(self, row: int, column: int, count: int, attr: int) -> None:
        self.reversed.append((row, column, count))

    def refresh(self) -> None:
        self.frames += 1
        self.drawn.append("\n".join(self.rows.get(row, "") for row in range(self.size[0])))

    def keypad(self, flag: bool) -> None:
        pass

    def timeout(self, ms: int) -> None:
        pass

    def get_wch(self):
        key = self.keys.pop(0)
        if key is None:
            time.sleep(0.05)  # The screen's timeout: no key came, and the reader's thread gets time to finish.
            raise curses.error("no input")
        return key


def test_the_curses_loop_draws_frames_and_follows_keys_on_a_fake_screen() -> None:
    reads: list = []

    def read(open_item):
        reads.append(open_item)
        screen = _synthetic_board()
        screen.detail = ["DETAIL of " + str(open_item)] if open_item else None
        return screen

    screen = FakeScreen([None, "\t", curses.KEY_DOWN, "\n", None, None, "\x1b", None, "q"])
    loop(curses, screen, read, 60.0)

    # A read at start, then one when Enter opened the queue's second job; the detail was drawn, then left.
    assert reads == [None, ("job", "job-1")]
    assert any("DETAIL of ('job', 'job-1')" in text for text in screen.drawn)
    assert screen.drawn[-1].startswith("SMA watch") and "QUEUE  30 jobs" in screen.drawn[-1]
    # The focused panel's name and its selection are drawn reversed; the top line never is.
    assert screen.reversed and all(row != 0 for row, _column, _count in screen.reversed)


def test_a_read_that_fails_ends_the_tui_with_its_error() -> None:
    from sma.errors import SmaError

    def read(open_item):
        if open_item:
            raise SmaError("Plan file plan.txt: 'x' is not 'REPO QUANT'.")
        return _synthetic_board()

    screen = FakeScreen([None, "4", "\n", None, None, "q"])
    with pytest.raises(SmaError, match="is not 'REPO QUANT'"):
        loop(curses, screen, read, 60.0)


def test_segments_and_attributes_map_the_painters_colors_to_curses() -> None:
    assert segments("\033[32m✓\033[0m done \033[38;5;196m1.0%\033[0m") == [("✓", "32"), (" done ", ""), ("1.0%", "38;5;196")]
    assert attributes(curses, "1;36", None) == curses.A_BOLD
    assert attributes(curses, "2", None) == curses.A_DIM
    assert attributes(curses, "32", lambda code: 1 << 20 if code == "32" else 0) == curses.A_NORMAL | 1 << 20
    from sma.tui import _foreground

    assert _foreground("38;5;196", 256) == 196 and _foreground("38;5;196", 8) == 1 and _foreground("1;36", 8) == 6
    assert _foreground("2", 8) is None


def test_the_tui_screen_is_scrubbed_like_a_run_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sma.scrub.host_names", lambda: ["sparkbox"])
    cfg = make_config(tmp_path)
    run_dir = _run(cfg, "20261029T020534Z-aaaa", "LiquidAI/LFM3-1.7B-GGUF", "BF16", steps={"serve_quality": "failed"})
    _set_manifest(run_dir, lambda manifest: manifest["steps"]["serve_quality"].update(error="sparkbox ran out of memory"))
    Catalog(cfg.runs_path()).rebuild()
    up = ["sma-20261029T020534Z-aaaa-quality"]

    screen = board(cfg, View(up, None, NOW, color=True), open_item=("run", "20261029T020534Z-aaaa"))
    text = "\n".join(layout(screen, Ui(), 120, 40).lines + (screen.detail or []) + screen.alerts)

    assert "sparkbox" not in text and "<host> ran out of memory" in ANSI.sub("", text)


def test_frame_page_and_detail_rows_bound_scrolling() -> None:
    ui = Ui(detail=("run", "x"))
    screen = Board("top", [], [], detail=["a"] * 5)
    frame = layout(screen, ui, 40, 20)
    assert isinstance(frame, Frame) and frame.page == 18 and frame.detail_rows == 5
    handle_key(ui, "pgdn", screen, frame)
    assert ui.scroll == 0  # Everything fits: nothing to scroll.


def test_a_record_a_crash_left_running_is_not_a_run_unless_a_process_holds_the_gpu_lock(tmp_path: Path) -> None:
    # Owner, 2026-11-01: a VL run that died on 2026-10-28 showed as CURRENT RUN, "running", four days later.
    import fcntl

    from sma.locks import lock_held

    cfg = make_config(tmp_path)
    dead = _run(cfg, "20261025T010000Z-dead", "Org/Dead", "F16", steps={"fetch": "done", "serve_quality": "running"})
    Catalog(cfg.runs_path()).rebuild()
    os.utime(dead / "manifest.json", (NOW.timestamp() - 4 * 86400,) * 2)
    lock = Path(cfg.lock_path)
    assert lock_held(lock) is None  # No lock file: not known, so the old rule holds.
    lock.touch()
    assert lock_held(lock) is False

    screen = ANSI.sub("", render(cfg, View([], None, NOW)))
    assert "CURRENT RUN  none" in screen and "· no run ·" in screen.splitlines()[0] and "nothing written" not in screen
    assert "◼ Org/Dead F16 (stopped, resumable)" in screen.split("FINISHED")[1]

    # While a process holds the GPU lock, the newest such record is the run at work, and its silence a warning.
    with lock.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert lock_held(lock) is True
        screen = ANSI.sub("", render(cfg, View([], None, NOW)))
    assert "CURRENT RUN  20261025T010000Z-dead" in screen and "nothing written to its run folder" in screen


def test_slots_in_the_config_show_before_any_dispatcher_ran_with_the_queue_waiting_and_only_their_runs_finished(tmp_path: Path) -> None:
    # Owner, 2026-11-01: with four jobs queued and the dispatcher not started, the screen showed no slots at all.
    from sma.config import RemoteSlot

    cfg = make_config(tmp_path)
    cfg.remote.enabled = True
    cfg.remote.slots = [RemoteSlot("v100-0", "0", remote_port=8001), RemoteSlot("v100-1", "1", remote_port=8002, local_port=18002, search_port=8792)]
    queue = JobQueue(cfg.queue_path())
    first = queue.add([{"repo": "Org/Model-GGUF", "quant": "gguf", "gguf": g, "engine": "llamacpp"} for g in ("F16", "Q4_0")], kind="pair")
    _run(cfg, "20261028T010000Z-sprk", "Org/Spark", "F16", steps={name: "done" for name in STEP_NAMES}, finished=True)
    v100 = _run(cfg, "20261028T020000Z-v100", "Org/Volta", "F16", steps={name: "done" for name in STEP_NAMES}, finished=True)
    _set_manifest(v100, lambda manifest: manifest.update(serving={"hardware": "v100", "slot": "v100-0"}))
    Catalog(cfg.runs_path()).rebuild()

    screen = ANSI.sub("", render(cfg, View([], None, NOW)))

    assert "SLOTS  0 of 2 busy on the serving machine  ·  the dispatcher is not running" in screen and "CURRENT RUN" not in screen
    assert "· v100-0   GPU 0  idle" in screen and "· v100-1   GPU 1  idle" in screen
    assert "✗ the dispatcher is not running: 1 queued job waits, and the memory guard is off. Start it: " in screen
    # What runs comes first, then what is next.
    assert screen.index("SLOTS") < screen.index("QUEUE") < screen.index("FINISHED")
    assert "CPP   1. Org/Model-GGUF  waiting  · F16  · Q4_0" in screen
    # A started job stays in its place: the queue is in the order jobs were queued.
    queue.add([{"repo": "Org/Later-GGUF", "quant": "gguf", "gguf": "F16", "engine": "llamacpp"}], kind="single")
    queue.claim_job(first["job_id"], "dispatcher-v100-pod", "v100-pod")
    lines = ANSI.sub("", render(cfg, View([], None, NOW))).split("QUEUE")[1].splitlines()
    assert "1. Org/Model-GGUF  started" in lines[1] and "2. Org/Later-GGUF  waiting" in lines[2]
    finished = screen.split("FINISHED")[1]
    assert finished.startswith("  1 run served on v100, newest first") and "Org/Volta" in finished and "Org/Spark" not in finished
    assert "SERVING  nothing is serving now" in screen


def test_a_stopped_dispatchers_slot_whose_unit_is_gone_is_idle(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    run = _run(cfg, "20261029T020534Z-aaaa", "Org/Model", "F16", steps={"quality": "running"})
    _dispatcher(cfg, [{"name": "v100-0", "gpus": "0", "state": "busy", "run_id": run.name, "pid": DEAD_PID, "repo": "Org/Model", "gguf": "F16"}],
                pid=DEAD_PID)  # fmt: skip
    Catalog(cfg.runs_path()).rebuild()
    screen = ANSI.sub("", render(cfg, View([], None, NOW)))
    assert "· v100-0   GPU 0  idle" in screen and "0 of 1 slots busy" in screen.splitlines()[0]


def test_the_top_line_and_alerts_wrap_at_a_space_instead_of_losing_their_end() -> None:
    alert = "✗ the dispatcher is not running: 4 queued jobs wait, and the memory guard is off. Start it: sudo systemctl start sma-dispatcher.service"
    rows = wrap(alert, 60)
    assert len(rows) == 3 and all(len(row) <= 60 for row in rows) and rows[1].startswith("  ")
    assert " ".join(row.strip() for row in rows) == alert  # No word is cut in two.
    lines = _plain(layout(Board("SMA watch", [alert], []), Ui(), 88, 20).lines)
    assert lines[1].startswith("✗ the dispatcher") and lines[2].strip().endswith("sma-dispatcher.service")
