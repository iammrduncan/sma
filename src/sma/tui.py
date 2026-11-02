"""The terminal UI of `sma watch`: panels laid out for the terminal's size, keys to move between them, and a
detail view of one run or queue job. What each panel says comes from sma.watch; this module lays it out,
follows keys, and draws it with curses from the standard library. Everything above `run` is pure, so it is
tested without a terminal. Lines carry ANSI colors (sma.watch's painter); curses draws them as attributes.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

_ANSI = re.compile(r"\033\[([0-9;]*)m")


@dataclass
class Item:
    """One selectable thing in a panel: its lines, what Enter opens (("run", run id) or ("job", job id)), and its
    window: lines shown under the panel while the item is selected (a slot's run, whole)."""

    lines: list[str]
    key: tuple[str, str] | None = None
    window: list[str] | None = None


@dataclass
class Panel:
    name: str
    title: str
    items: list[Item] = field(default_factory=list)
    # Lines after the items that are not selectable (the serving table, the memory guard's last stop).
    notes: list[str] = field(default_factory=list)
    # One line that stands for the whole panel when it gets fewer rows than it needs (MACHINES).
    compact: str | None = None
    # With no height limit (--once), at most this many items; None: all.
    once_items: int | None = None


@dataclass
class Board:
    """One read of the world, as painted lines: the top line, the alerts, the panels, and the open item's detail."""

    summary: str
    alerts: list[str]
    panels: list[Panel]
    # The open item's lines; None when nothing is open or it is gone.
    detail: list[str] | None = None


@dataclass
class Ui:
    """What the person chose: the focused panel, each panel's selected item, the open item, and so on."""

    focus: str = "active"
    selected: dict[str, int] = field(default_factory=dict)
    detail: tuple[str, str] | None = None
    scroll: int = 0
    filter: str = ""
    typing: bool = False
    help: bool = False
    message: str = ""


@dataclass
class Frame:
    """One drawn screen: its lines, the ranges drawn reversed (row, first column, end column), the rows the body
    has, and the detail view's length (for scrolling)."""

    lines: list[str]
    highlight: list[tuple[int, int, int]] = field(default_factory=list)
    page: int = 0
    detail_rows: int = 0


# The panels Tab moves between, in order; 1 to 4 jump to them.
FOCUSABLE = ("active", "queue", "kernelbench", "finished")
# The order panels get rows in when the screen is short, after the focused one.
_PRIORITY = ("active", "window", "queue", "finished", "kernelbench", "serving", "machines")
# Two columns from this width: the panels with wide lines on the left, the rest in a column as wide as the
# serving table.
_TWO_COLUMNS_AT = 190
_RIGHT_WIDTH = 94
_LEFT = ("active", "window", "finished")
_ALERTS_SHOWN = 4
_HINTS = "Tab panel · ↑↓ select · Enter open · / filter · r refresh · ? help · q quit"
_DETAIL_HINTS = "↑↓ PgUp PgDn scroll · Esc back · r refresh · q quit"
HELP = [
    "KEYS",
    "  Tab, Shift-Tab   next, previous panel: running, queue, KernelBench, finished",
    "  1 2 3 4          jump to that panel",
    "  ↑ ↓, k j         move the selection; in SLOTS or RUNNING, it picks the run shown whole below;",
    "                   in a detail view, scroll a line",
    "  PgUp PgDn        a page",
    "  Home End, g G    the first, the last",
    "  Enter            open the selected run or queue job: where it runs, its list with whole errors, its",
    "                   serving rates, and its recent events",
    "  Esc, Backspace   back to the main screen",
    "  /                filter: type text, Enter keeps it, Esc clears it",
    "  r                read everything now (it reads again every --interval seconds)",
    "  ?                this help; any key closes it",
    "  q, Ctrl-C        quit",
    "",
    "MARKS",
    "  ✓ done   ▶ running   ✗ failed   · queued   ◼ stopped, resumable",
    "  ✗ on the alert lines is a problem, ! a warning: a slot down, a dead or silent dispatcher, a memory-guard",
    "  stop, low memory, a machine that does not answer, a failed step, a step far past its usual time, a run",
    "  that writes nothing, a stale queue claim.",
    "",
    "sma watch only reads. It changes nothing, and the screen is scrubbed like a run record.",
]


def plain(board: Board, width: int | None = None) -> str:
    """The main screen as text, every panel in full (FINISHED up to its once_items), then every item's window:
    `sma watch --once`."""
    lines = [board.summary, *board.alerts, ""]
    for panel in board.panels:
        items = panel.items if panel.once_items is None else panel.items[: panel.once_items]
        lines.append(panel.title)
        for item in items:
            lines += item.lines
        if len(items) < len(panel.items):
            lines.append(f"  … {len(panel.items) - len(items)} more")
        lines += panel.notes
        for item in items:
            lines += [""] + item.window if item.window else []
        lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(clip(line, width) for line in lines) if width else "\n".join(lines)


def layout(board: Board, ui: Ui, width: int, height: int) -> Frame:
    """The TUI's screen for width x height: the top line, the alerts, the body (panels, a detail view, or help),
    and a line of key hints."""
    width, height = max(width, 20), max(height, 4)
    alerts = board.alerts if len(board.alerts) <= _ALERTS_SHOWN else board.alerts[: _ALERTS_SHOWN - 1] + [
        _dim(f"  … {len(board.alerts) - _ALERTS_SHOWN + 1} more: sma watch --once lists every one")
    ]
    # The top line and each alert wrap onto a second row rather than lose their end (an alert ends with its fix).
    top = [row for line in [board.summary, *alerts] for row in wrap(line, width)[:2]][: max(1, height - 3)]
    page = height - len(top) - 1
    frame = Frame([], page=page)
    if ui.help:
        body = HELP[:page]
    elif ui.detail is not None:
        wrapped = [row for line in (board.detail or ["This item is gone: Esc goes back."]) for row in wrap(line, width)]
        frame.detail_rows = len(wrapped)
        start = min(max(ui.scroll, 0), max(0, len(wrapped) - page))
        body = wrapped[start : start + page]
    elif width >= _TWO_COLUMNS_AT:
        left_width = width - _RIGHT_WIDTH - 3
        panels = _with_windows(board.panels, ui)
        left = _stack([p for p in panels if p.name in _LEFT], ui, left_width, page)
        right = _stack([p for p in panels if p.name not in _LEFT], ui, _RIGHT_WIDTH, page)
        body = []
        for row in range(max(len(left[0]), len(right[0]))):
            text = left[0][row] if row < len(left[0]) else ""
            body.append(pad(text, left_width) + _dim(" │ ") + (right[0][row] if row < len(right[0]) else ""))
        frame.highlight = [(len(top) + row, x0, x1) for row, x0, x1 in left[1]]
        frame.highlight += [(len(top) + row, left_width + 3 + x0, left_width + 3 + x1) for row, x0, x1 in right[1]]
    else:
        body, marks = _stack(_with_windows(board.panels, ui), ui, width, page)
        frame.highlight = [(len(top) + row, x0, x1) for row, x0, x1 in marks]
    frame.lines = [clip(line, width) for line in top + body]
    frame.lines += [""] * (height - 1 - len(frame.lines))
    frame.lines.append(clip(_dim(_footer(ui)), width))
    return frame


def _footer(ui: Ui) -> str:
    if ui.typing:
        return f"/{ui.filter}▏  Enter keeps the filter · Esc clears it"
    if ui.message:
        return ui.message
    if ui.help:
        return "any key closes the help"
    hints = _DETAIL_HINTS if ui.detail is not None else _HINTS
    return hints + (f" · filter: {ui.filter}" if ui.filter else "")


def visible(panel: Panel, ui: Ui) -> list[Item]:
    """The panel's items that contain the filter (case ignored). The filter applies to the panels Tab reaches."""
    if not ui.filter or panel.name not in FOCUSABLE:
        return panel.items
    needle = ui.filter.lower()
    return [item for item in panel.items if any(needle in plain_text(line).lower() for line in item.lines)]


def _selected(ui: Ui, name: str, count: int) -> int:
    return min(max(ui.selected.get(name, 0), 0), max(count - 1, 0))


def _with_windows(panels: list[Panel], ui: Ui) -> list[Panel]:
    """The panels, each followed by its selected item's window when it has one, as a panel of its own that Tab does
    not reach. It stays while another panel has the focus."""
    out = []
    for panel in panels:
        out.append(panel)
        items = visible(panel, ui)
        window = items[_selected(ui, panel.name, len(items))].window if items else None
        if window:
            out.append(Panel("window", window[0], notes=window[1:]))
    return out


def _stack(panels: list[Panel], ui: Ui, width: int, height: int) -> tuple[list[str], list[tuple[int, int, int]]]:
    """Panels one under the other in at most height rows. Rows go first to each panel's title and one row, in
    priority order (the focused panel first, with room for its selected item), then fill each panel in turn."""
    shown = {panel.name: visible(panel, ui) for panel in panels}
    hidden = {panel.name: len(panel.items) - len(shown[panel.name]) for panel in panels}
    need = {p.name: 1 + sum(len(item.lines) for item in shown[p.name]) + len(p.notes) + (1 if hidden[p.name] else 0) for p in panels}
    # Highest priority first: the focused panel, then _PRIORITY's order.
    order = sorted(panels, key=lambda p: (p.name != ui.focus, _PRIORITY.index(p.name) if p.name in _PRIORITY else len(_PRIORITY)))
    give: dict[str, int] = {}
    left = height
    for panel in order:
        items = shown[panel.name]
        first = 1 if panel.compact else 2
        if panel.name == ui.focus and items:
            # Room for the selected item, up to half the rows: a tall item must not hide every other panel.
            first = max(first, min(1 + len(items[_selected(ui, panel.name, len(items))].lines), height // 2))
        give[panel.name] = min(need[panel.name], first, left)
        left -= give[panel.name]
    for panel in order:
        extra = min(need[panel.name] - give[panel.name], left) if give[panel.name] else 0
        give[panel.name] += extra
        left -= extra
    lines: list[str] = []
    marks: list[tuple[int, int, int]] = []
    for panel in panels:
        rows = give[panel.name]
        if not rows:
            continue
        if panel.compact and rows < need[panel.name]:
            lines.append(panel.compact)
            continue
        focused = panel.name == ui.focus
        block, selected_row = _panel_rows(panel, shown[panel.name], hidden[panel.name], rows, ui, focused)
        if focused:
            # The focused panel's name, reversed; its selected item's first line, reversed across the column.
            marks.append((len(lines), 0, len(plain_text(panel.title).split("  ")[0])))
        if selected_row is not None:
            marks.append((len(lines) + selected_row, 0, width))
        lines += block
    return lines, marks


def _panel_rows(panel: Panel, items: list[Item], hidden: int, rows: int, ui: Ui, focused: bool) -> tuple[list[str], int | None]:
    """The panel's title and as many of its lines as fit in rows, scrolled so the selected item shows. Returns the
    lines and the row of the selected item's first line (focused panels only)."""
    body: list[tuple[str, int | None]] = [(line, index) for index, item in enumerate(items) for line in item.lines]
    body += [(line, None) for line in panel.notes]
    if hidden:
        body.append((_dim(f"  ({hidden} hidden by the filter)"), None))
    window = rows - 1
    selected = _selected(ui, panel.name, len(items)) if items else None
    rows_of = [row for row, (_line, index) in enumerate(body) if index == selected and selected is not None]
    start = 0
    if len(body) > window and rows_of:
        # Scroll just enough to show the selected item's last line, but never past its first.
        start = min(max(0, rows_of[-1] - window + 1), rows_of[0])
    shown = body[start : start + window]
    title = panel.title
    if len(body) > window:
        # What does not fit is counted on the title line, so it costs no row.
        seen = {index for _line, index in shown}
        above = len({index for _line, index in body[:start] if index is not None})
        below = len({index for _line, index in body[start + window :] if index is not None and index not in seen})
        parts = [f"{above} above" if above else "", f"{below} below" if below else ""]
        lines_left = len(body) - window - start
        title += _dim("  … " + (", ".join(part for part in parts if part) or f"{lines_left} more line{'' if lines_left == 1 else 's'}"))
    selected_row = None
    if focused and rows_of and start <= rows_of[0] < start + window:
        selected_row = 1 + rows_of[0] - start
    return [title] + [line for line, _index in shown], selected_row


def handle_key(ui: Ui, key: str, board: Board, frame: Frame) -> str | None:
    """Follow one key. Returns "quit", "refresh" (read the world now), or None (draw again from the last read)."""
    ui.message = ""
    if ui.typing:
        if key == "enter":
            ui.typing = False
        elif key == "esc":
            ui.typing, ui.filter = False, ""
        elif key == "backspace":
            ui.filter = ui.filter[:-1]
        elif len(key) == 1 and key.isprintable():
            ui.filter += key
        ui.selected = {}
        return None
    if key in ("q", "ctrl-c"):
        return "quit"
    if ui.help:
        ui.help = False
        return None
    if key == "?":
        ui.help = True
        return None
    if key == "r":
        return "refresh"
    if key == "resize":
        return None
    if ui.detail is not None:
        if key in ("esc", "backspace", "left", "h"):
            ui.detail, ui.scroll = None, 0
            return None
        last = max(0, frame.detail_rows - frame.page)
        moves = {"up": -1, "k": -1, "down": 1, "j": 1, "pgup": -frame.page, "pgdn": frame.page, "home": -last - 1, "g": -last - 1}
        if key in ("end", "G"):
            ui.scroll = last
        elif key in moves:
            ui.scroll = min(max(min(ui.scroll, last) + moves[key], 0), last)
        return None
    present = [name for name in FOCUSABLE if any(panel.name == name for panel in board.panels)]
    if ui.focus not in present and present:
        ui.focus = present[0]
    if key == "/":
        ui.typing = True
        return None
    if key == "esc":
        ui.filter = ""
        return None
    if key in ("tab", "btab") and present:
        step = 1 if key == "tab" else -1
        ui.focus = present[(present.index(ui.focus) + step) % len(present)] if ui.focus in present else present[0]
        return None
    if key in ("1", "2", "3", "4"):
        name = FOCUSABLE[int(key) - 1]
        if name in present:
            ui.focus = name
        else:
            ui.message = "That panel has nothing to show now."
        return None
    panel = next((panel for panel in board.panels if panel.name == ui.focus), None)
    items = visible(panel, ui) if panel else []
    current = _selected(ui, ui.focus, len(items))
    page = max(1, frame.page // 4)
    moves = {"up": -1, "k": -1, "down": 1, "j": 1, "pgup": -page, "pgdn": page, "home": -len(items), "g": -len(items),
             "end": len(items), "G": len(items)}  # fmt: skip
    if key in moves:
        ui.selected[ui.focus] = min(max(current + moves[key], 0), max(len(items) - 1, 0))
        return None
    if key in ("enter", "right", "l"):
        item = items[current] if items else None
        if item is None or item.key is None:
            ui.message = "Nothing to open here: an idle or down slot, or a plan line with no run yet."
            return None
        ui.detail, ui.scroll = item.key, 0
        return "refresh"
    return None


# Text with ANSI colors.


def plain_text(line: str) -> str:
    return _ANSI.sub("", line)


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m"


def pad(line: str, width: int) -> str:
    line = clip(line, width)
    return line + " " * (width - len(plain_text(line)))


def clip(line: str, width: int | None) -> str:
    """The line cut to width visible characters, the last one an ellipsis when it was cut. Colors stay closed."""
    if width is None or len(plain_text(line)) <= width:
        return line
    out, visible_count, index, open_color = [], 0, 0, False
    while index < len(line):
        code = _ANSI.match(line, index)
        if code:
            out.append(code.group(0))
            open_color = code.group(0) != "\033[0m"
            index = code.end()
            continue
        if visible_count == width - 1:
            out.append("…")
            break
        out.append(line[index])
        visible_count += 1
        index += 1
    return "".join(out) + ("\033[0m" if open_color else "")


def wrap(line: str, width: int) -> list[str]:
    """The line cut into rows of width visible characters, at a space when there is one in the row's second half,
    each row's colors closed and opened again on the next. Continuation rows are indented like the line."""
    text = plain_text(line)
    if len(text) <= width:
        return [line]
    indent = len(text) - len(text.lstrip(" "))
    indent = indent + 2 if indent + 12 < width else 0
    # Each visible character with the color codes just before it, and the color in force after them.
    chars: list[tuple[str, str, str]] = []
    codes, color, index = "", "", 0
    while index < len(line):
        code = _ANSI.match(line, index)
        if code:
            codes += code.group(0)
            color = "" if code.group(0) == "\033[0m" else code.group(0)
            index = code.end()
            continue
        chars.append((line[index], codes, color))
        codes, index = "", index + 1
    rows: list[str] = []
    start, carried = 0, ""
    while start < len(chars):
        room = width - (indent if rows else 0)
        end = min(start + room, len(chars))
        if end < len(chars) and chars[end][0] != " ":
            end = max((i + 1 for i in range(start + room // 2, end) if chars[i][0] == " "), default=end)
        before = chars[start - 1][2] if start else ""
        row = (" " * indent + before if rows else "") + carried + "".join(prefix + char for char, prefix, _color in chars[start:end])
        rows.append(row + (codes if end == len(chars) else "\033[0m" if chars[end - 1][2] else ""))
        # A continuation row does not start with the space the break fell on; that space's colors carry over.
        carried, start = "", end
        while start < len(chars) and chars[start][0] == " ":
            carried, start = carried + chars[start][1], start + 1
    return rows


def segments(line: str) -> list[tuple[str, str]]:
    """The line as (text, SGR code) pieces: "" for plain text, "32" for green, "38;5;196" for a 256-color red."""
    pieces, code, index = [], "", 0
    for match in _ANSI.finditer(line):
        if match.start() > index:
            pieces.append((line[index : match.start()], code))
        code = "" if match.group(1) in ("", "0") else match.group(1)
        index = match.end()
    if index < len(line):
        pieces.append((line[index:], code))
    return pieces


# Curses: a thin loop around the pure functions above.

# 256-color codes the painter uses, as one of the 8 basic colors for a terminal without 256.
_BASIC = {196: 1, 202: 1, 208: 1, 214: 3, 220: 3, 226: 3, 190: 2, 154: 2, 118: 2, 46: 2, 39: 4, 135: 5}


def _foreground(code: str, colors: int) -> int | None:
    parts = code.split(";")
    if len(parts) >= 3 and parts[0] == "38" and parts[1] == "5" and parts[2].isdigit():
        value = int(parts[2])
        return value if colors >= 256 else _BASIC.get(value, 7)
    basic = [int(part) - 30 for part in parts if part.isdigit() and 30 <= int(part) <= 37]
    return basic[-1] if basic else None


def attributes(curses: Any, code: str, color: Callable[[str], int] | None) -> int:
    """The curses attribute for an SGR code: bold, dim, and reverse always; the color only with color(code)."""
    parts = code.split(";")
    attr = curses.A_NORMAL
    if parts[0] == "1":
        attr |= curses.A_BOLD
    if parts[0] == "2":
        attr |= curses.A_DIM
    if "7" in parts:
        attr |= curses.A_REVERSE
    return attr | color(code) if color and code else attr


def _palette(curses: Any) -> Callable[[str], int] | None:
    """A color pair per foreground color, on the terminal's own background. None: no color (NO_COLOR, or none)."""
    if os.environ.get("NO_COLOR") or not curses.has_colors():
        return None
    curses.start_color()
    try:
        curses.use_default_colors()
        background = -1
    except curses.error:
        background = curses.COLOR_BLACK
    pairs: dict[int, int] = {}

    def pair(code: str) -> int:
        foreground = _foreground(code, curses.COLORS)
        if foreground is None:
            return 0
        if foreground not in pairs:
            number = len(pairs) + 1
            if number >= curses.COLOR_PAIRS:
                return 0
            curses.init_pair(number, foreground, background)
            pairs[foreground] = number
        return curses.color_pair(pairs[foreground])

    return pair


def draw(curses: Any, screen: Any, frame: Frame, color: Callable[[str], int] | None) -> None:
    screen.erase()
    height, width = screen.getmaxyx()
    for row, line in enumerate(frame.lines[:height]):
        column = 0
        for text, code in segments(line):
            text = text[: max(0, width - column)]
            if not text:
                break
            try:
                screen.addstr(row, column, text, attributes(curses, code, color))
            except curses.error:
                pass  # The bottom-right cell: curses moves the cursor past the screen's end and complains.
            column += len(text)
    for row, start, end in frame.highlight:
        if row < height and start < width:
            try:
                screen.chgat(row, start, min(end, width) - start, curses.A_REVERSE)
            except curses.error:
                pass
    screen.refresh()


def _key(curses: Any, screen: Any) -> str | None:
    """The next key's name, or None when none came before the screen's timeout."""
    try:
        key = screen.get_wch()
    except curses.error:
        return None
    if isinstance(key, int):
        names = {
            curses.KEY_UP: "up", curses.KEY_DOWN: "down", curses.KEY_LEFT: "left", curses.KEY_RIGHT: "right",
            curses.KEY_PPAGE: "pgup", curses.KEY_NPAGE: "pgdn", curses.KEY_HOME: "home", curses.KEY_END: "end",
            curses.KEY_ENTER: "enter", curses.KEY_BACKSPACE: "backspace", curses.KEY_BTAB: "btab", curses.KEY_RESIZE: "resize",
        }  # fmt: skip
        return names.get(key)
    return {"\n": "enter", "\r": "enter", "\t": "tab", "\x1b": "esc", "\x7f": "backspace", "\b": "backspace", "\x03": "ctrl-c"}.get(key, key)


class Reader:
    """Reads the world on its own thread, one read at a time, so keys and scrolling stay quick while a slow
    source is asked (a machine that does not answer can take its whole timeout). The thread only reads and hands
    back a Board when it ends; a read still going at quit is dropped."""

    def __init__(self, read: Callable[[tuple[str, str] | None], Board]) -> None:
        self._read = read
        self._thread: threading.Thread | None = None
        self._result: tuple[tuple[str, str] | None, Board | None, BaseException | None] | None = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, open_item: tuple[str, str] | None) -> None:
        self._thread = threading.Thread(target=self._run, args=(open_item,), name="sma-watch-read", daemon=True)
        self._thread.start()

    def _run(self, open_item: tuple[str, str] | None) -> None:
        try:
            self._result = (open_item, self._read(open_item), None)
        except BaseException as exc:  # Raised again on the screen's thread, which ends the TUI with it.
            self._result = (open_item, None, exc)

    def take(self) -> tuple[tuple[str, str] | None, Board] | None:
        """The finished read: (the item it opened, its board). None while it runs or when it was taken."""
        if self.busy or self._result is None:
            return None
        open_item, board, error = self._result
        self._result = None
        if error is not None:
            raise error
        assert board is not None
        return open_item, board


def loop(curses: Any, screen: Any, read: Callable[[tuple[str, str] | None], Board], interval_s: float, *,
         color: Callable[[str], int] | None = None, clock: Callable[[], float] = time.monotonic,
         reader: Reader | None = None) -> None:  # fmt: skip
    """Draw, wait for a key, follow it; read again every interval_s, or at once after r or Enter. Until q."""
    ui = Ui()
    reader = reader or Reader(read)
    board, board_for = Board("SMA watch  reading…", [], []), None
    reader.start(None)
    read_at, wanted = clock(), False
    screen.keypad(True)
    screen.timeout(200)
    while True:
        finished = reader.take()
        if finished is not None:
            board_for, board = finished
        # A detail opened since the last read comes with the next one.
        shown = board if board_for == ui.detail else replace(board, detail=["Reading…"])
        height, width = screen.getmaxyx()
        frame = layout(shown, ui, width, height)
        draw(curses, screen, frame, color)
        key = _key(curses, screen)
        action = handle_key(ui, key, board, frame) if key else None
        if action == "quit":
            return
        wanted = wanted or action == "refresh" or clock() - read_at >= interval_s
        if wanted and not reader.busy:
            reader.start(ui.detail)
            read_at, wanted = clock(), False


def run(read: Callable[[tuple[str, str] | None], Board], interval_s: float) -> bool:
    """The TUI on this terminal until q or Ctrl-C. read(open) reads the world and returns the board, with the
    detail of open (("run", id) or ("job", id)) when it is not None. False when curses cannot start here."""
    import curses
    import locale

    # Wide characters (✓ ▶ ·) are one cell each only under the terminal's own locale.
    locale.setlocale(locale.LC_ALL, "")
    # Esc goes back at once instead of after curses' default wait for an escape sequence.
    os.environ.setdefault("ESCDELAY", "25")
    started = False

    def main(screen: Any) -> None:
        nonlocal started
        started = True
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        loop(curses, screen, read, interval_s, color=_palette(curses))

    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    except curses.error:
        if started:
            raise
        return False
    return True
