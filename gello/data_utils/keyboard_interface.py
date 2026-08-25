"""Terminal key reader for the data-collection save interface.

Returns key *events* only. Recording state lives in SaveInterface -- when this
class owned it, "already recording" was invisible to the caller and pressing the
start key mid-take silently opened a second episode directory.

Terminal restore is the fragile part. `tty.setcbreak` clears ICANON and ECHO, so
a process that exits without restoring leaves the operator's shell with no echo.
Relying on `__del__` is not enough: `launch_yaml.signal_handler` calls
`os._exit(0)`, which runs neither finalisers nor atexit handlers. Use this as a
context manager, and make sure whatever runs on the signal path calls restore().
"""

import atexit
import os
import re
import select
import sys
import termios
import tty
from collections import deque
from contextlib import contextmanager
from typing import Deque, Optional

# Arrow keys and friends arrive as multi-byte escape sequences; without this a
# single arrow press would look like several ordinary keys.
_ANSI_ESCAPE = re.compile(r"\x1b\[?[0-9;]*[A-Za-z~]?")

KEY_EVENTS = {
    "s": "start",
    "q": "stop",
    "f": "fail",
    "d": "discard",
    "x": "quit",
    "t": "relabel",
    "h": "help",
    "?": "help",
}

HELP_LINES = [
    "  s  start recording a take",
    "  q  stop and keep as SUCCESS",
    "  f  stop and keep as FAILURE",
    "  d  stop and DISCARD (deletes it)",
    "  t  set/change the task description",
    "  x  quit (finishes an in-progress take first)",
    "  h  show this help",
]


class KBReset:
    """Non-blocking single-key reader over stdin in cbreak mode."""

    def __init__(self) -> None:
        self._events: Deque[str] = deque()
        self._old: Optional[list] = None
        self._raw = False
        self._fd = -1

        try:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
        except (ValueError, OSError, termios.error):
            # Not a tty (piped stdin, no controlling terminal). Stay inert
            # rather than crashing a run that does not need keyboard input.
            print("Warning: stdin is not a terminal; keyboard controls disabled.")
            self._old = None
            return

        self._enter_raw()
        # Belt and braces: covers a plain sys.exit or an unhandled exception.
        atexit.register(self.restore)

    @property
    def enabled(self) -> bool:
        return self._old is not None

    def _enter_raw(self) -> None:
        if self.enabled and not self._raw:
            # cbreak, not raw: this leaves ISIG on so Ctrl-C still signals.
            tty.setcbreak(self._fd)
            self._raw = True

    def restore(self) -> None:
        """Put the terminal back. Safe to call repeatedly and from a signal path."""
        if self._old is not None and self._raw:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except (OSError, termios.error):
                pass
            self._raw = False

    def __enter__(self) -> "KBReset":
        self._enter_raw()
        return self

    def __exit__(self, *exc) -> None:
        self.restore()

    def __del__(self) -> None:
        self.restore()

    @contextmanager
    def suspended(self):
        """Temporarily restore canonical mode so input() works, then resume."""
        self.restore()
        try:
            yield
        finally:
            self._enter_raw()

    def poll(self) -> Optional[str]:
        """Return the next pending key event, or None. Never blocks."""
        if not self.enabled:
            return None

        if not self._events and select.select([self._fd], [], [], 0)[0]:
            try:
                chunk = os.read(self._fd, 1024).decode("utf-8", "ignore")
            except OSError:
                return None
            # Drain the whole buffer at once, so a fast double-press is not lost
            # and a held-down key does not queue hundreds of events.
            for ch in _ANSI_ESCAPE.sub("", chunk).lower():
                event = KEY_EVENTS.get(ch)
                if event is not None and (
                    not self._events or self._events[-1] != event
                ):
                    self._events.append(event)

        return self._events.popleft() if self._events else None
