"""Tests for KBReset. Uses a pty so the cbreak/restore path is exercised for real."""

import os
import pty
import sys
import termios
import time

from gello.data_utils.keyboard_interface import KBReset


class _StdinAs:
    """Swap sys.stdin for a fd-backed file object."""

    def __init__(self, fd):
        self.fd = fd

    def __enter__(self):
        self._saved = sys.stdin
        sys.stdin = os.fdopen(os.dup(self.fd), "r")
        return self

    def __exit__(self, *exc):
        sys.stdin.close()
        sys.stdin = self._saved


def _drain(kb, timeout=1.0):
    """Collect events until the pty goes quiet."""
    out, deadline = [], time.time() + timeout
    while time.time() < deadline:
        ev = kb.poll()
        if ev is None:
            if out:
                break
            time.sleep(0.01)
            continue
        out.append(ev)
    return out


def test_keys_map_to_events():
    controller, follower = pty.openpty()
    try:
        with _StdinAs(follower):
            kb = KBReset()
            assert kb.enabled, "pty should be a tty"
            try:
                os.write(controller, b"sqdtxh")
                assert _drain(kb) == [
                    "start", "stop", "discard", "relabel", "quit", "help"
                ]

                os.write(controller, b"S")  # uppercase folds to lowercase
                assert _drain(kb) == ["start"]

                os.write(controller, b"zZ1 \n")  # unmapped keys are ignored
                assert kb.poll() is None
            finally:
                kb.restore()
    finally:
        os.close(controller)
        os.close(follower)
    print("test_keys_map_to_events OK")


def test_arrow_keys_do_not_fire_events():
    """An arrow press is ESC [ A -- the bare bytes would look like real keys."""
    controller, follower = pty.openpty()
    try:
        with _StdinAs(follower):
            kb = KBReset()
            try:
                os.write(controller, b"\x1b[A\x1b[B\x1b[C\x1b[D")
                time.sleep(0.05)
                assert kb.poll() is None, "escape sequence leaked through"
                os.write(controller, b"\x1b[Aq")  # real key after a sequence
                assert _drain(kb) == ["stop"]
            finally:
                kb.restore()
    finally:
        os.close(controller)
        os.close(follower)
    print("test_arrow_keys_do_not_fire_events OK")


def test_terminal_is_restored():
    """The failure mode: exiting without restoring leaves the shell with no echo."""
    controller, follower = pty.openpty()
    try:
        before = termios.tcgetattr(follower)
        with _StdinAs(follower):
            kb = KBReset()
            during = termios.tcgetattr(follower)
            assert during[3] != before[3], "cbreak should have changed lflags"
            assert not (during[3] & termios.ECHO), "ECHO should be off in cbreak"
            kb.restore()
            assert termios.tcgetattr(follower)[3] == before[3], "not restored"
            kb.restore()  # idempotent
            assert termios.tcgetattr(follower)[3] == before[3]
    finally:
        os.close(controller)
        os.close(follower)
    print("test_terminal_is_restored OK")


def test_context_manager_and_suspend():
    controller, follower = pty.openpty()
    try:
        before = termios.tcgetattr(follower)
        with _StdinAs(follower):
            with KBReset() as kb:
                assert not (termios.tcgetattr(follower)[3] & termios.ECHO)
                with kb.suspended():
                    # Canonical mode restored, so input() would work here.
                    assert termios.tcgetattr(follower)[3] == before[3]
                assert not (termios.tcgetattr(follower)[3] & termios.ECHO)
            assert termios.tcgetattr(follower)[3] == before[3], "__exit__ must restore"
    finally:
        os.close(controller)
        os.close(follower)
    print("test_context_manager_and_suspend OK")


if __name__ == "__main__":
    test_keys_map_to_events()
    test_arrow_keys_do_not_fire_events()
    test_terminal_is_restored()
    test_context_manager_and_suspend()
    print("\nall tests passed")
