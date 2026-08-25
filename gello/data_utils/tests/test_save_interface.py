"""End-to-end test of SaveInterface driven by real keystrokes over a pty.

No robot and no cameras: a fake obs dict stands in for RobotEnv.get_obs().
"""

import json
import os
import pty
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

VIEWS = ["top", "left_wrist", "right_wrist"]


def fake_obs(i):
    obs = {}
    for v in VIEWS:
        obs[f"{v}_rgb"] = np.full((480, 640, 3), (i * 7) % 256, np.uint8)
        obs[f"{v}_depth"] = np.zeros((480, 640, 1), np.uint16)
        obs[f"{v}_stamp"] = 1000.0 + i * 0.033
    obs["joint_positions"] = np.arange(14, dtype=float) + i
    obs["joint_velocities"] = np.zeros(14)
    obs["ee_pos_quat"] = np.zeros(14)
    obs["gripper_position"] = np.array([0.0, 1.0])
    return obs


class _Pty:
    def __enter__(self):
        self.controller, self.follower = pty.openpty()
        self._saved = sys.stdin
        sys.stdin = os.fdopen(os.dup(self.follower), "r")
        return self

    def send(self, keys: str):
        os.write(self.controller, keys.encode())
        time.sleep(0.05)

    def __exit__(self, *exc):
        sys.stdin.close()
        sys.stdin = self._saved
        os.close(self.controller)
        os.close(self.follower)


def _spin(si, n, start=0):
    """Run n control-loop iterations. Returns "quit" if the interface asked to stop."""
    for i in range(start, start + n):
        if si.update(fake_obs(i), np.full(14, float(i))) == "quit":
            return "quit"
        time.sleep(1 / 60.0)  # faster than realtime, still paced
    return None


def test_record_keep_discard_and_quit():
    from gello.utils.control_utils import SaveInterface

    with tempfile.TemporaryDirectory() as td, _Pty() as tty:
        si = SaveInterface(data_dir=td, agent_name="BimanualAgent",
                           task="pick up the red block", meta_extra={"robot": "FakeRobot"})
        try:
            # idle: nothing recorded yet
            _spin(si, 3)
            assert not list(Path(td).glob("*/*/meta.json"))

            # 's' starts a take, 'q' keeps it
            tty.send("s")
            _spin(si, 20)
            assert si._writer is not None, "start key did not begin a take"
            tty.send("q")
            _spin(si, 2)
            assert si._writer is None, "stop key did not end the take"

            eps = sorted(Path(td).glob("pick_up_the_red_block/*/meta.json"))
            assert len(eps) == 1, f"expected 1 episode, found {len(eps)}"
            meta = json.loads(eps[0].read_text())
            assert meta["task"] == "pick up the red block"
            assert meta["outcome"] == "success", "q should record a success"
            assert meta["agent"] == "BimanualAgent"
            assert meta["robot"] == "FakeRobot"
            assert meta["views"] == VIEWS, "views should follow config order"
            assert meta["dropped_frames"] == 0
            assert 15 <= meta["num_frames"] <= 22, meta["num_frames"]
            ep_dir = eps[0].parent
            for v in VIEWS:
                assert len(list((ep_dir / v).glob("*.jpg"))) == meta["num_frames"]
            print(f"  kept episode: {meta['num_frames']} frames, "
                  f"{meta['bytes_images']/1e6:.1f} MB")

            # 'd' discards: directory must be gone, kept episode untouched
            tty.send("s")
            _spin(si, 10)
            discarded = si._writer.dir
            tty.send("d")
            _spin(si, 2)
            assert not discarded.exists(), "discard left the directory behind"
            assert len(list(Path(td).glob("*/*/meta.json"))) == 1

            # pressing 's' twice must not open a second directory
            tty.send("s")
            _spin(si, 5)
            first = si._writer.dir
            tty.send("s")
            _spin(si, 5)
            assert si._writer.dir == first, "double-press opened a new episode"

            # 'f' keeps a take but marks it a failure
            tty.send("s")
            _spin(si, 10)
            tty.send("f")
            _spin(si, 2)
            fails = [json.loads(p.read_text())
                     for p in Path(td).glob("*/*/meta.json")]
            assert sorted(m["outcome"] for m in fails) == ["failure", "success"], \
                [m["outcome"] for m in fails]

            # 'x' quits, finalising the in-progress take rather than losing it
            tty.send("s")
            _spin(si, 6)
            assert si._writer is not None, "expected a take in progress before quit"
            tty.send("x")
            assert _spin(si, 3) == "quit", "quit key was not reported"
            metas = [json.loads(p.read_text())
                     for p in Path(td).glob("*/*/meta.json")]
            assert len(metas) == 3, "quit must finalise the in-progress take"
            # A take interrupted by quit was never judged, so it must not be
            # silently counted as a success.
            assert sorted(m["outcome"] for m in metas) == \
                ["failure", "success", "unspecified"], [m["outcome"] for m in metas]
        finally:
            si.close()
    print("test_record_keep_discard_and_quit OK")


def test_close_restores_terminal_and_finalises():
    """The Ctrl-C path: cleanup() calls close() before os._exit(0)."""
    import termios

    from gello.utils.control_utils import SaveInterface

    with tempfile.TemporaryDirectory() as td, _Pty() as tty:
        before = termios.tcgetattr(tty.follower)
        si = SaveInterface(data_dir=td, agent_name="A", task="t")
        assert not (termios.tcgetattr(tty.follower)[3] & termios.ECHO)

        tty.send("s")
        _spin(si, 10)
        assert si._writer is not None

        si.close()  # what cleanup() does on the signal path
        assert termios.tcgetattr(tty.follower)[3] == before[3], "terminal not restored"
        assert len(list(Path(td).glob("*/*/meta.json"))) == 1, \
            "in-progress take was lost on close()"
        si.close()  # idempotent
    print("test_close_restores_terminal_and_finalises OK")


def test_unlabeled_task_still_records():
    from gello.utils.control_utils import SaveInterface

    with tempfile.TemporaryDirectory() as td, _Pty() as tty:
        # A blank task triggers the interactive prompt, so feed it an empty line.
        tty.send("\n")
        si = SaveInterface(data_dir=td, agent_name="A", task="   ")
        try:
            assert si.task == "unlabeled"
            tty.send("s")
            _spin(si, 5)
            tty.send("q")
            _spin(si, 2)
            assert list(Path(td).glob("unlabeled/*/meta.json"))
        finally:
            si.close()
    print("test_unlabeled_task_still_records OK")


if __name__ == "__main__":
    test_record_keep_discard_and_quit()
    test_close_restores_terminal_and_finalises()
    test_unlabeled_task_still_records()
    print("\nall tests passed")
