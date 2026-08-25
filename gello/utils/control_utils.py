"""Shared utilities for robot control loops."""

import atexit
import datetime
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gello.agents.agent import Agent
from gello.env import RobotEnv

DEFAULT_MAX_JOINT_DELTA = 1.0


def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually.

    Args:
        env: Robot environment
        agent: Agent that provides target position
        max_delta: Maximum joint delta per step
        steps: Number of steps for gradual movement

    Returns:
        bool: True if successful, False if position too far
    """
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = DEFAULT_MAX_JOINT_DELTA
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return False

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    return True


_RESET = "\033[0m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"

GIB = 1024 ** 3


def _fmt_bytes(n: int) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n}B"


def _views_of(obs: Dict[str, Any]) -> List[str]:
    """Camera view names in config order.

    obs preserves the order RobotEnv iterated its camera_dict, which is the
    order the YAML lists them -- so the overhead view stays first in the monitor
    tiles and in meta.json instead of being alphabetised into the middle.
    """
    return [k[: -len("_rgb")] for k in obs if k.endswith("_rgb")]


class SaveInterface:
    """Keyboard-driven episode recorder for teleop data collection.

    Owns the recording state machine (KBReset only reports key events) and the
    terminal status line. One EpisodeWriter is created per take, so encoding and
    disk I/O stay off the control loop.
    """

    #: Refuse to start a take below this much free space, and warn below the second.
    MIN_FREE_GB = 5.0
    WARN_FREE_GB = 20.0

    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
        task: Optional[str] = None,
        meta_extra: Optional[Dict[str, Any]] = None,
        jpeg_quality: int = 95,
        num_workers: int = 3,
        monitor_port: Optional[int] = None,
    ):
        from gello.data_utils.keyboard_interface import HELP_LINES, KBReset

        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.agent_name = agent_name
        self.meta_extra = dict(meta_extra or {})
        self.jpeg_quality = jpeg_quality
        self.num_workers = num_workers
        self._help_lines = HELP_LINES

        # Prompt before entering cbreak mode, otherwise input() cannot echo.
        self.task = task.strip() if task and task.strip() else self._prompt_task_plain()

        self._writer = None
        self._episodes: List[Dict[str, Any]] = []
        self._last_summary = ""
        self._last_render = 0.0
        self._line_len = 0
        self._closed = False

        self.kb = KBReset()
        atexit.register(self.close)

        self.monitor = None
        if monitor_port:
            from gello.data_utils.monitor import MonitorServer

            try:
                self.monitor = MonitorServer(port=monitor_port)
            except OSError as exc:  # port in use -- not worth failing the run
                print(f"Could not start monitor on port {monitor_port}: {exc}")
        self._views: List[str] = []

        print(f"\nSave interface enabled. Task: {self.task!r}")
        print(f"Writing to {self.data_dir.resolve()}")
        if self.monitor is not None:
            print(f"Live monitor: {self.monitor.url}")
        for line in self._help_lines:
            print(line)
        print()

    # ---- task label --------------------------------------------------------

    @staticmethod
    def _prompt_task_plain() -> str:
        try:
            answer = input("Task description (natural language, used as the "
                           "policy prompt): ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        return answer or "unlabeled"

    def _relabel(self) -> None:
        if self._writer is not None:
            self._message("Cannot relabel mid-take; stop the take first.", _YELLOW)
            return
        self._clear_line()
        with self.kb.suspended():
            try:
                answer = input("New task description: ").strip()
            except (EOFError, KeyboardInterrupt):
                answer = ""
        if answer:
            self.task = answer
            self._message(f"Task is now {self.task!r}", _GREEN)
        else:
            self._message("Task unchanged.", _DIM)

    # ---- status line -------------------------------------------------------

    @property
    def _use_color(self) -> bool:
        return sys.stdout.isatty()

    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{_RESET}" if self._use_color else text

    def _eps_label(self) -> str:
        c = self._counts()
        if not c:
            return "eps=0"
        parts = [f"{c[k]}{k[0]}" for k in ("success", "failure", "unspecified")
                 if c.get(k)]
        return "eps=" + "/".join(parts)

    def _clear_line(self) -> None:
        if self._line_len:
            sys.stdout.write("\r" + " " * self._line_len + "\r")
            sys.stdout.flush()
            self._line_len = 0

    def _message(self, text: str, color: str = "") -> None:
        """Print above the status line without the two clobbering each other."""
        self._clear_line()
        sys.stdout.write((self._c(text, color) if color else text) + "\n")
        sys.stdout.flush()
        self._render(force=True)

    def _free_gb(self) -> float:
        return shutil.disk_usage(self.data_dir).free / GIB

    def _render(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_render < 0.1:  # ~10 Hz is plenty
            return
        self._last_render = now

        free = self._free_gb()
        free_txt = f"free={free:.0f}GB"
        if free < self.MIN_FREE_GB:
            free_txt = self._c(free_txt, _RED)
        elif free < self.WARN_FREE_GB:
            free_txt = self._c(free_txt, _YELLOW)

        w = self._writer
        if w is not None:
            hz = (w.num_frames - 1) / w.duration if w.duration > 0 else 0.0
            parts = [
                self._c("* REC", _GREEN),
                self.task[:28],
                f"t={w.duration:5.1f}s",
                f"n={w.num_frames:5d}",
                f"{hz:4.1f}Hz",
                f"{_fmt_bytes(w.bytes_written):>7}",
                f"q={w.queue_depth:2d}",
                free_txt,
                self._c("[q]success [f]fail [d]discard", _DIM),
            ]
            if w.dropped:
                parts.insert(-1, self._c(f"DROPPED={w.dropped}", _RED))
        else:
            parts = [
                self._c("o idle", _DIM),
                self.task[:28],
                self._eps_label(),
                self._last_summary or "",
                free_txt,
                self._c("[s]start [t]task [x]quit", _DIM),
            ]

        line = "  ".join(p for p in parts if p)
        width = shutil.get_terminal_size((120, 24)).columns
        visible = len(line) if not self._use_color else len(
            line.replace(_RESET, "").replace(_DIM, "").replace(_RED, "")
            .replace(_GREEN, "").replace(_YELLOW, "")
        )
        if visible > width - 1:
            line, visible = line[: width - 1], width - 1
        sys.stdout.write("\r" + line + " " * max(0, self._line_len - visible))
        sys.stdout.flush()
        self._line_len = visible

    # ---- episode lifecycle -------------------------------------------------

    def _episode_dir(self) -> Path:
        from gello.data_utils.episode_writer import slugify

        base = self.data_dir / slugify(self.task)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base / stamp
        n = 2
        while path.exists():  # same-second restart
            path = base / f"{stamp}_{n}"
            n += 1
        return path

    def _start(self, obs: Dict[str, Any]) -> None:
        from gello.data_utils.episode_writer import EpisodeWriter

        if self._writer is not None:
            self._message("Already recording; press q to keep or d to discard.",
                          _YELLOW)
            return

        free = self._free_gb()
        if free < self.MIN_FREE_GB:
            self._message(
                f"Refusing to record: only {free:.1f} GB free "
                f"(need {self.MIN_FREE_GB:.0f} GB). Free up space first.",
                _RED,
            )
            return

        views = self._views or _views_of(obs)
        if not views:
            self._message("No *_rgb keys in obs -- no cameras configured?", _RED)
            return

        meta = dict(self.meta_extra)
        meta.setdefault("agent", self.agent_name)
        path = self._episode_dir()
        self._writer = EpisodeWriter(
            path, task=self.task, views=views, meta_extra=meta,
            num_workers=self.num_workers, jpeg_quality=self.jpeg_quality,
        )
        self._message(f"Recording -> {path}", _GREEN)

    def _stop(self, outcome: Optional[str]) -> None:
        """End the take. outcome=None discards it; otherwise it is recorded."""
        from gello.data_utils.episode_writer import OUTCOME_FAILURE

        w = self._writer
        if w is None:
            if outcome is not None:
                self._message("Not recording.", _DIM)
            return
        self._writer = None

        if outcome is None:
            frames, path = w.num_frames, w.dir
            w.abort()
            self._message(f"Discarded {frames} frames ({path.name})", _YELLOW)
            return

        meta = w.finish(outcome)
        self._episodes.append(meta)
        self._last_summary = (
            f"last={outcome[:4]}/{meta['num_frames']}f/{meta['duration_s']:.1f}s"
        )
        msg = (f"Saved [{outcome.upper()}] {meta['num_frames']} frames in "
               f"{meta['duration_s']:.1f}s @ {meta['hz_effective']:.1f}Hz -> {w.dir}")
        if meta["dropped_frames"]:
            self._message(msg + f"  [{meta['dropped_frames']} DROPPED]", _RED)
        else:
            self._message(msg, _RED if outcome == OUTCOME_FAILURE else _GREEN)

    # ---- called every control-loop iteration -------------------------------

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Handle keys and record a frame. Returns "quit" to stop the loop."""
        from gello.data_utils.episode_writer import (
            OUTCOME_FAILURE, OUTCOME_SUCCESS, OUTCOME_UNSPECIFIED,
        )

        event = self.kb.poll()
        if event == "start":
            self._start(obs)
        elif event == "stop":
            self._stop(OUTCOME_SUCCESS)
        elif event == "fail":
            self._stop(OUTCOME_FAILURE)
        elif event == "discard":
            self._stop(None)
        elif event == "relabel":
            self._relabel()
        elif event == "help":
            self._clear_line()
            print("\n".join(self._help_lines))
        elif event == "quit":
            # Never lose an in-progress take, but do not claim it succeeded --
            # it was never judged.
            self._stop(OUTCOME_UNSPECIFIED)
            self._clear_line()
            print("Exiting.")
            return "quit"

        if self._writer is not None:
            self._writer.append(obs, action, time.monotonic(), time.time())

        if self.monitor is not None:
            if not self._views:
                self._views = _views_of(obs)
            # set_status only on a publish tick; between ticks publish() is a
            # no-op and there is nothing new to report.
            if self.monitor.publish(obs, self._views):
                self.monitor.set_status(**self._status_dict())

        self._render()
        return None

    def _counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self._episodes:
            key = e.get("outcome", "unspecified")
            out[key] = out.get(key, 0) + 1
        return out

    def _status_dict(self) -> Dict[str, Any]:
        """Snapshot for the web monitor. Mirrors the terminal status line."""
        w = self._writer
        status: Dict[str, Any] = {
            "recording": w is not None,
            "task": self.task,
            "episodes": len(self._episodes),
            "counts": self._counts(),
            "free_gb": round(self._free_gb(), 1),
            "last": self._last_summary,
        }
        if w is not None:
            status.update(
                episode=w.dir.name,
                elapsed_s=round(w.duration, 1),
                frames=w.num_frames,
                hz=round((w.num_frames - 1) / w.duration, 1) if w.duration > 0 else 0.0,
                size=_fmt_bytes(w.bytes_written),
                queue=w.queue_depth,
                dropped=w.dropped,
            )
        return status

    # ---- shutdown ----------------------------------------------------------

    def close(self) -> None:
        """Finalise any in-progress take and restore the terminal.

        Must be reachable from the signal path: launch_yaml's signal_handler
        calls os._exit(0), which skips atexit entirely, so cleanup() has to call
        this explicitly.
        """
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            try:
                from gello.data_utils.episode_writer import OUTCOME_UNSPECIFIED

                self._stop(OUTCOME_UNSPECIFIED)
            except Exception as exc:
                print(f"\nError finalising episode: {exc}")
        self._clear_line()
        self.kb.restore()
        if self.monitor is not None:
            self.monitor.close()
        if self._episodes:
            counts = self._counts()
            breakdown = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            total = sum(e["num_frames"] for e in self._episodes)
            dropped = sum(e["dropped_frames"] for e in self._episodes)
            secs = sum(e["duration_s"] for e in self._episodes)
            size = sum(e["bytes_images"] for e in self._episodes)
            print(f"\nSession: {len(self._episodes)} episode(s) ({breakdown}), "
                  f"{total} frames, {secs:.1f}s, {_fmt_bytes(size)}"
                  + (f", {dropped} DROPPED" if dropped else ""))


def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # The save interface owns the status line. Two writers on one line clobber
    # each other, which is what the old "Time passed" / "Recording started"
    # interleaving did.
    if save_interface is not None:
        print_timing = False

    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart ...", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart ..."
    else:
        start_msg = "\nStart ..."

    print(start_msg)

    start_time = time.time()
    obs = env.get_obs()

    try:
        while True:
            if print_timing:
                num = time.time() - start_time
                message = f"\rTime passed: {round(num, 2)}          "
                if colors_available:
                    print(colored(message, color="white", attrs=["bold"]),
                          end="", flush=True)
                else:
                    print(message, end="", flush=True)

            action = agent.act(obs)

            if save_interface is not None:
                if save_interface.update(obs, action) == "quit":
                    break

            obs = env.step(action)
    finally:
        if save_interface is not None:
            save_interface.close()
