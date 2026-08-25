"""Background episode writer for teleop data collection.

Replaces the pickle-per-frame path in `format_obs.save_frame`, which wrote 3 MB
per frame (raw RGB + z16 depth for every camera) -- roughly 500 GB/hour with
three cameras. Here RGB is JPEG-encoded on worker threads and depth is dropped,
which is ~19x smaller and keeps encoding off the control loop.

Layout, one directory per episode:

    <root>/<task_slug>/<YYYYmmdd_HHMMSS>/
        meta.json          run provenance, camera serials, resolved configs
        state.npz          (T, ...) low-dim arrays, one key per obs field
        base/000000.jpg    one directory per camera view
        left_wrist/000000.jpg
        right_wrist/000000.jpg

Partial episodes stay readable: images land as they are encoded, and only
state.npz and meta.json are written at the end.
"""

import datetime
import json
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gello.data_utils.image_io import encode_jpeg, rgb_to_bgr

SCHEMA_VERSION = 2

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
#: A take finalised without a judgement -- quit or Ctrl-C mid-recording. Not the
#: same as a failure, and worth excluding from training until reviewed.
OUTCOME_UNSPECIFIED = "unspecified"

# Fields that are images rather than state. Depth is deliberately not recorded:
# pi0.5 consumes multi-view RGB + language + proprioception, never depth.
_IMAGE_SUFFIX = "_rgb"
_SKIP_SUFFIXES = ("_depth",)


def slugify(text: str, max_len: int = 48) -> str:
    """Filesystem-safe directory name. The verbatim task string lives in meta.json."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug[:max_len].rstrip("_") or "unlabeled")


def _git_info(path: Path) -> Dict[str, Any]:
    """Best-effort {sha, dirty} for the repo containing `path`."""

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(path), *args],
                capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    if sha is None:
        return {}
    status = run("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status)}


def provenance() -> Dict[str, Any]:
    """Record which code produced an episode.

    i2rt is tracked by git sha rather than version: it is installed from an
    editable checkout whose pyproject says 1.1.2 regardless of the release tag,
    so the version string cannot distinguish revisions.
    """
    info = {"gello": _git_info(Path(__file__).resolve().parents[2])}
    try:
        import i2rt

        info["i2rt"] = _git_info(Path(i2rt.__file__).resolve().parent)
    except Exception:
        info["i2rt"] = {}
    return info


class EpisodeWriter:
    """Accumulates one episode. Encoding and disk I/O happen off the caller's thread.

    `append` is called from the control loop and must stay far below the loop
    period, so it only converts colour space (which also copies the frame out of
    the recycled librealsense buffer) and enqueues.
    """

    def __init__(
        self,
        episode_dir: Path,
        task: str,
        views: List[str],
        meta_extra: Optional[Dict[str, Any]] = None,
        num_workers: int = 3,
        queue_size: int = 60,
        jpeg_quality: int = 95,
    ):
        self.dir = Path(episode_dir)
        self.task = task
        self.views = list(views)
        self.meta_extra = dict(meta_extra or {})
        self.jpeg_quality = int(jpeg_quality)

        self.dir.mkdir(parents=True, exist_ok=True)
        for view in self.views:
            (self.dir / view).mkdir(exist_ok=True)

        self._state: Dict[str, List[Any]] = {}
        self._n = 0
        self._dropped = 0
        self._dropped_idx: List[int] = []
        self._bytes = 0
        self._t0: Optional[float] = None
        self._t_last: Optional[float] = None

        self._counter_lock = threading.Lock()
        self._abort = threading.Event()
        self._closed = False
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._workers = [
            threading.Thread(target=self._worker, name=f"enc-{i}", daemon=True)
            for i in range(max(1, num_workers))
        ]
        for w in self._workers:
            w.start()

    # ---- control-loop side -------------------------------------------------

    def append(self, obs: Dict[str, Any], action: np.ndarray, t_mono: float,
               t_wall: float) -> None:
        if self._closed:
            raise RuntimeError("append() after finish()/abort()")

        frames = {}
        for key, value in obs.items():
            if key.endswith(_IMAGE_SUFFIX):
                view = key[: -len(_IMAGE_SUFFIX)]
                if view in self.views:
                    # Copies as well as reorders: read() returns a view onto the
                    # librealsense frame buffer, which is recycled once the
                    # frame is released, so a worker thread cannot be handed the
                    # original safely.
                    frames[view] = rgb_to_bgr(value)
            elif key.endswith(_SKIP_SUFFIXES):
                continue
            else:
                self._state.setdefault(key, []).append(value)

        self._state.setdefault("control", []).append(np.asarray(action))
        if self._t0 is None:
            self._t0 = t_mono
        self._state.setdefault("timestamp", []).append(t_mono - self._t0)
        self._state.setdefault("wall_time", []).append(t_wall)
        self._t_last = t_mono

        idx = self._n
        self._n += 1
        try:
            self._q.put_nowait((idx, frames))
        except queue.Full:
            # Never block the control loop. A full queue means encoding cannot
            # keep up, which the status line surfaces rather than hiding as jitter.
            with self._counter_lock:
                self._dropped += 1
                self._dropped_idx.append(idx)

    # ---- writer side -------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                if self._abort.is_set():
                    continue
                idx, frames = item
                written = 0
                for view, bgr in frames.items():
                    data = encode_jpeg(bgr, self.jpeg_quality)
                    (self.dir / view / f"{idx:06d}.jpg").write_bytes(data)
                    written += len(data)
                with self._counter_lock:
                    self._bytes += written
            except Exception as exc:  # a dead worker would stall the queue
                print(f"\n[EpisodeWriter] encode/write failed: {exc}")
            finally:
                self._q.task_done()

    def _shutdown_workers(self) -> None:
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            w.join(timeout=30)

    # ---- live stats for the status line -----------------------------------

    @property
    def num_frames(self) -> int:
        return self._n

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def bytes_written(self) -> int:
        with self._counter_lock:
            return self._bytes

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    @property
    def duration(self) -> float:
        if self._t0 is None or self._t_last is None:
            return 0.0
        return self._t_last - self._t0

    # ---- finalisation ------------------------------------------------------

    def finish(self, outcome: str = OUTCOME_UNSPECIFIED) -> Dict[str, Any]:
        """Drain the queue, then write state.npz and meta.json. Returns the meta.

        Args:
            outcome: whether the operator judged the take a success. Kept in
                meta.json rather than encoded in the path, so the converter can
                filter on it and a mislabelled take can be corrected by editing
                one file instead of moving a directory.
        """
        if self._closed:
            raise RuntimeError("already closed")
        self._closed = True
        self._shutdown_workers()

        arrays = {k: np.asarray(v) for k, v in self._state.items()}
        np.savez_compressed(self.dir / "state.npz", **arrays)

        duration = self.duration
        meta = {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "outcome": outcome,
            "episode_id": self.dir.name,
            "num_frames": self._n,
            "duration_s": round(duration, 3),
            "hz_effective": round((self._n - 1) / duration, 2) if duration > 0 else 0.0,
            "dropped_frames": self._dropped,
            "dropped_indices": self._dropped_idx,
            "views": self.views,
            "image_format": "jpg",
            "jpeg_quality": self.jpeg_quality,
            "bytes_images": self.bytes_written,
            "state_keys": {k: list(np.shape(v)) for k, v in arrays.items()},
            "action_key": "control",
            "created_at": datetime.datetime.now().isoformat(),
            "provenance": provenance(),
            # joint_velocities is finite-differenced with a hardcoded dt=0.01 in
            # YAMRobot.command_joint_state while the loop runs at ~30 Hz, so it
            # is ~3.3x too large; ee_pos_quat is an unimplemented FK placeholder
            # and is all zeros. Recorded as-is, flagged so nobody trusts them.
            "known_bad_fields": {
                "joint_velocities": "scaled by hardcoded dt=0.01, not the true loop dt",
                "ee_pos_quat": "unimplemented FK placeholder, always zeros",
            },
        }
        meta.update(self.meta_extra)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    def abort(self) -> None:
        """Discard the episode entirely, including anything already on disk."""
        if self._closed:
            return
        self._closed = True
        self._abort.set()
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass
        self._shutdown_workers()  # no writer can still be creating files
        shutil.rmtree(self.dir, ignore_errors=True)
