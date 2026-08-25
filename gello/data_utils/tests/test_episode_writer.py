"""Tests for EpisodeWriter. Runs without hardware and without pytest installed."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from gello.data_utils.episode_writer import EpisodeWriter, slugify

VIEWS = ["base", "left_wrist", "right_wrist"]


def fake_obs(i: int):
    """Mimics RobotEnv.get_obs() for a bimanual YAM with three cameras."""
    obs = {}
    for v in VIEWS:
        obs[f"{v}_rgb"] = np.full((480, 640, 3), i % 256, np.uint8)
        obs[f"{v}_depth"] = np.zeros((480, 640, 1), np.uint16)  # must be dropped
        obs[f"{v}_stamp"] = 1000.0 + i * 0.033
    obs["joint_positions"] = np.arange(14, dtype=np.float64) + i
    obs["joint_velocities"] = np.zeros(14)
    obs["ee_pos_quat"] = np.zeros(14)
    obs["gripper_position"] = np.array([0.1, 0.2])
    return obs


def _record(root: Path, n: int, hz: float = 30.0, **kw) -> EpisodeWriter:
    """Record n frames. hz=0 means append as fast as possible (saturation test)."""
    w = EpisodeWriter(root / "ep", task="pick up the red block", views=VIEWS, **kw)
    period = 1.0 / hz if hz else 0.0
    start = time.perf_counter()
    for i in range(n):
        if period:
            target = start + i * period
            while time.perf_counter() < target:
                time.sleep(0.0005)
        w.append(fake_obs(i), np.full(14, float(i)), t_mono=i / 30.0, t_wall=1e9 + i / 30.0)
    return w


def test_writes_full_episode():
    n = 60  # paced at 30 Hz -> a 2 s test
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w = _record(root, n)
        meta = w.finish("success")

        for v in VIEWS:
            files = sorted((root / "ep" / v).glob("*.jpg"))
            assert len(files) == n, f"{v}: {len(files)} jpgs, expected {n}"
        assert files[0].name == "000000.jpg" and files[-1].name == f"{n-1:06d}.jpg"

        state = np.load(root / "ep" / "state.npz")
        assert state["joint_positions"].shape == (n, 14)
        assert state["control"].shape == (n, 14)
        assert state["gripper_position"].shape == (n, 2)
        assert state["timestamp"].shape == (n,)
        assert state["wall_time"].shape == (n,)
        assert state["base_stamp"].shape == (n,)
        assert not any("depth" in k for k in state.files), "depth must not be recorded"
        assert np.isclose(state["timestamp"][0], 0.0)
        assert np.allclose(state["control"][7], 7.0)

        on_disk = json.loads((root / "ep" / "meta.json").read_text())
        assert on_disk == json.loads(json.dumps(meta, default=str))
        assert on_disk["num_frames"] == n
        assert on_disk["dropped_frames"] == 0
        assert on_disk["task"] == "pick up the red block"
        assert on_disk["outcome"] == "success"
        assert on_disk["views"] == VIEWS
        assert abs(on_disk["hz_effective"] - 30.0) < 0.5
        assert "gello" in on_disk["provenance"]
        assert "joint_velocities" in on_disk["known_bad_fields"]
    print("test_writes_full_episode OK")


def test_abort_removes_everything():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w = _record(root, 40)
        assert (root / "ep").exists()
        w.abort()
        assert not (root / "ep").exists(), "abort() must remove the episode directory"
    print("test_abort_removes_everything OK")


def test_append_is_cheap():
    """append() runs inside the 33 ms control loop, so it must stay well under it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w = EpisodeWriter(root / "ep", task="t", views=VIEWS)
        obs, act = fake_obs(0), np.zeros(14)
        worst = 0.0
        for i in range(90):
            t0 = time.perf_counter()
            w.append(obs, act, i / 30.0, 1e9 + i / 30.0)
            worst = max(worst, time.perf_counter() - t0)
            time.sleep(1 / 30.0)
        w.finish()
        print(f"  worst append(): {worst*1000:.2f} ms of a 33.3 ms budget")
        assert worst < 0.010, f"append() took {worst*1000:.1f} ms"
    print("test_append_is_cheap OK")


def test_drop_accounting_is_exact_under_saturation():
    """A saturated queue must drop visibly, never lose frames silently.

    Appending with no pacing outruns any encoder pool by design; what matters is
    that dropped_frames accounts for every missing file, so the status line and
    meta.json can be trusted.
    """
    n = 300
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w = _record(root, n, hz=0)
        meta = w.finish()
        dropped = meta["dropped_frames"]
        assert dropped > 0, "expected saturation; raise n if this pool got faster"
        assert len(meta["dropped_indices"]) == dropped
        for v in VIEWS:
            written = len(list((root / "ep" / v).glob("*.jpg")))
            assert written + dropped == n, (
                f"{v}: {written} written + {dropped} dropped != {n}"
            )
        # State is accumulated in RAM, so it is complete even when images drop.
        state = np.load(root / "ep" / "state.npz")
        assert state["control"].shape == (n, 14)
        print(f"  dropped {dropped}/{n} under an unpaced burst, accounting exact")
    print("test_drop_accounting_is_exact_under_saturation OK")


def test_encoder_throughput_headroom():
    """How far above 30 Hz the encoder pool sustains, with three views."""
    n = 240
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w = EpisodeWriter(root / "ep", task="t", views=VIEWS, queue_size=n + 10)
        obs, act = fake_obs(0), np.zeros(14)
        for i in range(n):
            w.append(obs, act, i / 30.0, 1e9 + i / 30.0)
        t0 = time.perf_counter()
        meta = w.finish()  # blocks until the queue drains
        elapsed = time.perf_counter() - t0
        fps = n / elapsed
        print(f"  drained {n} frames x {len(VIEWS)} views in {elapsed:.2f}s "
              f"-> {fps:.0f} frames/s ({fps/30:.1f}x the 30 Hz loop)")
        assert meta["dropped_frames"] == 0
        assert fps > 60, f"only {fps:.0f} frames/s of encode throughput"
    print("test_encoder_throughput_headroom OK")


def test_slugify():
    assert slugify("Pick up the RED block!") == "pick_up_the_red_block"
    assert slugify("  ") == "unlabeled"
    assert slugify("a" * 200) == "a" * 48
    print("test_slugify OK")


if __name__ == "__main__":
    test_slugify()
    test_writes_full_episode()
    test_abort_removes_everything()
    test_append_is_cheap()
    test_drop_accounting_is_exact_under_saturation()
    test_encoder_throughput_headroom()
    print("\nall tests passed")
