"""Tests for the live web monitor. No hardware, no robot.

Every server binds port 0 so the OS picks a free one. Hard-coding ports made
these collide with a real teleop session running on the documented default.
"""

import json
import time
import urllib.request

import numpy as np

from gello.data_utils.monitor import MonitorServer

VIEWS = ["top", "left_wrist", "right_wrist"]


def fake_obs():
    return {f"{v}_rgb": np.full((480, 640, 3), 90, np.uint8) for v in VIEWS}


def server(**kw):
    return MonitorServer(port=0, **kw)


def get(path, port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.read()


def first_jpeg(port, path):
    """Pull one frame out of an MJPEG multipart stream."""
    r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
    try:
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += r.read(256)
        head, rest = buf.split(b"\r\n\r\n", 1)
        assert b"image/jpeg" in head, head
        n = int([l for l in head.split(b"\r\n")
                 if b"Content-Length" in l][0].split(b":")[1])
        while len(rest) < n:
            rest += r.read(n - len(rest))
        return rest[:n]
    finally:
        r.close()


def test_binds_an_ephemeral_port():
    m = server(fps=20)
    try:
        assert m.port > 0 and str(m.port) in m.url
    finally:
        m.close()
    print("test_binds_an_ephemeral_port OK")


def test_serves_page_status_and_stream():
    m = server(fps=20)
    try:
        m.publish(fake_obs(), VIEWS)
        m.set_status(recording=True, task="zip tie", frames=42, hz=29.8,
                     size="10.0MB", queue=0, dropped=0, episodes=1, free_gb=391.0)

        assert b"YAM teleop monitor" in get("/", m.port)

        s = json.loads(get("/status", m.port))
        assert s["recording"] is True and s["task"] == "zip tie"
        assert s["frames"] == 42 and s["dropped"] == 0

        assert first_jpeg(m.port, "/stream")[:2] == b"\xff\xd8"
    finally:
        m.close()
    print("test_serves_page_status_and_stream OK")


def test_per_view_streams_and_view_list():
    """The page builds its grid from status.views and one stream per view."""
    m = server(fps=20)
    try:
        m.publish(fake_obs(), VIEWS)
        s = json.loads(get("/status", m.port))
        assert s["views"] == VIEWS, f"views should keep config order, got {s['views']}"
        for view in VIEWS:
            assert first_jpeg(m.port, f"/stream/{view}")[:2] == b"\xff\xd8", view
        # an unknown view still streams a placeholder rather than erroring
        assert first_jpeg(m.port, "/stream/nope")[:2] == b"\xff\xd8"
    finally:
        m.close()
    print("test_per_view_streams_and_view_list OK")


def test_publish_is_throttled_and_cheap():
    """publish() runs in the control loop, so between ticks it must be a no-op."""
    m = server(fps=10)
    try:
        obs = fake_obs()
        assert m.publish(obs, VIEWS) is True, "first call should take frames"
        taken = sum(m.publish(obs, VIEWS) for _ in range(50))
        assert taken == 0, f"{taken} publishes inside one 100 ms window"

        worst = 0.0
        for _ in range(300):
            t0 = time.perf_counter()
            m.publish(obs, VIEWS)
            worst = max(worst, time.perf_counter() - t0)
        print(f"  worst publish(): {worst*1000:.2f} ms of a 33.3 ms budget")
        assert worst < 0.010
    finally:
        m.close()
    print("test_publish_is_throttled_and_cheap OK")


def test_survives_bad_input():
    """A monitor failure must never take the control loop down."""
    m = server(fps=100)
    try:
        # No cameras configured: the tick still fires so status keeps updating.
        assert m.publish({}, VIEWS) is True
        time.sleep(0.02)
        assert m.publish({"top_rgb": None}, ["top"]) is True
        time.sleep(0.02)
        # Something genuinely broken is swallowed, not raised at the caller.
        assert m.publish({"top_rgb": "not an array"}, ["top"]) is False
        time.sleep(0.02)
        assert m.publish(fake_obs(), VIEWS) is True    # recovers
        json.loads(get("/status", m.port))             # still serving
    finally:
        m.close()
    print("test_survives_bad_input OK")


def test_stream_works_before_any_frame():
    m = server(fps=20)
    try:
        assert first_jpeg(m.port, "/stream")[:2] == b"\xff\xd8"
    finally:
        m.close()
    print("test_stream_works_before_any_frame OK")


def test_port_in_use_is_reported_not_swallowed():
    """SaveInterface catches this to keep a run alive; it must be an OSError."""
    a = server(fps=20)
    try:
        try:
            MonitorServer(port=a.port)
        except OSError:
            pass
        else:
            raise AssertionError("expected OSError for a port already in use")
    finally:
        a.close()
    print("test_port_in_use_is_reported_not_swallowed OK")


if __name__ == "__main__":
    test_binds_an_ephemeral_port()
    test_serves_page_status_and_stream()
    test_per_view_streams_and_view_list()
    test_publish_is_throttled_and_cheap()
    test_survives_bad_input()
    test_stream_works_before_any_frame()
    test_port_in_use_is_reported_not_swallowed()
    print("\nall tests passed")
