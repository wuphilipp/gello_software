"""Live multi-camera preview for tuning the teleop workspace.

Serves an MJPEG stream over HTTP rather than opening a window, because the
collection machine is normally reached over SSH with no DISPLAY. Point a browser
at http://<host>:8080/ (add `-L 8080:localhost:8080` to your ssh command if the
port is not directly reachable). Pass --display to use a local cv2 window
instead when you are sitting at the machine.

Each camera gets a row showing the native frame next to the 224x224 square a
pi0.5-style policy actually receives, in both squash and center-crop form.
Framing that reads fine at 640x480 can lose the manipulated object entirely at
224, so tune against those panes, not the native one.

    python scripts/preview_cameras.py --config configs/yam_left.yaml
"""

import argparse
import collections
import datetime
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gello.data_utils.image_io import (  # noqa: E402
    POLICY_VIEW_SIZE,
    encode_jpeg,
    rgb_to_bgr,
    to_policy_view,
)
from gello.utils.launch_utils import instantiate_from_dict  # noqa: E402

ROW_H = 336
FULL_W = 448
PANE = 336
LABEL_BG = (28, 28, 28)


class CameraReader:
    """Polls one camera in its own thread and caches the newest frame.

    RealSenseCamera.read() blocks in wait_for_frames(), so reading N cameras
    serially couples the preview rate to the slowest camera's phase. One thread
    each keeps the compositor non-blocking.
    """

    def __init__(self, name: str, camera):
        self.name = name
        self.camera = camera
        self.device_id = getattr(camera, "device_id", None) or "?"
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None  # BGR
        self._stamps = collections.deque(maxlen=30)
        self._error: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{name}", daemon=True
        )

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rgb, _depth = self.camera.read()
                # rgb_to_bgr copies: read() hands back a view onto the
                # librealsense buffer, which gets recycled under us.
                bgr = rgb_to_bgr(rgb)
            except Exception as exc:  # keep the other cameras alive
                with self._lock:
                    self._error = str(exc)
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame = bgr
                self._error = None
                self._stamps.append(time.monotonic())

    def snapshot(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            stamps = list(self._stamps)
            error = self._error
        fps = 0.0
        if len(stamps) > 1:
            span = stamps[-1] - stamps[0]
            if span > 0:
                fps = (len(stamps) - 1) / span
        return frame, fps, error

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        close = getattr(self.camera, "close", None)
        if close is not None:
            close()


def _label(img: np.ndarray, text: str, scale: float = 0.5) -> np.ndarray:
    """Draw a text label on a dark strip along the top of img (in place)."""
    h = 22
    cv2.rectangle(img, (0, 0), (img.shape[1], h), LABEL_BG, -1)
    cv2.putText(
        img, text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1,
        cv2.LINE_AA,
    )
    return img


def _placeholder(w: int, h: int, text: str) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(
        img, text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 220), 1,
        cv2.LINE_AA,
    )
    return img


def _build_row(reader: CameraReader) -> np.ndarray:
    frame, fps, error = reader.snapshot()
    if frame is None:
        msg = f"{reader.name}: {error or 'waiting for frames...'}"
        return _placeholder(FULL_W + 2 * PANE, ROW_H, msg)

    h, w = frame.shape[:2]

    full = cv2.resize(frame, (FULL_W, ROW_H), interpolation=cv2.INTER_AREA)
    # Outline what mode="crop" would keep, so framing decisions are visible.
    side = min(h, w)
    x0 = int((w - side) / 2 * FULL_W / w)
    x1 = int(((w - side) / 2 + side) * FULL_W / w)
    cv2.rectangle(full, (x0, 1), (x1 - 1, ROW_H - 2), (70, 200, 70), 1)
    _label(full, f"{reader.name}  sn={reader.device_id}  {w}x{h}  {fps:4.1f} Hz")

    panes = [full]
    for mode in ("squash", "crop"):
        small = to_policy_view(frame, POLICY_VIEW_SIZE, mode)
        # INTER_NEAREST on purpose: shows the real detail level rather than
        # smoothing it into looking sharper than the policy's input is.
        big = cv2.resize(small, (PANE, ROW_H), interpolation=cv2.INTER_NEAREST)
        _label(big, f"{POLICY_VIEW_SIZE}px {mode}")
        panes.append(big)

    return np.hstack(panes)


def build_composite(readers) -> np.ndarray:
    return np.vstack([_build_row(r) for r in readers])


PAGE = b"""<!doctype html><title>YAM camera preview</title>
<style>
 body{background:#141414;color:#ddd;font:13px system-ui,sans-serif;margin:0;padding:14px}
 img{max-width:100%;height:auto;display:block;border:1px solid #333}
 a{color:#7ab;margin-right:14px} .bar{margin:0 0 10px}
</style>
<div class=bar>
 <a href="/snapshot" target="_blank">save snapshot</a>
 <span>green box = what 224 crop keeps &middot; tune against the 224 panes</span>
</div>
<img src="/stream">
"""


def make_handler(readers, fps_limit, snapshot_dir):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the terminal clean
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream":
                self._stream()
            elif self.path == "/snapshot":
                self._snapshot()
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            period = 1.0 / fps_limit
            try:
                while True:
                    t0 = time.monotonic()
                    jpeg = encode_jpeg(build_composite(readers), 85)
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    slack = period - (time.monotonic() - t0)
                    if slack > 0:
                        time.sleep(slack)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser navigated away

        def _snapshot(self):
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            written = []
            for r in readers:
                frame, _fps, _err = r.snapshot()
                if frame is None:
                    continue
                path = snapshot_dir / f"{stamp}_{r.name}.jpg"
                path.write_bytes(encode_jpeg(frame, 95))
                written.append(str(path))
            body = ("saved:\n" + "\n".join(written) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def build_readers(config_path: str) -> Dict[str, CameraReader]:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    cam_cfg = cfg.get("cameras")
    if not cam_cfg:
        raise SystemExit(
            f"No 'cameras:' block in {config_path}. Add one, e.g.\n"
            "  cameras:\n"
            "    base:\n"
            "      _target_: gello.cameras.realsense_camera.RealSenseCamera\n"
            '      device_id: "213322071600"'
        )
    print(f"Opening {len(cam_cfg)} camera(s): {', '.join(cam_cfg)}")
    cameras = instantiate_from_dict(cam_cfg)
    readers = [CameraReader(name, cam) for name, cam in cameras.items()]
    for r in readers:
        r.start()
    return readers


def run_display(readers, fps_limit):
    print("Showing cv2 window; press q or Esc to quit.")
    period = 1.0 / fps_limit
    while True:
        t0 = time.monotonic()
        cv2.imshow("YAM camera preview", build_composite(readers))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
        slack = period - (time.monotonic() - t0)
        if slack > 0:
            time.sleep(slack)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yam_left.yaml")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=float, default=15.0, help="preview render rate")
    ap.add_argument("--display", action="store_true", help="cv2 window instead of HTTP")
    ap.add_argument("--snapshot-dir", default="images/preview")
    args = ap.parse_args()

    readers = build_readers(args.config)
    try:
        if args.display:
            run_display(readers, args.fps)
            return
        handler = make_handler(readers, args.fps, Path(args.snapshot_dir))
        server = ThreadingHTTPServer((args.host, args.port), handler)
        server.daemon_threads = True
        print(f"\nPreview at http://localhost:{args.port}/   (Ctrl-C to stop)")
        print(f"  remote:  ssh -L {args.port}:localhost:{args.port} <this-host>\n")
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        stop.wait()
        server.shutdown()
        server.server_close()
    finally:
        print("Closing cameras...")
        for r in readers:
            r.stop()


if __name__ == "__main__":
    main()
