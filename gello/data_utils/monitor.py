"""Live web monitor for a teleop recording session.

Runs inside the teleop process, because the RealSense devices are held
exclusively -- a separate viewer process cannot open them. The control loop
hands frames and status over cheaply; all JPEG encoding happens on the HTTP
threads, so the monitor cannot slow the loop down.

    python experiments/launch_yaml.py ... --monitor-port 8081
    # then open http://<host>:8081/ on a monitor

Shows whether a take is recording, its task and episode id, elapsed time, frame
count, effective rate, size on disk, encoder queue depth and dropped frames.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from gello.data_utils.image_io import encode_jpeg

PAGE = b"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>YAM teleop monitor</title>
<style>
 :root{--bg:#0d0d10;--fg:#e8e8ee;--dim:#8a8a99;--rec:#e0263c;--idle:#3a3a46}
 *{box-sizing:border-box} html,body{margin:0;height:100%;overflow:hidden}
 body{background:var(--bg);color:var(--fg);font:16px/1.3 system-ui,sans-serif;
      display:flex;flex-direction:column}
 #bar{display:flex;align-items:center;gap:28px;padding:16px 24px;
      background:var(--idle);transition:background .2s;flex:0 0 auto}
 #bar.rec{background:var(--rec)}
 #dot{width:36px;height:36px;border-radius:50%;background:#fff;opacity:.35}
 #bar.rec #dot{opacity:1;animation:p 1s infinite}
 @keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
 #state{font-size:48px;font-weight:700;letter-spacing:.04em;min-width:280px}
 #task{font-size:48px;font-weight:600;flex:1;overflow:hidden;
       text-overflow:ellipsis;white-space:nowrap}
 .stat{text-align:right;min-width:164px}
 .stat b{display:block;font-size:44px;font-variant-numeric:tabular-nums}
 .stat span{font-size:20px;color:rgba(255,255,255,.75);text-transform:uppercase;
            letter-spacing:.08em}
 #full{background:rgba(255,255,255,.15);border:0;color:#fff;border-radius:8px;
       padding:14px 20px;cursor:pointer;font-size:28px;line-height:1}
 #warn{background:#7a1020;color:#fff;padding:12px 24px;font-weight:600;
       font-size:32px;
       display:none;flex:0 0 auto}
 /* the video area takes every pixel left over, and each tile fills its cell */
 #grid{flex:1 1 auto;min-height:0;display:grid;gap:2px;background:#000;
       grid-auto-rows:1fr}
 .cell{position:relative;min-height:0;min-width:0;overflow:hidden}
 .cell img{width:100%;height:100%;object-fit:cover;display:block}
 .cell b{position:absolute;left:0;top:0;padding:6px 16px;font-size:26px;
         font-weight:600;background:rgba(0,0,0,.6);color:#fff;
         border-bottom-right-radius:6px}
 #foot{padding:12px 24px;color:var(--dim);font-size:24px;display:flex;gap:32px;
       flex:0 0 auto}
 body.zen #bar,body.zen #foot{display:none}
</style>
<div id=bar>
  <div id=dot></div><div id=state>--</div><div id=task></div>
  <div class=stat><b id=t>0.0s</b><span>elapsed</span></div>
  <div class=stat><b id=n>0</b><span>frames</span></div>
  <div class=stat><b id=hz>0.0</b><span>Hz</span></div>
  <div class=stat><b id=sz>0</b><span>size</span></div>
  <button id=full title="fullscreen (f) - double-click video to hide chrome">&#9974;</button>
</div>
<div id=warn></div>
<div id=grid></div>
<div id=foot>
  <span id=ep></span><span id=eps></span><span id=q></span>
  <span id=disk></span><span id=last></span>
</div>
<script>
const $ = i => document.getElementById(i);
let built = '';
function layout(views){
  const key = views.join(',');
  if (key === built) return;
  built = key;
  const g = $('grid');
  g.innerHTML = '';
  // choose the column count whose resulting cell aspect best matches the screen
  const n = views.length;
  let best = 1, bestErr = Infinity;
  const target = (window.innerWidth / Math.max(1, window.innerHeight)) ;
  for (let c = 1; c <= n; c++){
    const r = Math.ceil(n / c);
    const cell = (window.innerWidth / c) / (window.innerHeight / r);
    const err = Math.abs(Math.log(cell / (4/3)));   // cameras are 4:3
    if (err < bestErr){ bestErr = err; best = c; }
  }
  g.style.gridTemplateColumns = `repeat(${best}, 1fr)`;
  for (const v of views){
    const d = document.createElement('div');
    d.className = 'cell';
    d.innerHTML = `<img src="/stream/${encodeURIComponent(v)}" alt="${v}"><b>${v}</b>`;
    g.appendChild(d);
  }
}
async function tick(){
  try{
    const s = await (await fetch('/status',{cache:'no-store'})).json();
    if (s.views) layout(s.views);
    $('bar').classList.toggle('rec', s.recording);
    $('state').textContent = s.recording ? 'RECORDING' : 'IDLE';
    $('task').textContent  = s.task || '';
    $('t').textContent  = (s.elapsed_s||0).toFixed(1)+'s';
    $('n').textContent  = s.frames||0;
    $('hz').textContent = (s.hz||0).toFixed(1);
    $('sz').textContent = s.size||'0';
    $('ep').textContent   = s.episode ? 'episode: '+s.episode : '';
    const c = s.counts || {};
    const bits = [];
    if (c.success)     bits.push(c.success+' ok');
    if (c.failure)     bits.push(c.failure+' fail');
    if (c.unspecified) bits.push(c.unspecified+' unjudged');
    $('eps').textContent = 'saved: '+(s.episodes||0)
      + (bits.length ? ' ('+bits.join(', ')+')' : '');
    $('q').textContent    = 'queue: '+(s.queue||0);
    $('disk').textContent = 'free: '+(s.free_gb||0).toFixed(0)+' GB';
    $('last').textContent = s.last || '';
    const w = [];
    if (s.dropped) w.push(s.dropped+' DROPPED FRAMES');
    if (s.free_gb !== undefined && s.free_gb < 20) w.push('LOW DISK');
    $('warn').textContent = w.join('  -  ');
    $('warn').style.display = w.length ? 'block' : 'none';
  }catch(e){ $('state').textContent = 'DISCONNECTED'; }
}
$('full').onclick = () => document.fullscreenElement
  ? document.exitFullscreen() : document.documentElement.requestFullscreen();
$('grid').ondblclick = () => document.body.classList.toggle('zen');
addEventListener('keydown', e => {
  if (e.key === 'f') $('full').click();
  if (e.key === 'z') document.body.classList.toggle('zen');
});
addEventListener('resize', () => { built = ''; });
tick(); setInterval(tick, 400);
</script>
"""


class MonitorServer:
    """Serves an MJPEG camera view plus recording status over HTTP."""

    def __init__(self, port: int = 8081, host: str = "0.0.0.0",
                 fps: float = 10.0, tile_height: int = 260):
        self.port, self.host = port, host
        self.fps, self.tile_height = fps, tile_height

        self._lock = threading.Lock()
        self._tiles: Dict[str, np.ndarray] = {}  # view -> downscaled BGR
        self._status: Dict[str, Any] = {"recording": False}
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._last_publish = 0.0

        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        # Pass port=0 to get any free port; record what was actually bound so
        # the caller can print a usable URL.
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="monitor", daemon=True
        )
        self._thread.start()

    # ---- control-loop side (must stay cheap) -------------------------------

    def publish(self, obs: Dict[str, Any], views: List[str]) -> bool:
        """Offer the newest frames to the monitor.

        Returns True when a monitor tick fired -- which the caller uses to
        decide whether to refresh the status too. That deliberately does not
        mean frames were found: a state-only run with no cameras configured
        still needs its status to keep updating. False means either throttled
        or something went wrong.

        Throttled to the monitor's rate and a no-op in between. What little
        work remains is deliberately the cheapest that still copies the frame
        out of the recycled librealsense buffer: INTER_NEAREST downscaling runs
        ~17x faster than INTER_AREA (0.24 ms vs 4.0 ms for three views) and the
        difference is invisible on a preview. Labelling and compositing happen
        on the HTTP thread, not here.
        """
        now = time.monotonic()
        if now - self._last_publish < self._period:
            return False
        self._last_publish = now
        try:
            tiles = {}
            h = self.tile_height
            for view in views:
                rgb = obs.get(f"{view}_rgb")
                if rgb is None:
                    continue
                w = max(1, int(rgb.shape[1] * h / rgb.shape[0]))
                small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
                tiles[view] = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
            if tiles:
                with self._lock:
                    self._tiles = tiles
            return True
        except Exception:
            return False  # the monitor must never take the control loop down

    def _composite(self) -> Optional[np.ndarray]:
        """Label and stack the latest tiles, in the order publish() supplied.

        Runs on the HTTP thread, not the control loop.
        """
        with self._lock:
            tiles = self._tiles
        if not tiles:
            return None
        out = []
        for view, tile in tiles.items():
            tile = tile.copy()
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 20), (25, 25, 25), -1)
            cv2.putText(tile, view, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1, cv2.LINE_AA)
            out.append(tile)
        return np.hstack(out)

    def set_status(self, **status: Any) -> None:
        with self._lock:
            self._status = status

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    # ---- HTTP --------------------------------------------------------------

    def _tile(self, view: str) -> Optional[np.ndarray]:
        """One camera's latest frame, for a per-view stream."""
        with self._lock:
            return self._tiles.get(view)

    def _snapshot(self):
        with self._lock:
            status = dict(self._status)
        return self._composite(), status

    def _handler(self):
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass  # keep the status line clean

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(PAGE, "text/html; charset=utf-8")
                elif self.path.startswith("/status"):
                    with monitor._lock:
                        status = dict(monitor._status)
                        status["views"] = list(monitor._tiles)
                    self._send(json.dumps(status).encode(), "application/json")
                elif self.path.startswith("/stream/"):
                    from urllib.parse import unquote
                    self._stream(unquote(self.path[len("/stream/"):]))
                elif self.path.startswith("/stream"):
                    self._stream(None)
                else:
                    self.send_error(404)

            def _stream(self, view):
                self.send_response(200)
                self.send_header("Cache-Control", "no-store, private")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                blank = np.zeros((monitor.tile_height, 640, 3), np.uint8)
                cv2.putText(blank, "waiting for frames...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 140), 1)
                try:
                    while True:
                        t0 = time.monotonic()
                        frame = (monitor._tile(view) if view is not None
                                 else monitor._composite())
                        jpeg = encode_jpeg(blank if frame is None else frame, 80)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        slack = monitor._period - (time.monotonic() - t0)
                        if slack > 0:
                            time.sleep(slack)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # viewer closed the tab

        return Handler
