"""Inspect one recorded episode: timing, fields, action ranges, video preview.

Handles both on-disk layouts:

  new     <ep>/meta.json + state.npz + <view>/000000.jpg
  legacy  <ep>/<iso-timestamp>.pkl   (raw pickle per frame, ~3 MB each)

The legacy reader is kept so episodes recorded before the writer rewrite stay
inspectable rather than being orphaned.

    python scripts/inspect_episode.py data/pick_red_block/20260821_143012
    python scripts/inspect_episode.py <ep> --no-video      # stats only, fast
"""

import argparse
import glob
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- shared ----

def report_timing(times: np.ndarray, label: str = "Timing") -> None:
    """Timing histogram. This is what makes control-loop regressions visible."""
    if len(times) < 2:
        print(f"\n{label}: too few frames")
        return
    dt = np.diff(times)
    duration = times[-1] - times[0]
    print(f"\n{label}")
    print(f"  frames:       {len(times)}")
    print(f"  duration:     {duration:.2f} s")
    print(f"  effective:    {(len(times) - 1) / duration:.2f} Hz")
    print(f"  median dt:    {np.median(dt) * 1000:.1f} ms")
    print(f"  p95 dt:       {np.percentile(dt, 95) * 1000:.1f} ms")
    print(f"  max dt:       {np.max(dt) * 1000:.1f} ms")
    print(f"  >50 ms gaps:  {int(np.sum(dt > 0.05))}")
    print(f"  >100 ms gaps: {int(np.sum(dt > 0.10))}")


def report_actions(actions: np.ndarray) -> None:
    print("\nActions")
    print(f"  shape: {actions.shape}")
    print(f"  min:   {np.round(actions.min(axis=0), 3)}")
    print(f"  max:   {np.round(actions.max(axis=0), 3)}")
    print(f"  range: {np.round(np.ptp(actions, axis=0), 3)}")


def resize_height(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


def write_video(out: str, frames_iter, n: int, times: np.ndarray, fps: float) -> None:
    """frames_iter yields BGR frames, all the same size."""
    writer = None
    for i, frame in enumerate(frames_iter):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
        label = f"t={times[i]:.2f}s  frame={i}" if i < len(times) else f"frame={i}"
        cv2.putText(frame, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        writer.write(frame)
    if writer is not None:
        writer.release()
        print(f"\nPreview written to: {out}")


# ------------------------------------------------------------- new layout ----

def inspect_new(ep: Path, args) -> None:
    meta = json.loads((ep / "meta.json").read_text())
    state = np.load(ep / "state.npz")

    print(f"\nEpisode: {ep}  (schema v{meta.get('schema_version')})")
    print(f"  task:       {meta.get('task')!r}")
    outcome = meta.get("outcome", "unspecified")
    note = "  <- never judged; review before training" if outcome == "unspecified" else ""
    print(f"  outcome:    {outcome}{note}")
    print(f"  recorded:   {meta.get('created_at')}")
    print(f"  agent:      {meta.get('agent')}   robot: {meta.get('robot')}")
    print(f"  hz target:  {meta.get('hz_target')}  "
          f"effective: {meta.get('hz_effective')}")
    print(f"  images:     {meta.get('image_format')} q{meta.get('jpeg_quality')}, "
          f"{meta.get('bytes_images', 0) / 1e6:.1f} MB")
    cams = meta.get("cameras") or {}
    for name, sn in cams.items():
        print(f"    {name:14s} sn={sn}")

    dropped = meta.get("dropped_frames", 0)
    if dropped:
        print(f"  DROPPED:    {dropped} frames -- encoder could not keep up")
    else:
        print("  dropped:    0")

    bad = meta.get("known_bad_fields") or {}
    if bad:
        print("  do not trust:")
        for k, why in bad.items():
            print(f"    {k}: {why}")

    print("\nFields in state.npz")
    for k in sorted(state.files):
        v = state[k]
        print(f"  {k:20s} shape={str(v.shape):14s} dtype={v.dtype}")

    times = state["timestamp"] if "timestamp" in state.files else np.array([])
    report_timing(times, "Timing (control loop)")

    # Per-camera capture stamps reveal how much the serial camera reads in
    # get_obs() spread one observation out in time.
    stamp_keys = sorted(k for k in state.files if k.endswith("_stamp"))
    if len(stamp_keys) > 1:
        stamps = np.stack([state[k] for k in stamp_keys], axis=1)
        spread = (stamps.max(axis=1) - stamps.min(axis=1)) * 1000
        print(f"\nCamera capture spread within one observation "
              f"({len(stamp_keys)} cams)")
        print(f"  median: {np.median(spread):.1f} ms   "
              f"p95: {np.percentile(spread, 95):.1f} ms   max: {spread.max():.1f} ms")
        print("  (large values mean views in a frame are not simultaneous)")

    if "control" in state.files:
        report_actions(state["control"])

    views = meta.get("views") or [d.name for d in sorted(ep.iterdir()) if d.is_dir()]
    print(f"\nViews: {views}")
    if args.no_video or not views:
        return

    per_view = {v: sorted((ep / v).glob("*.jpg")) for v in views}
    n = min((len(f) for f in per_view.values()), default=0)
    if n == 0:
        print("No images found; skipping video.")
        return

    first = [cv2.imread(str(per_view[v][0])) for v in views]
    target_h = min(img.shape[0] for img in first)
    dt = np.diff(times) if len(times) > 1 else np.array([])
    fps = min(60.0, 1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else 30.0

    def frames():
        for i in range(n):
            # imread returns BGR, which is what VideoWriter wants -- no flip.
            imgs = [resize_height(cv2.imread(str(per_view[v][i])), target_h)
                    for v in views]
            yield np.hstack(imgs)

    write_video(args.out, frames(), n, times, fps)


# ---------------------------------------------------------- legacy layout ----

def inspect_legacy(ep: Path, args) -> None:
    files = sorted(glob.glob(str(ep / "*.pkl")))
    print(f"\nEpisode: {ep}  (legacy pickle-per-frame layout)")

    stamps = [datetime.fromisoformat(Path(f).stem) for f in files]
    times = np.array([(t - stamps[0]).total_seconds() for t in stamps])
    report_timing(times)

    with open(files[0], "rb") as f:
        sample = pickle.load(f)
    print("\nFields")
    for k, v in sample.items():
        print(f"  {k:20s} shape={str(getattr(v, 'shape', None)):14s} "
              f"dtype={getattr(v, 'dtype', None)}")

    per_frame = Path(files[0]).stat().st_size
    print(f"\n  {per_frame / 1e6:.2f} MB/frame -> "
          f"{per_frame * 30 / 1e6:.0f} MB/s at 30 Hz")

    views = [k[:-4] for k in sample if k.endswith("_rgb")]
    print(f"\nCameras: {views}")
    if args.no_video or not views:
        return

    target_h = min(sample[f"{v}_rgb"].shape[0] for v in views)
    dt = np.diff(times)
    fps = min(60.0, 1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else 30.0

    actions = []

    def frames():
        for path in files:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            if "control" in d:
                actions.append(np.asarray(d["control"]))
            # Legacy pickles store RGB; VideoWriter wants BGR.
            imgs = [resize_height(d[f"{v}_rgb"][:, :, ::-1].copy(), target_h)
                    for v in views]
            yield np.hstack(imgs)

    write_video(args.out, frames(), len(files), times, fps)
    if actions:
        report_actions(np.stack(actions))


# ---------------------------------------------------------------- driver ----

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode", help="Path to one recorded episode directory")
    ap.add_argument("--out", default="/tmp/gello_episode_preview.mp4")
    ap.add_argument("--no-video", action="store_true",
                    help="print stats only, skip decoding frames")
    args = ap.parse_args()

    ep = Path(args.episode)
    if not ep.is_dir():
        sys.exit(f"Not a directory: {ep}")

    if (ep / "meta.json").exists():
        inspect_new(ep, args)
    elif list(ep.glob("*.pkl")):
        inspect_legacy(ep, args)
    else:
        sys.exit(f"No meta.json and no *.pkl in {ep} -- not a recognised episode")


if __name__ == "__main__":
    main()
