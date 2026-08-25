"""Find exposure/gain that hold detail, and print config to paste in.

Auto-exposure meters the whole frame, so a workspace dominated by white paper
gets exposed for the average and the paper itself blows out -- measured here at
48-56% of pixels clipped on the wrist cameras, taking the printed harness
diagram with it. Shorter exposure also cuts motion blur, which matters most on
a wrist camera that is being swung around.

Values are specific to the lighting and camera poses, so re-run this after
moving a camera or changing the lights.

    python scripts/tune_exposure.py --config configs/yam_left.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Clipping destroys the printed diagram, so it is the hard limit. Dark pixels
# are mostly the black gripper fingers, which are legitimately black -- being
# strict there just rejects good settings.
MAX_CLIP = 3.0     # percent of pixels at/above 250
MAX_DARK = 15.0    # percent at/below 5
MIN_MEAN = 85.0    # below this the frame is genuinely underexposed


def sweep_camera(name, serial, settle=15):
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    sensor = next(
        s for s in profile.get_device().query_sensors()
        if any(q.stream_type() == rs.stream.color for q in s.get_stream_profiles())
    )
    try:
        e_rng = sensor.get_option_range(rs.option.exposure)
        g_rng = sensor.get_option_range(rs.option.gain)
        sensor.set_option(rs.option.enable_auto_exposure, 0)

        # Higher gain buys a shorter exposure at the same brightness, and
        # exposure time is what smears a moving wrist camera, so sweep both and
        # then prefer the shortest exposure that is still properly exposed.
        gains = sorted({int(round(g)) for g in
                        (g_rng.min, max(g_rng.min, 1) * 2, max(g_rng.min, 1) * 4)
                        if g <= g_rng.max})
        hi = min(e_rng.max, 40000)
        exposures = sorted({int(round(v))
                            for v in np.geomspace(max(e_rng.min, 20), hi, 12)})

        print(f"\n{name}  (sn={serial})")
        print(f"  {'gain':>5} {'exposure':>9} {'mean':>7} {'clip>250':>9} {'dark<5':>8}")
        results = []
        for gain in gains:
            sensor.set_option(rs.option.gain, gain)
            for exp in exposures:
                sensor.set_option(rs.option.exposure, exp)
                for _ in range(settle):
                    frame = pipe.wait_for_frames().get_color_frame()
                a = np.asanyarray(frame.get_data())
                clip = (a > 250).mean() * 100
                dark = (a < 5).mean() * 100
                mean = a.mean()
                ok = clip <= MAX_CLIP and dark <= MAX_DARK and mean >= MIN_MEAN
                if ok or clip <= MAX_CLIP:
                    print(f"  {gain:>5} {exp:>9} {mean:7.1f} {clip:8.2f}% {dark:7.2f}%"
                          + ("   ok" if ok else ""))
                if ok:
                    results.append((exp, gain, mean, clip, dark))
        if not results:
            print("  nothing met the limits -- adjust the lighting")
            return None
        exp, gain, mean, clip, dark = min(results)  # shortest exposure wins
        print(f"  -> best: exposure={exp}us gain={gain} "
              f"(mean {mean:.0f}, clip {clip:.2f}%, dark {dark:.1f}%)")
        return exp, int(gain)
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yam_left.yaml")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cams = cfg.get("cameras") or {}
    if not cams:
        raise SystemExit(f"No 'cameras:' block in {args.config}")

    chosen = {}
    for name, c in cams.items():
        got = sweep_camera(name, c["device_id"])
        if got:
            chosen[name] = got

    if not chosen:
        return
    print("\n\nPaste into the cameras block:\n")
    for name, (exp, gain) in chosen.items():
        print(f"  {name}:")
        print(f"    exposure: {exp}")
        print(f"    gain: {gain}")


if __name__ == "__main__":
    main()
