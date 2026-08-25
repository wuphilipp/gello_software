"""Image encoding helpers shared by the episode writer and the camera preview.

Both paths must agree on channel order, otherwise the preview would validate a
different pixel layout than the one that lands in the dataset.
"""

from typing import Tuple

import cv2
import numpy as np

# What pi0.5's PaliGemma backbone consumes. Recorded frames stay at native
# resolution; this is only for previewing what the policy will actually see.
POLICY_VIEW_SIZE = 224


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB frame to the BGR layout OpenCV's encoders expect.

    `RealSenseCamera.read` configures `rs.format.bgr8` and then returns
    `color_image[:, :, ::-1]`, so the array reaching us is RGB. Handing that
    straight to `cv2.imencode` would write red/blue-swapped files.

    This also returns a fresh contiguous array, which matters for the writer:
    `read()` wraps the librealsense frame buffer via `np.asanyarray` rather than
    copying it, and that buffer gets recycled once the frame is released. The
    copy is what makes it safe to hand a frame to a background encoder thread.
    """
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def encode_jpeg(bgr: np.ndarray, quality: int = 95) -> bytes:
    """JPEG-encode a BGR frame. Releases the GIL, so worker threads parallelise."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for frame shape {bgr.shape}")
    return buf.tobytes()


def encode_rgb_to_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    """Convenience: RGB in, JPEG bytes out."""
    return encode_jpeg(rgb_to_bgr(rgb), quality)


def center_crop_square(img: np.ndarray) -> np.ndarray:
    """Crop to the largest centered square, preserving aspect ratio."""
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return img[top : top + side, left : left + side]


def to_policy_view(
    img: np.ndarray, size: int = POLICY_VIEW_SIZE, mode: str = "squash"
) -> np.ndarray:
    """Resize to the square input a pi0.5-style policy receives.

    mode="squash" resizes the full frame, distorting 4:3 into 1:1 but keeping
    the whole field of view. mode="crop" center-crops to a square first, which
    preserves geometry but throws away the left and right edges. Which one is
    right depends on where the manipulated objects sit, which is exactly what
    the preview is for.
    """
    if mode == "crop":
        img = center_crop_square(img)
    elif mode != "squash":
        raise ValueError(f"mode must be 'squash' or 'crop', got {mode!r}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
