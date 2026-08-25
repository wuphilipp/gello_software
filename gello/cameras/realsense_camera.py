import os
import time
from typing import List, Optional, Tuple

import numpy as np

from gello.cameras.camera import CameraDriver


def get_device_ids() -> List[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = ctx.query_devices()
    device_ids = []
    for dev in devices:
        dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    return device_ids


def _frame_meta(rs, frame, name):
    """Frame metadata value, or None when the device does not expose it."""
    field = getattr(rs.frame_metadata_value, name, None)
    if frame is None or field is None:
        return None
    try:
        return frame.get_frame_metadata(field) if frame.supports_frame_metadata(field) else None
    except Exception:
        return None


class RealSenseCamera(CameraDriver):
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"

    def __init__(
        self,
        device_id: Optional[str] = None,
        flip: bool = False,
        enable_depth: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        exposure: Optional[float] = None,
        gain: Optional[float] = None,
        white_balance: Optional[float] = None,
        lock_exposure_after: Optional[int] = None,
        constant_framerate: bool = True,
    ):
        """Open one RealSense camera.

        Args:
            device_id: serial number. If None, hardware-resets every attached
                device and opens the default one -- slow and disruptive, so
                prefer passing a serial.
            flip: rotate 180 degrees.
            enable_depth: stream depth as well as colour. Off by default:
                depth is not recorded (pi0.5 takes RGB + language +
                proprioception, never depth) and at 640x480 z16/30 it costs
                ~147 Mbit/s of USB bandwidth. On a USB 2.0 link that is the
                difference between working and starving whatever else shares
                the controller.
            width, height, fps: colour (and depth) stream geometry.
            exposure: fixed exposure in microseconds. Setting this (or gain)
                turns auto-exposure off. Auto-exposure re-hunts as a wrist
                camera sweeps over light and dark surfaces, so every frame gets
                a slightly different brightness -- nuisance variation a policy
                has to learn to ignore. Lower values also cut motion blur, at
                the cost of noise you have to buy back with gain.
            gain: fixed sensor gain. Also disables auto-exposure.
            white_balance: fixed white balance in Kelvin; disables auto white
                balance, which otherwise drifts the colour cast per frame.
            lock_exposure_after: run auto-exposure for this many frames, then
                freeze whatever it converged to. Gives a correct exposure for
                the current lighting with no manual tuning, and no drift
                afterwards. Ignored if exposure/gain are given explicitly.
            constant_framerate: clear auto_exposure_priority where supported
                (D435 sets it by default), which otherwise lets the camera drop
                below the requested fps to buy exposure time. A dataset
                recorded at a fixed rate wants a fixed rate.
        """
        import pyrealsense2 as rs

        self._rs = rs
        self._device_id = device_id
        self._enable_depth = enable_depth

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
            self._pipeline = rs.pipeline()
            config = rs.config()
        else:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)

        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self._pipeline.start(config)
        self._flip = flip

        self._configure_color(
            profile, exposure, gain, white_balance, lock_exposure_after,
            constant_framerate,
        )

    def _color_sensor(self, profile):
        """The sensor carrying the colour stream.

        Not always the one called "RGB Camera": a D405 has no separate colour
        sensor and serves colour off its Stereo Module.
        """
        for sensor in profile.get_device().query_sensors():
            for sp in sensor.get_stream_profiles():
                if sp.stream_type() == self._rs.stream.color:
                    return sensor
        return None

    def _configure_color(self, profile, exposure, gain, white_balance,
                         lock_exposure_after, constant_framerate) -> None:
        rs = self._rs
        sensor = self._color_sensor(profile)
        if sensor is None:
            return

        def set_opt(name, value):
            opt = getattr(rs.option, name, None)
            if opt is None or not sensor.supports(opt):
                return False
            try:
                sensor.set_option(opt, float(value))
                return True
            except Exception as exc:
                print(f"  {self._device_id}: could not set {name}={value}: {exc}")
                return False

        def get_opt(name):
            opt = getattr(rs.option, name, None)
            if opt is None or not sensor.supports(opt):
                return None
            try:
                return sensor.get_option(opt)
            except Exception:
                return None

        if constant_framerate:
            set_opt("auto_exposure_priority", 0)

        if exposure is None and gain is None and lock_exposure_after:
            # Let auto-exposure converge on the actual scene, then freeze it.
            set_opt("enable_auto_exposure", 1)
            frame = None
            for _ in range(int(lock_exposure_after)):
                try:
                    frame = self._pipeline.wait_for_frames().get_color_frame()
                except Exception:
                    break

            # Read what auto-exposure actually settled on from the FRAME, not
            # from the sensor options: while auto-exposure is running the option
            # is stale and can be wildly wrong -- observed reporting 2000 us
            # against a true 31979 us, a 16x error that silently locks the
            # camera to a badly exposed setting.
            exposure = _frame_meta(rs, frame, "actual_exposure")
            gain = _frame_meta(rs, frame, "gain_level")
            if exposure is None:
                print(f"  {self._device_id}: auto-exposure readback unavailable; "
                      "leaving auto-exposure on. Set exposure/gain explicitly to lock.")
            else:
                if white_balance is None:
                    white_balance = get_opt("white_balance")
                print(f"  {self._device_id}: locked exposure={exposure:g}us gain={gain:g}")

        if exposure is not None or gain is not None:
            set_opt("enable_auto_exposure", 0)
            if exposure is not None:
                set_opt("exposure", exposure)
            if gain is not None:
                set_opt("gain", gain)

        if white_balance is not None:
            set_opt("enable_auto_white_balance", 0)
            set_opt("white_balance", white_balance)

        self.color_settings = {
            "auto_exposure": get_opt("enable_auto_exposure"),
            "exposure": get_opt("exposure"),
            "gain": get_opt("gain"),
            "auto_white_balance": get_opt("enable_auto_white_balance"),
            "white_balance": get_opt("white_balance"),
        }

    @property
    def device_id(self) -> Optional[str]:
        return self._device_id

    def close(self) -> None:
        """Release the RealSense pipeline so another process can open the camera."""
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is None:
            return
        try:
            pipeline.stop()
        except Exception:
            pass  # already stopped, or never successfully started
        self._pipeline = None

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # farthest: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image, shape=(H, W, 3)
            np.ndarray: The depth image, shape=(H, W, 1)
        """
        import cv2

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # The stream is bgr8, so this view is RGB -- see image_io.rgb_to_bgr
        # for why that distinction matters downstream.
        image = color_image[:, :, ::-1]
        if img_size is not None:
            image = cv2.resize(image, img_size)

        depth = None
        if self._enable_depth:
            depth_image = np.asanyarray(frames.get_depth_frame().get_data())
            depth = (
                depth_image if img_size is None else cv2.resize(depth_image, img_size)
            )

        # rotate 180 degrees because everything is upside down in order to
        # center the camera
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            if depth is not None:
                depth = cv2.rotate(depth, cv2.ROTATE_180)
        if depth is not None:
            depth = depth[:, :, None]

        return image, depth


def _debug_read(camera, save_datastream=False):
    import cv2

    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    if not os.path.exists("images"):
        os.makedirs("images")
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    while True:
        time.sleep(0.1)
        image, depth = camera.read()
        key = cv2.waitKey(1)
        cv2.imshow("image", image[:, :, ::-1])
        if depth is not None:
            cv2.imshow("depth", np.concatenate([depth, depth, depth], axis=-1))
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
        if key == 27:
            break


if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0], enable_depth=True)
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)
