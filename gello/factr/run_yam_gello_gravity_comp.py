"""
Standalone FACTR Gravity Compensation Script (Non-ROS)

This script provides the similar gravity compensation functionality as the ROS-based
FACTR teleop system, but without ROS dependencies. 
Usage:
    python3 gello/factr/gravity_compensation.py --config configs/yam_gello_factr_hw.yaml
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import yaml

from gello.dynamixel.driver import DynamixelDriver

import threading
from importlib import import_module
from typing import Any, Dict, cast


def find_ttyusb(port_name: str) -> str:
    """Locate the underlying ttyUSB device."""
    base_path = "/dev/serial/by-id/"
    full_path = os.path.join(base_path, port_name)
    if not os.path.exists(full_path):
        raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        resolved_path = os.readlink(full_path)
        actual_device = os.path.basename(resolved_path)
        if actual_device.startswith("ttyUSB"):
            return actual_device
        else:
            raise Exception(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
            )
    except Exception as e:
        raise Exception(
            f"Unable to resolve the symbolic link for '{port_name}'. {e}"
        ) from e


def _instantiate_from_dict(cfg: Dict[str, Any]) -> Any:
    """Lightweight instantiation from a dict with a _target_ path.

    Keeps this script self-contained without importing broader launch utilities.
    """
    assert isinstance(cfg, dict) and "_target_" in cfg, "Invalid instantiation config"
    module_path, class_name = cfg["_target_"].rsplit(".", 1)
    cls = getattr(import_module(module_path), class_name)
    kwargs = {k: v for k, v in cfg.items() if k != "_target_"}

    # Recurse into nested dicts/lists
    def _recurse(v):
        if isinstance(v, dict) and "_target_" in v:
            return _instantiate_from_dict(v)
        if isinstance(v, dict):
            return {kk: _recurse(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_recurse(x) for x in v]
        return v

    return cls(**{k: _recurse(v) for k, v in kwargs.items()})


class FACTRGravityCompensation:
    """
    Standalone FACTR gravity compensation system without ROS dependencies.

    This class implements the core functionality of FACTR teleop gravity compensation,
    including:
    - Gravity compensation using inverse dynamics
    - Null-space regulation
    - Joint limit barriers
    - Static friction compensation
    """

    CALIBRATION_RANGE_MULTIPLIER = 20  # Range: -20π to 20π
    CALIBRATION_STEP_COUNT = 81  # 20 * 4 + 1 steps

    def __init__(self, config_path: str):
        self.running = False
        self.config_path = config_path
        self.driver: Optional[DynamixelDriver] = None  # Initialize early for cleanup

        # Teleop-related fields
        self.teleop_enabled: bool = False
        self.teleop_env = None
        self.teleop_client = None
        self.teleop_rate_hz: float = 30.0
        self.teleop_thread: Optional[threading.Thread] = None
        self.teleop_robot_server = None
        self.teleop_threads: list[threading.Thread] = []
        self.teleop_prepared: bool = False
        # Mapping from leader (arm joints) -> follower (first K joints)
        self.map_index: Optional[np.ndarray] = None
        self.map_signs: Optional[np.ndarray] = None
        self.map_offsets: Optional[np.ndarray] = None
        # Optional gripper teleop mapping using explicit open/close angles (degrees)
        self.gripper_open_rad: Optional[float] = None
        self.gripper_close_rad: Optional[float] = None
        # Last raw leader gripper reading in radians (before offsets/signs)
        self.leader_gripper_raw_rad: float = 0.0
        # Teleop smoothing (match baseline non-FACTR behavior)
        self.teleop_smoothing_alpha: float = 0.99
        self._teleop_last_action: Optional[np.ndarray] = None

        try:
            self._load_config()
            self._setup_parameters()
            self._prepare_dynamixel()
            self._prepare_inverse_dynamics()
            self._calibrate_system()
            # Optional teleop setup
            self._maybe_setup_teleop()
        except Exception as e:
            # Cleanup on initialization failure
            self.shutdown()
            raise RuntimeError(f"Failed to initialize FACTR system: {e}") from e

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        print(f"Loaded config: {self.config['name']}")

    def _setup_parameters(self) -> None:
        """Initialize parameters from config."""
        self.name = self.config["name"]
        self.dt = 1 / self.config["controller"]["frequency"]

        # Leader arm parameters
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.safety_margin = self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        self.arm_joint_limits_max = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_max"])
            - self.safety_margin
        )
        self.arm_joint_limits_min = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_min"])
            + self.safety_margin
        )
        self.calibration_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["calibration_joint_pos"]
        )
        self.initial_match_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["initial_match_joint_pos"]
        )

        # Gripper parameters
        self.gripper_limit_min = 0.0
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0

        # Control parameters
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.gravity_comp_modifier = self.config["controller"]["gravity_comp"]["gain"]
        self.tau_g = np.zeros(self.num_arm_joints)

        # Friction compensation
        self.stiction_comp_enable_speed = self.config["controller"][
            "static_friction_comp"
        ]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"][
            "gain"
        ]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)

        # Joint limit barrier
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]

        # Null space regulation
        self.null_space_joint_target = np.array(
            self.config["controller"]["null_space_regulation"][
                "null_space_joint_target"
            ]
        )
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]

        print(f"Control frequency: {1 / self.dt:.1f} Hz")
        print(
            f"Gravity compensation: {'enabled' if self.enable_gravity_comp else 'disabled'}"
        )

    def _prepare_dynamixel(self) -> None:
        """Initialize Dynamixel servo driver."""
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(
            self.config["dynamixel"]["joint_signs"], dtype=float
        )
        self.dynamixel_port = (
            "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]
        )

        # Check latency timer
        try:
            port_name = os.path.basename(self.dynamixel_port)
            ttyUSBx = find_ttyusb(port_name)
            latency_path = f"/sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            result = subprocess.run(
                ["cat", latency_path], capture_output=True, text=True, check=True
            )
            ttyUSB_latency_timer = int(result.stdout)
            if ttyUSB_latency_timer != 1:
                print(
                    f"Warning: Latency timer of {ttyUSBx} is {ttyUSB_latency_timer}, should be 1 for optimal performance."
                )
                print(
                    f"Run: echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
                )
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
            print(f"Could not check latency timer (file access issue): {e}")
        except (ValueError, IndexError) as e:
            print(f"Could not parse latency timer value: {e}")
        except Exception as e:
            print(f"Unexpected error checking latency timer: {e}")

        # Initialize driver
        joint_ids = (np.arange(self.num_motors) + 1).tolist()
        try:
            self.driver = DynamixelDriver(
                joint_ids, self.servo_types, self.dynamixel_port
            )
            print(f"Connected to Dynamixel servos on {self.dynamixel_port}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Dynamixel servos: {e}") from e

        # Configure servos
        self.driver.set_torque_mode(False)
        if self.enable_gravity_comp:
            # Use current control with torque enabled when GC is active
            self.driver.set_operating_mode(0)  # Current control mode
            self.driver.set_torque_mode(True)
        else:
            # When GC is disabled, keep motors free/backdrivable similar to baseline
            # Try to switch to position mode (not strictly necessary) and keep torque disabled
            try:
                self.driver.set_operating_mode(3)  # Position control mode
            except Exception:
                pass
            self.driver.set_torque_mode(False)

    def _prepare_inverse_dynamics(self) -> None:
        """Initialize Pinocchio model for inverse dynamics."""
        # Construct URDF path - make GELLO completely self-contained
        urdf_filename = self.config["arm_teleop"]["leader_urdf"]

        # Try relative to config file first
        config_dir = Path(self.config_path).parent
        urdf_path = config_dir / urdf_filename

        # If not found, try relative to GELLO factr directory (self-contained)
        if not urdf_path.exists():
            gello_factr_urdf_path = (
                Path(__file__).parent / "urdf" / Path(urdf_filename).name
            )
            if gello_factr_urdf_path.exists():
                urdf_path = gello_factr_urdf_path
            else:
                # Try the full GELLO path
                gello_full_urdf_path = (
                    Path(__file__).parent.parent.parent / urdf_filename
                )
                if gello_full_urdf_path.exists():
                    urdf_path = gello_full_urdf_path
                else:
                    raise FileNotFoundError(
                        f"URDF file {urdf_filename} not found in GELLO paths"
                    )

        print(f"Loading URDF: {urdf_path}")
        urdf_model_dir = str(urdf_path.parent)
        self.pin_model, _, _ = pin.buildModelsFromUrdf(
            filename=str(urdf_path), package_dirs=urdf_model_dir
        )
        self.pin_data = self.pin_model.createData()

    def _calibrate_system(self) -> None:
        """Calibrate Dynamixel offsets and match initial position."""
        print("Calibrating Dynamixel offsets...")
        self._get_dynamixel_offsets()
        print("Skipping initial position match...")
        print("System calibrated and ready!")

    def _maybe_setup_teleop(self) -> None:
        """Optionally set up a follower robot and teleop loop if enabled by config."""
        teleop_cfg = self.config.get("teleop", {})
        enabled = bool(teleop_cfg.get("enable", False))
        if not enabled:
            return

        # Lazily import here to avoid adding dependencies when teleop is disabled
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot
        from gello.env import RobotEnv

        self.teleop_enabled = True
        self.teleop_rate_hz = float(teleop_cfg.get("hz", 30))
        server_host = teleop_cfg.get("robot", {}).get("host", "127.0.0.1")
        base_port = int(teleop_cfg.get("robot", {}).get("port", 6001))
        server_timeout_s = float(teleop_cfg.get("wait_for_server_timeout_s", 5.0))

        robot_cfg = teleop_cfg.get("robot")
        if not isinstance(robot_cfg, dict) or "_target_" not in robot_cfg:
            raise ValueError("teleop.robot must be a dict containing a _target_ field")

        # Optional gripper config: [id, open_deg, close_deg]
        gc = teleop_cfg.get("gripper_config")
        if isinstance(gc, list) and len(gc) == 3:
            try:
                _, open_deg, close_deg = gc
                self.gripper_open_rad = float(open_deg) * np.pi / 180.0
                self.gripper_close_rad = float(close_deg) * np.pi / 180.0
                print(
                    f"Teleop gripper_config set (rad): open={self.gripper_open_rad:.3f}, close={self.gripper_close_rad:.3f}"
                )
            except Exception as e:
                print(f"Warning: invalid teleop.gripper_config, ignoring: {e}")

        # Instantiate follower robot from config
        follower_robot = _instantiate_from_dict(robot_cfg)
        self.teleop_robot_server = follower_robot

        # Determine server port to use
        server_port = base_port
        if hasattr(follower_robot, "serve"):
            # Sim server exposes port in its own config; prefer that
            try:
                server_port = int(robot_cfg.get("port", base_port))
            except Exception:
                server_port = base_port

        # Start server in background if needed
        if hasattr(follower_robot, "serve"):
            server_thread = threading.Thread(target=follower_robot.serve, daemon=True)
            server_thread.start()
            self.teleop_threads.append(server_thread)
        else:
            # Hardware robot; wrap with ZMQServerRobot and auto-select port if needed
            from zmq.error import ZMQError  # type: ignore

            selected_port = None
            last_error: Optional[Exception] = None
            for port_delta in range(0, 16):
                try:
                    candidate_port = base_port + port_delta
                    server = ZMQServerRobot(
                        follower_robot, port=candidate_port, host=server_host
                    )
                    server_thread = threading.Thread(target=server.serve, daemon=True)
                    server_thread.start()
                    self.teleop_robot_server = server
                    self.teleop_threads.append(server_thread)
                    selected_port = candidate_port
                    break
                except (ZMQError, Exception) as e:  # bind may fail if address in use
                    last_error = e
                    msg = str(e)
                    if "Address already in use" in msg or "address in use" in msg:
                        continue
                    raise
            if selected_port is None:
                raise RuntimeError(
                    f"Failed to create ZMQ server for hardware follower: {last_error}"
                )
            server_port = selected_port

        # Wait for server to become ready
        start = time.time()
        while True:
            try:
                client = ZMQClientRobot(port=server_port, host=server_host)
                # Probe an RPC to ensure server responsiveness
                _ = client.num_dofs()
                self.teleop_client = client
                break
            except Exception:
                if time.time() - start > server_timeout_s:
                    raise RuntimeError(
                        f"Follower server failed to start on {server_host}:{server_port} within {server_timeout_s} seconds"
                    )
                time.sleep(0.1)

        # Create env for follower
        self.teleop_env = RobotEnv(
            self.teleop_client, control_rate_hz=self.teleop_rate_hz
        )

        # Determine follower DOFs and build mapping defaults
        try:
            follower_dofs = int(self.teleop_client.num_dofs())
        except Exception:
            follower_dofs = self.num_arm_joints
        # If follower appears to include a gripper as last joint
        follower_has_gripper = follower_dofs == (self.num_arm_joints + 1)
        map_dims = min(
            self.num_arm_joints, follower_dofs - (1 if follower_has_gripper else 0)
        )
        default_index = np.arange(map_dims, dtype=int)
        default_signs = np.ones(map_dims, dtype=float)
        default_offsets = np.zeros(map_dims, dtype=float)

        # Load mapping config
        mapping_cfg = (
            teleop_cfg.get("mapping", {})
            if isinstance(teleop_cfg.get("mapping", {}), dict)
            else {}
        )
        index_map = mapping_cfg.get("index_map")
        signs = mapping_cfg.get("signs")
        offsets = mapping_cfg.get("offsets")
        auto_align = bool(mapping_cfg.get("auto_align", True))

        if index_map is None:
            self.map_index = default_index
        else:
            self.map_index = np.array(index_map, dtype=int)
        if signs is None:
            self.map_signs = default_signs
        else:
            self.map_signs = np.array(signs, dtype=float)
        if offsets is None:
            self.map_offsets = default_offsets
        else:
            self.map_offsets = np.array(offsets, dtype=float)

        # Use local, non-optional views for validation
        map_index_local = cast(np.ndarray, self.map_index)
        map_signs_local = cast(np.ndarray, self.map_signs)
        map_offsets_local = cast(np.ndarray, self.map_offsets)

        # Validate lengths
        if not (
            len(map_index_local)
            == len(map_signs_local)
            == len(map_offsets_local)
            == map_dims
        ):
            print(
                "Warning: teleop.mapping lengths mismatch or not equal to map_dims; using defaults."
            )
            self.map_index = default_index
            self.map_signs = default_signs
            self.map_offsets = default_offsets
            map_index_local = default_index
            map_signs_local = default_signs
            map_offsets_local = default_offsets

        # Optionally move follower to a start position
        start_joints = teleop_cfg.get("start_joints")
        if start_joints is not None:
            try:
                self._move_follower_to_start(np.array(start_joints, dtype=float))
            except Exception as e:
                print(f"Warning: failed to move follower to start position: {e}")

        # Optional auto-alignment: compute offsets so current follower == mapped leader
        if auto_align:
            try:
                obs = self.teleop_env.get_obs()
                follower_curr = obs["joint_positions"]
                leader_arm_pos, _, _, _ = self.get_leader_joint_states()
                leader_mapped = map_signs_local * leader_arm_pos[map_index_local]
                follower_slice = follower_curr[: int(len(map_index_local))]
                self.map_offsets = follower_slice - leader_mapped
                map_offsets_local = cast(np.ndarray, self.map_offsets)
                print(
                    f"Teleop auto-aligned offsets set to: {[float(x) for x in map_offsets_local]}"
                )
            except Exception as e:
                print(f"Warning: auto_align failed: {e}")

        # Mark prepared; DO NOT start thread yet (start in run() after running=True)
        self.teleop_prepared = True
        print(
            f"Teleop prepared: follower ready at {server_host}:{server_port}, will mirror at {self.teleop_rate_hz:.1f} Hz"
        )
        # Print mapping and initial states for quick tuning
        try:
            map_index_local = cast(np.ndarray, self.map_index)
            map_signs_local = cast(np.ndarray, self.map_signs)
            map_offsets_local = cast(np.ndarray, self.map_offsets)
            leader_arm_pos, _, _, _ = self.get_leader_joint_states()
            leader_mapped_dbg = (
                map_signs_local * leader_arm_pos[map_index_local] + map_offsets_local
            )
            follower_dbg = self.teleop_env.get_obs()["joint_positions"][
                : int(len(map_index_local))
            ]

            def _fmt(arr):
                return [float(f"{x:.3f}") for x in np.array(arr).tolist()]

            print("Teleop mapping:")
            print(f"  index_map: {map_index_local.tolist()}")
            print(f"  signs: {_fmt(map_signs_local)}")
            print(f"  offsets: {_fmt(map_offsets_local)}")
            print(f"  leader(mapped): {_fmt(leader_mapped_dbg)}")
            print(f"  follower(curr): {_fmt(follower_dbg)}")
        except Exception as e:
            print(f"Warning: teleop debug print failed: {e}")

    def _move_follower_to_start(self, target_joints: np.ndarray) -> None:
        assert self.teleop_env is not None
        obs = self.teleop_env.get_obs()
        curr = obs["joint_positions"]
        if curr.shape != target_joints.shape:
            print("Warning: follower start joints shape mismatch; skipping")
            return
        steps = int(min(max(np.abs(curr - target_joints).max() / 0.01, 1), 100))
        for jnt in np.linspace(curr, target_joints, steps):
            self.teleop_env.step(jnt)
            time.sleep(0.001)

    def _build_follower_action(
        self,
        self_arm_pos: npt.NDArray[np.float64],
        self_gripper_pos: float,
    ) -> np.ndarray:
        """Build follower joint command from leader arm and gripper positions.

        Applies configured mapping: index, signs, offsets. Handles extra gripper DOF.
        """
        assert self.teleop_client is not None
        try:
            follower_dofs = self.teleop_client.num_dofs()
        except Exception:
            follower_dofs = len(self_arm_pos)

        # Prepare mapped arm command
        if self.map_index is None or self.map_signs is None or self.map_offsets is None:
            map_index = np.arange(min(len(self_arm_pos), follower_dofs), dtype=int)
            map_signs = np.ones_like(map_index, dtype=float)
            map_offsets = np.zeros_like(map_index, dtype=float)
        else:
            map_index = self.map_index
            map_signs = self.map_signs
            map_offsets = self.map_offsets

        arm_cmd = map_signs * self_arm_pos[map_index] + map_offsets

        # Compose final command with optional gripper channel
        if follower_dofs == len(arm_cmd) + 1:
            # Determine normalized gripper using explicit config if provided
            if self.gripper_open_rad is not None and self.gripper_close_rad is not None:
                denom = self.gripper_close_rad - self.gripper_open_rad
                if abs(denom) < 1e-6:
                    gripper_norm = 0.0
                else:
                    gripper_norm = (
                        self.leader_gripper_raw_rad - self.gripper_open_rad
                    ) / denom
            else:
                # Fallback to actuation_range
                gripper_norm = self_gripper_pos / max(self.gripper_limit_max, 1e-6)
            gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
            return np.concatenate([arm_cmd, np.array([gripper_norm], dtype=float)])

        if follower_dofs == len(arm_cmd):
            return arm_cmd
        if follower_dofs < len(arm_cmd):
            return arm_cmd[:follower_dofs]
        # Pad with zeros for any extra joints (unlikely)
        padded = np.zeros((follower_dofs,), dtype=float)
        padded[: len(arm_cmd)] = arm_cmd
        return padded

    def _teleop_loop(self) -> None:
        assert self.teleop_env is not None
        rate_dt = 1.0 / max(self.teleop_rate_hz, 1e-3)
        print("Starting teleop loop (follower control)")
        while self.running:
            t0 = time.time()
            try:
                # Use the same leader state access used by GC, which already applies offsets/signs
                (
                    leader_arm_pos,
                    leader_arm_vel,
                    leader_gripper_pos,
                    leader_gripper_vel,
                ) = self.get_leader_joint_states()
                action = self._build_follower_action(leader_arm_pos, leader_gripper_pos)
                # Apply exponential smoothing to follower command to mirror baseline
                if self._teleop_last_action is None or len(
                    self._teleop_last_action
                ) != len(action):
                    self._teleop_last_action = action
                else:
                    action = (
                        self._teleop_last_action * (1.0 - self.teleop_smoothing_alpha)
                        + action * self.teleop_smoothing_alpha
                    )
                    self._teleop_last_action = action
                self.teleop_env.step(action)
            except Exception as e:
                print(f"Teleop loop warning: {e}")
            # Timing
            elapsed = time.time() - t0
            sleep_t = rate_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _get_dynamixel_offsets(self, verbose: bool = True) -> None:
        """Calibrate Dynamixel servos to match expected joint positions."""
        # Warm up
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        for _ in range(10):
            self.driver.get_positions_and_velocities()

        def get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        # Get arm offsets
        self.joint_offsets = []
        curr_joints, _ = self.driver.get_positions_and_velocities()
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            # Search over intervals of pi/2
            for offset in np.linspace(
                -self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
                self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
                self.CALIBRATION_STEP_COUNT,
            ):
                error = get_error(self.calibration_joint_pos, offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            self.joint_offsets.append(best_offset)

        # Get gripper offset
        curr_gripper_joint = curr_joints[-1]
        self.joint_offsets.append(curr_gripper_joint)
        self.joint_offsets = np.asarray(self.joint_offsets)

        if verbose:
            print(f"Joint offsets: {[f'{x:.3f}' for x in self.joint_offsets]}")

    def _match_start_pos(self) -> None:
        """Wait for leader arm to be moved to initial position."""
        while True:
            curr_pos, _, _, _ = self.get_leader_joint_states()
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0 : self.num_arm_joints]
            )
            if current_joint_error <= 0.6:
                break
            print(
                f"Please match starting joint position. Current error: {current_joint_error:.3f}"
            )
            time.sleep(0.5)

    def get_leader_joint_states(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Get current joint positions and velocities."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()

        # Apply offsets and signs for arm joints
        joint_pos_arm = (
            joint_pos[0 : self.num_arm_joints]
            - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]
        joint_vel_arm = (
            joint_vel[0 : self.num_arm_joints]
            * self.joint_signs[0 : self.num_arm_joints]
        )

        # Process gripper
        self.leader_gripper_raw_rad = float(joint_pos[-1])
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[
            -1
        ]
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt

        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def set_leader_joint_torque(
        self, arm_torque: npt.NDArray[np.float64], gripper_torque: float
    ) -> None:
        """Apply torque to leader arm and gripper."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque((arm_gripper_torque * self.joint_signs).tolist())

    def joint_limit_barrier(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
        gripper_joint_pos: float,
        gripper_joint_vel: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Compute joint limit repulsive torques."""
        # Arm joint limits
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_max_mask

        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_min_mask

        # Gripper limits
        if gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max)
                - self.joint_limit_kd * gripper_joint_vel
            )
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min)
                - self.joint_limit_kd * gripper_joint_vel
            )
        else:
            tau_l_gripper = 0.0

        return tau_l, tau_l_gripper

    def gravity_compensation(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute gravity compensation torques using inverse dynamics."""
        self.tau_g = pin.rnea(  # type: ignore[attr-defined]
            self.pin_model,
            self.pin_data,
            arm_joint_pos,
            arm_joint_vel,
            np.zeros_like(arm_joint_vel),
        )
        self.tau_g *= self.gravity_comp_modifier
        return self.tau_g

    def friction_compensation(
        self, arm_joint_vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute static friction compensation torques."""
        tau_ss = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def null_space_regulation(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute null-space regulation torques."""
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints)  # type: ignore[attr-defined]
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0 : self.num_arm_joints]
        tau_n = null_space_projector @ (
            -self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel
        )
        return tau_n

    def control_loop_step(self) -> None:
        """Execute one step of the control loop."""
        # Get current joint states
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = (
            self.get_leader_joint_states()
        )

        # Initialize torque commands
        torque_arm = np.zeros(self.num_arm_joints)

        # Joint limit barriers
        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l

        # Null space regulation
        torque_arm += self.null_space_regulation(leader_arm_pos, leader_arm_vel)

        # Gravity compensation and friction compensation
        if self.enable_gravity_comp:
            torque_arm += self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += self.friction_compensation(leader_arm_vel)

        # Apply torques only if GC is enabled (torque mode is off otherwise)
        if self.enable_gravity_comp:
            self.set_leader_joint_torque(torque_arm, torque_gripper)

    def run(self) -> None:
        """Run the main control loop."""
        print(f"Starting gravity compensation control loop at {1 / self.dt:.1f} Hz")
        print("Press Ctrl+C to stop")

        self.running = True
        # Start teleop thread now that running is True
        if self.teleop_enabled and self.teleop_prepared and self.teleop_thread is None:
            self.teleop_thread = threading.Thread(target=self._teleop_loop, daemon=True)
            self.teleop_thread.start()
            print("Teleop started.")
        try:
            while self.running:
                start_time = time.time()

                self.control_loop_step()

                # Maintain loop timing
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"Warning: Control loop overrun by {elapsed - self.dt:.4f}s")

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Safely shutdown the system."""
        self.running = False
        # Stop teleop thread and close ZMQ resources first
        try:
            if self.teleop_thread is not None and self.teleop_thread.is_alive():
                # Give teleop loop a cycle to exit since self.running is False
                time.sleep(0.05)
                # Join briefly
                self.teleop_thread.join(timeout=1.0)
            if self.teleop_client is not None and hasattr(self.teleop_client, "close"):
                self.teleop_client.close()
            # Attempt to stop underlying server if it has a stop
            if self.teleop_robot_server is not None and hasattr(
                self.teleop_robot_server, "stop"
            ):
                try:
                    self.teleop_robot_server.stop()
                except Exception:
                    pass
        except Exception as e:
            print(f"Teleop shutdown warning: {e}")

        if hasattr(self, "driver") and self.driver is not None:
            print("Disabling motor torques...")
            try:
                self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
            except Exception:
                pass
            self.driver.set_torque_mode(False)
            self.driver.close()
        print("Shutdown complete")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone FACTR Gravity Compensation"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="xdof/sandboxs/jliu/factr_grav_comp_demo.yaml",
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        # Create and run gravity compensation system
        system = FACTRGravityCompensation(args.config)

        # Set up signal handler for clean shutdown
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal")
            system.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the system
        system.run()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
