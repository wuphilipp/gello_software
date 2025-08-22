"""
Standalone FACTR Gravity Compensation Script (Non-ROS)

This script provides the same gravity compensation functionality as the ROS-based
FACTR teleop system, but without ROS dependencies. It can be used within the lab42
codebase where FACTR_Teleop is a submodule.

Usage:
    python xdof/factr/factr_grav_comp.py --config xdof/sandboxs/jliu/factr_grav_comp_demo.yaml
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

# Import FACTR's dynamixel driver directly (no ROS dependencies)
from gello.factr.lab42_driver import DynamixelDriver


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
        raise Exception(f"Unable to resolve the symbolic link for '{port_name}'. {e}") from e


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

        try:
            self._load_config()
            self._setup_parameters()
            self._prepare_dynamixel()
            self._prepare_inverse_dynamics()
            self._calibrate_system()
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
        self.arm_joint_limits_max = np.array(self.config["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin
        self.arm_joint_limits_min = np.array(self.config["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin
        self.calibration_joint_pos = np.array(self.config["arm_teleop"]["initialization"]["calibration_joint_pos"])
        self.initial_match_joint_pos = np.array(self.config["arm_teleop"]["initialization"]["initial_match_joint_pos"])

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
        self.stiction_comp_enable_speed = self.config["controller"]["static_friction_comp"]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"]["gain"]
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)

        # Joint limit barrier
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]

        # Null space regulation
        self.null_space_joint_target = np.array(
            self.config["controller"]["null_space_regulation"]["null_space_joint_target"]
        )
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]

        print(f"Control frequency: {1 / self.dt:.1f} Hz")
        print(f"Gravity compensation: {'enabled' if self.enable_gravity_comp else 'disabled'}")

    def _prepare_dynamixel(self) -> None:
        """Initialize Dynamixel servo driver."""
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config["dynamixel"]["joint_signs"], dtype=float)
        self.dynamixel_port = "/dev/serial/by-id/" + self.config["dynamixel"]["dynamixel_port"]

        # Check latency timer
        try:
            port_name = os.path.basename(self.dynamixel_port)
            ttyUSBx = find_ttyusb(port_name)
            latency_path = f"/sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            result = subprocess.run(["cat", latency_path], capture_output=True, text=True, check=True)
            ttyUSB_latency_timer = int(result.stdout)
            if ttyUSB_latency_timer != 1:
                print(
                    f"Warning: Latency timer of {ttyUSBx} is {ttyUSB_latency_timer}, should be 1 for optimal performance."
                )
                print(f"Run: echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer")
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
            print(f"Could not check latency timer (file access issue): {e}")
        except (ValueError, IndexError) as e:
            print(f"Could not parse latency timer value: {e}")
        except Exception as e:
            print(f"Unexpected error checking latency timer: {e}")

        # Initialize driver
        joint_ids = (np.arange(self.num_motors) + 1).tolist()
        try:
            self.driver = DynamixelDriver(joint_ids, self.servo_types, self.dynamixel_port)
            print(f"Connected to Dynamixel servos on {self.dynamixel_port}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Dynamixel servos: {e}") from e

        # Configure servos
        self.driver.set_torque_mode(False)
        self.driver.set_operating_mode(0)  # Current control mode
        self.driver.set_torque_mode(True)

    def _prepare_inverse_dynamics(self) -> None:
        """Initialize Pinocchio model for inverse dynamics."""
        # Construct URDF path - make GELLO completely self-contained
        urdf_filename = self.config["arm_teleop"]["leader_urdf"]

        # Try relative to config file first
        config_dir = Path(self.config_path).parent
        urdf_path = config_dir / urdf_filename

        # If not found, try relative to GELLO factr directory (self-contained)
        if not urdf_path.exists():
            gello_factr_urdf_path = Path(__file__).parent / "urdf" / Path(urdf_filename).name
            if gello_factr_urdf_path.exists():
                urdf_path = gello_factr_urdf_path
            else:
                # Try the full GELLO path
                gello_full_urdf_path = Path(__file__).parent.parent.parent / urdf_filename
                if gello_full_urdf_path.exists():
                    urdf_path = gello_full_urdf_path
                else:
                    raise FileNotFoundError(f"URDF file {urdf_filename} not found in GELLO paths")

        print(f"Loading URDF: {urdf_path}")
        urdf_model_dir = str(urdf_path.parent)
        self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=str(urdf_path), package_dirs=urdf_model_dir)
        self.pin_data = self.pin_model.createData()

    def _calibrate_system(self) -> None:
        """Calibrate Dynamixel offsets and match initial position."""
        print("Calibrating Dynamixel offsets...")
        self._get_dynamixel_offsets()
        print("Skipping initial position match...")
        print("System calibrated and ready!")

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
            current_joint_error = np.linalg.norm(curr_pos - self.initial_match_joint_pos[0 : self.num_arm_joints])
            if current_joint_error <= 0.6:
                break
            print(f"Please match starting joint position. Current error: {current_joint_error:.3f}")
            time.sleep(0.5)

    def get_leader_joint_states(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Get current joint positions and velocities."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        self.gripper_pos_prev = self.gripper_pos
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()

        # Apply offsets and signs for arm joints
        joint_pos_arm = (
            joint_pos[0 : self.num_arm_joints] - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]
        joint_vel_arm = joint_vel[0 : self.num_arm_joints] * self.joint_signs[0 : self.num_arm_joints]

        # Process gripper
        self.gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[-1]
        gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt

        return joint_pos_arm, joint_vel_arm, self.gripper_pos, gripper_vel

    def set_leader_joint_torque(self, arm_torque: npt.NDArray[np.float64], gripper_torque: float) -> None:
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
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max) - self.joint_limit_kd * arm_joint_vel
        ) * exceed_max_mask

        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min) - self.joint_limit_kd * arm_joint_vel
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
        self, arm_joint_pos: npt.NDArray[np.float64], arm_joint_vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute gravity compensation torques using inverse dynamics."""
        self.tau_g = pin.rnea(
            self.pin_model, self.pin_data, arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel)
        )
        self.tau_g *= self.gravity_comp_modifier
        return self.tau_g

    def friction_compensation(self, arm_joint_vel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
        self, arm_joint_pos: npt.NDArray[np.float64], arm_joint_vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute null-space regulation torques."""
        J = pin.computeJointJacobian(self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints)
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0 : self.num_arm_joints]
        tau_n = null_space_projector @ (-self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel)
        return tau_n

    def control_loop_step(self) -> None:
        """Execute one step of the control loop."""
        # Get current joint states
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()

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

        # Apply torques
        self.set_leader_joint_torque(torque_arm, torque_gripper)

    def run(self) -> None:
        """Run the main control loop."""
        print(f"Starting gravity compensation control loop at {1 / self.dt:.1f} Hz")
        print("Press Ctrl+C to stop")
        
        self.running = True
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
        if hasattr(self, "driver") and self.driver is not None:
            print("Disabling motor torques...")
            self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
            self.driver.set_torque_mode(False)
            self.driver.close()
        print("Shutdown complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone FACTR Gravity Compensation")
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


 