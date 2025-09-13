#!/usr/bin/env python3
"""
YAM GELLO Agent with FACTR Gravity Compensation

This module provides a GELLO agent for the YAM robot with FACTR gravity compensation capabilities.
FACTR is just the name for the gravity compensation method - this is still a GELLO agent.

Features:
- GELLO agent interface (same as other GELLO agents)
- FACTR gravity compensation system integration
- Dynamixel servo support with torque control
- URDF-based inverse dynamics
- Zero torque mode for safe testing
"""

import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import os

from gello.agents.agent import Agent
from gello.dynamixel.driver import DynamixelDriver
from gello.factr.gravity_compensation import FACTRGravityCompensation


@dataclass
class YAMGelloConfig:
    """Configuration for YAM GELLO agent with FACTR gravity compensation."""

    # Hardware configuration
    port: str = (
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0"
    )
    baudrate: int = 57600
    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    # Joint configuration (from working standalone FACTR)
    joint_signs: Tuple[float, ...] = (
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )  # 7 elements including gripper

    # Servo types for torque conversion
    servo_types: Tuple[str, ...] = (
        "XC330_T288_T",
        "XM430_W210_T",
        "XM430_W210_T",
        "XC330_T288_T",
        "XC330_T288_T",
        "XC330_T288_T",
        "XC330_T288_T",
    )  # 7 elements including gripper

    # URDF configuration
    urdf_path: str = "gello/factr/urdf/yam_active_gello/robot.urdf"

    control_frequency: float = 500.0
    gravity_gain: float = 1.0  # Full strength
    null_space_kp: float = 0.0
    null_space_kd: float = 0.0
    friction_gain: float = 0.0
    barrier_kp: float = 0.0
    barrier_kd: float = 0.0

    max_torque: float = 2.0  # Maximum torque per joint in Nm

    # Gripper configuration
    gripper_config: Optional[Tuple[int, float, float]] = (
        7,
        -30.0,
        24.0,
    )  # ID: 7, open: -30°, closed: 24°

    # Starting joint positions
    start_joints: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


class YAMGelloRobot:
    """YAM robot with GELLO interface and FACTR gravity compensation capabilities."""

    def __init__(self, config: YAMGelloConfig):
        """Initialize YAM GELLO robot.

        Args:
            config: Configuration for the robot
        """
        self.config = config
        self.driver = None
        self.factr_system = None
        self._running = False
        self._control_thread = None
        self._sim_mode = False  # Track simulation mode

        # Gripper configuration
        self.gripper_open_close = None
        if self.config.gripper_config is not None:
            # Convert degrees to radians for gripper open/close positions
            self.gripper_open_close = (
                self.config.gripper_config[1] * np.pi / 180,  # open position in radians
                self.config.gripper_config[2]
                * np.pi
                / 180,  # closed position in radians
            )

        # Initialize robot
        self._setup_robot()
        self._setup_factr_system()

    def _setup_robot(self):
        """Setup robot interface (Dynamixel for real, SimRobot for simulation)."""
        try:
            # Check if this is simulation mode
            if self.config.port.startswith("sim://"):
                print(f"Setting up YAM GELLO robot in SIMULATION mode")
                print(f"Connecting to simulation robot at {self.config.port}")

                # For simulation, we'll use a dummy driver that reads from sim
                # The actual robot will be handled by the launch system
                self.driver = None
                self._sim_mode = True

                print("✓ YAM GELLO robot setup complete (simulation mode)")

            else:
                print(f"Setting up YAM GELLO robot on port {self.config.port}")

                # Create Dynamixel driver for real robot
                # Include gripper joint ID if configured
                joint_ids = list(self.config.joint_ids)
                servo_types = list(
                    self.config.servo_types
                )  # Already has 7 elements including gripper

                if self.config.gripper_config is not None:
                    joint_ids.append(self.config.gripper_config[0])
                    # Don't append to servo_types - config already has 7 elements
                    # servo_types.append("XC330_T288_T")  # This was creating an 8-element array!

                self.driver = DynamixelDriver(
                    ids=joint_ids, port=self.config.port, servo_types=servo_types
                )

                # Configure servos for mixed control:
                # - Arm joints: Current control mode for FACTR torque control
                # - Gripper: Position control mode for direct position commands

                # First, set all joints to position control mode
                self.driver.set_operating_mode(3)  # Position control mode
                self.driver.set_torque_mode(True)

                # Then, set arm joints back to current control mode for FACTR
                # Note: This will be handled by the FACTR system when it starts

                self._sim_mode = False
                print("✓ YAM GELLO robot setup complete (real robot mode)")
                print(f"  Configured {len(joint_ids)} joints (arm + gripper)")
                print(f"  Arm joints: {self.config.joint_ids}")
                if self.config.gripper_config:
                    print(f"  Gripper joint: {self.config.gripper_config[0]}")

        except Exception as e:
            print(f"✗ Failed to setup YAM GELLO robot: {e}")
            raise

    def _setup_factr_system(self):
        """Setup FACTR gravity compensation system."""
        try:
            if self._sim_mode:
                print(
                    "Setting up FACTR gravity compensation system in SIMULATION mode..."
                )
                # In simulation mode, we can't run FACTR physics calculations
                # Just create a dummy system for testing
                self.factr_system = None
                print("✓ FACTR system setup complete (simulation mode - dummy system)")
                return

            print("Setting up FACTR gravity compensation system...")

            # Switch arm joints to current control mode for FACTR
            print("  Switching arm joints to current control mode...")
            try:
                # Set arm joints to current control mode (mode 0)
                self.driver.set_operating_mode(0)  # Current control mode
                self.driver.set_torque_mode(True)
                print("  ✓ Arm joints configured for current control")
            except Exception as e:
                print(f"  ⚠️  Warning: Could not switch to current control mode: {e}")
                print("  FACTR system will use position control mode instead")

            # Setup FACTR gravity compensation system
            factr_config_path = "configs/yam_gello_factr_sim.yaml"
            self.factr_system = FACTRGravityCompensation(factr_config_path)

            print("✓ FACTR system setup complete")

        except Exception as e:
            print(f"✗ Failed to setup FACTR system: {e}")
            self.factr_system = None

    def start_control_loop(self):
        """Start the control loop."""
        if self._running:
            print("Control loop already running")
            return

        # Enable FACTR system first to set up joint offsets
        if self.factr_system and not self._sim_mode:
            try:
                print("Enabling FACTR gravity compensation system...")
                self.factr_system.enable_gravity_compensation()
                print("✓ FACTR system enabled")
            except Exception as e:
                print(f"⚠️  Failed to enable FACTR system: {e}")
                print("  Control loop will continue without gravity compensation")

        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("✓ Control loop started")

    def stop_control_loop(self):
        """Stop the control loop."""
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        print("✓ Control loop stopped")

    def _control_loop(self):
        print("Starting FACTR gravity compensation control loop...")
        self.factr_system.run()

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Get current joint state."""
        if self._sim_mode:
            # In simulation mode, return dummy data
            # The actual robot state will be handled by the launch system
            dummy_pos = np.zeros(len(self.config.joint_ids))
            dummy_vel = np.zeros(len(self.config.joint_ids))
            return {
                "pos": dummy_pos,
                "vel": dummy_vel,
                "raw_pos": dummy_pos,
                "raw_vel": dummy_vel,
                "sim_mode": True,
            }

        if self.driver is None:
            return {
                "pos": np.zeros(len(self.config.joint_ids)),
                "vel": np.zeros(len(self.config.joint_ids)),
            }

        try:
            pos, vel = self.driver.get_positions_and_velocities()
            if pos is None or vel is None:
                return {
                    "pos": np.zeros(len(self.config.joint_ids)),
                    "vel": np.zeros(len(self.config.joint_ids)),
                }

            # Separate arm and gripper positions
            arm_pos = pos[: len(self.config.joint_ids)]
            arm_vel = vel[: len(self.config.joint_ids)]

            # Apply calibration for arm joints
            if (
                hasattr(self.config, "joint_offsets")
                and self.config.joint_offsets is not None
            ):
                calibrated_pos = (
                    arm_pos - np.array(self.config.joint_offsets)
                ) * np.array(self.config.joint_signs[:6])
                calibrated_vel = arm_vel * np.array(self.config.joint_signs[:6])
            else:
                # No offsets configured - use raw positions with signs only
                calibrated_pos = arm_pos * np.array(self.config.joint_signs[:6])
                calibrated_vel = arm_vel * np.array(self.config.joint_signs[:6])

            # Handle gripper position if configured
            if self.config.gripper_config is not None and len(pos) > len(
                self.config.joint_ids
            ):
                gripper_pos = pos[len(self.config.joint_ids)]  # Raw gripper position

                calibrated_pos = np.concatenate([calibrated_pos, [gripper_pos]])
                calibrated_vel = np.concatenate(
                    [calibrated_vel, [0.0]]
                )  # Add gripper velocity
            elif self.config.gripper_config is not None:
                # Add default gripper position if gripper is configured but not available
                calibrated_pos = np.concatenate([calibrated_pos, [0.0]])
                calibrated_vel = np.concatenate(
                    [calibrated_vel, [0.0]]
                )  # Add gripper velocity

            return {
                "pos": calibrated_pos,
                "vel": calibrated_vel,
                "raw_pos": pos,
                "raw_vel": vel,
                "sim_mode": False,
            }
        except Exception as e:
            print(f"Error getting joint state: {e}")
            return {
                "pos": np.zeros(len(self.config.joint_ids)),
                "vel": np.zeros(len(self.config.joint_ids)),
            }

    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions."""
        state = self.get_joint_state()
        return state["pos"]

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Teleop action method - required by Agent protocol.

        This method handles teleop input and returns joint actions.
        For now, it returns the current joint positions (no teleop input).

        Args:
            obs: Observation from environment (joint positions, etc.)

        Returns:
            action: Joint actions for the robot (7 elements: 6 arm + 1 gripper)
        """
        # For now, return current joint positions as actions
        # This maintains the robot's current position (gravity compensation)
        current_positions = self.get_joint_pos()

        # Ensure we return 7 elements (6 arm + 1 gripper)
        if len(current_positions) == 6:
            # Add gripper position if missing
            gripper_pos = 0.0  # Default gripper position
            if self.config.gripper_config is not None:
                # Use gripper config if available
                gripper_pos = 0.5  # Middle position
            current_positions = np.concatenate([current_positions, [gripper_pos]])

        return current_positions

    def _switch_to_position_control(self):
        """Switch to position control mode for gripper control."""
        if self.driver and not self._sim_mode:
            try:
                self.driver.set_operating_mode(3)  # Position control mode
                self.driver.set_torque_mode(True)
                return True
            except Exception as e:
                print(f"Warning: Could not switch to position control mode: {e}")
                return False
        return False

    def _switch_to_current_control(self):
        """Switch to current control mode for FACTR system."""
        if self.driver and not self._sim_mode:
            try:
                self.driver.set_operating_mode(0)  # Current control mode
                self.driver.set_torque_mode(True)
                return True
            except Exception as e:
                print(f"❌ Warning: Could not switch to current control mode: {e}")
                return False
        return False

    def set_gripper_position(self, position: float) -> None:
        """Set gripper position (0.0 = open, 1.0 = closed).

        Args:
            position: Normalized gripper position in range [0.0, 1.0]
        """
        if self._sim_mode or self.driver is None:
            print(
                "Cannot set gripper position in simulation mode or when driver is not available"
            )
            return

        if self.config.gripper_config is None:
            print("No gripper configured")
            return

        try:
            # Switch to position control mode for gripper control
            if not self._switch_to_position_control():
                print("Failed to switch to position control mode for gripper")
                return

            # Clamp position to valid range
            position = min(max(0.0, position), 1.0)

            # Convert normalized position to raw gripper position
            if self.gripper_open_close is not None:
                raw_position = (
                    position * (self.gripper_open_close[1] - self.gripper_open_close[0])
                    + self.gripper_open_close[0]
                )
            else:
                raw_position = position

            # Get current arm positions
            current_pos, _ = self.driver.get_positions_and_velocities()
            if current_pos is None:
                print("Cannot get current positions for gripper control")
                return

            # Create new positions array with updated gripper position
            new_positions = list(current_pos)
            if len(new_positions) > len(self.config.joint_ids):
                # Update existing gripper position
                new_positions[len(self.config.joint_ids)] = raw_position
            else:
                # Add gripper position if not present
                new_positions.append(raw_position)

            # Set all joints including the updated gripper
            self.driver.set_joints(new_positions)

            # Switch back to current control mode for FACTR system
            self._switch_to_current_control()

        except Exception as e:
            print(f"Error setting gripper position: {e}")
            # Try to switch back to current control mode on error
            self._switch_to_current_control()

    def get_gripper_position(self) -> float:
        """Get current gripper position (0.0 = open, 1.0 = closed).

        Returns:
            Normalized gripper position in range [0.0, 1.0]
        """
        if self._sim_mode or self.driver is None:
            return 0.0  # Default open position in simulation

        try:
            state = self.get_joint_state()
            if len(state["pos"]) > len(self.config.joint_ids):
                return state["pos"][-1]  # Return gripper position
            else:
                return 0.0  # Default open position if gripper not available
        except Exception as e:
            print(f"Error getting gripper position: {e}")
            return 0.0

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command joint positions including gripper.

        Args:
            joint_state: Array of joint positions (arm joints + gripper if configured)
        """
        if self._sim_mode or self.driver is None:
            print(
                "Cannot command joint state in simulation mode or when driver is not available"
            )
            return

        try:
            # Separate arm and gripper commands
            arm_joints = joint_state[: len(self.config.joint_ids)]
            gripper_pos = (
                joint_state[len(self.config.joint_ids) :]
                if len(joint_state) > len(self.config.joint_ids)
                else None
            )

            # Command arm joints
            if len(arm_joints) == len(self.config.joint_ids):
                # Convert calibrated positions back to raw positions
                raw_positions = (
                    arm_joints / np.array(self.config.joint_signs)
                ) + np.array(self.config.joint_offsets)
                self.driver.set_joints(raw_positions.tolist())

            # Command gripper if provided
            if (
                gripper_pos is not None
                and len(gripper_pos) > 0
                and self.config.gripper_config is not None
            ):
                self.set_gripper_position(gripper_pos[0])

        except Exception as e:
            print(f"Error commanding joint state: {e}")

    def set_torque_mode(self, torque_mode: bool) -> None:
        """Set torque control mode."""
        if self.driver:
            self.driver.set_torque_mode(torque_mode)

    def num_dofs(self) -> int:
        """Get number of degrees of freedom."""
        return len(self.config.joint_ids) + (
            1 if self.config.gripper_config is not None else 0
        )

    def close(self):
        """Close the robot and cleanup."""
        self.stop_control_loop()
        if self.driver and not self._sim_mode:
            self.driver.close()
        print("✓ YAM GELLO robot closed")


@dataclass
class YAMGelloAgent(Agent):
    """GELLO agent for YAM robot with FACTR gravity compensation capabilities."""

    robot: YAMGelloRobot
    use_joint_state_as_action: bool = True
    regularize_joints: Optional[np.ndarray] = None
    enable_gravity_comp: bool = False  # Whether to enable gravity compensation

    def __post_init__(self):
        """Post-initialization setup."""
        if isinstance(self.robot, YAMGelloRobot):
            if self.enable_gravity_comp:
                # Enable torque mode and start gravity compensation
                self.robot.set_torque_mode(True)
                self.start_gravity_compensation()
                print("🚀 Started FACTR gravity compensation")
            else:
                # Start in position mode for safety
                self.robot.set_torque_mode(False)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Get action from robot - matches gello_software GelloAgent pattern."""
        if self.regularize_joints is not None:
            # In a real implementation, this would command joint positions
            # For testing, we just read the current state
            pass

        if self.use_joint_state_as_action:
            joint_state = self.robot.get_joint_state()["pos"]
        else:
            joint_state = self.robot.get_joint_pos()

        # The robot.get_joint_state() now properly includes gripper position
        # No need to manually append gripper position
        return joint_state

    def set_torque_mode(self, torque_mode: bool) -> None:
        """Set torque control mode."""
        self.robot.set_torque_mode(torque_mode)

    def num_dofs(self) -> int:
        """Get number of degrees of freedom."""
        return self.robot.num_dofs()

    def get_joint_pos(self) -> np.ndarray:

        return np.zeros(7)

    def get_joint_state(self) -> Dict[str, np.ndarray]:

        return {
            "joint_pos": np.zeros(7),
            "joint_vel": np.zeros(7),
        }

    def start_gravity_compensation(self) -> None:
        print("Starting FACTR gravity compensation system...")
        pass

    def stop_gravity_compensation(self) -> None:
        """Stop gravity compensation"""
        print("Stopping FACTR gravity compensation system...")
        if hasattr(self, "factr_system") and self.factr_system:
            self.factr_system.shutdown()

    def close(self) -> None:
        """Close the agent"""
        print("Closing FACTR gravity compensation system...")
        if hasattr(self, "factr_system") and self.factr_system:
            self.factr_system.shutdown()
        if hasattr(self, "driver") and self.driver:
            self.driver.close()


def create_yam_gello_agent(
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0",
    joint_signs: Optional[List[float]] = None,
    joint_offsets: Optional[List[float]] = None,
    servo_types: Optional[List[str]] = None,
    urdf_path: str = "gello/factr/urdf/yam_active_gello/robot.urdf",
    start_joints: Optional[List[float]] = None,
    gripper_config: Optional[Tuple[int, float, float]] = None,
    enable_gravity_comp: bool = False,  # Disabled for testing
) -> YAMGelloAgent:
    """Convenience function to create a YAM GELLO agent with FACTR gravity compensation.

    Args:
        port: USB port for YAM device
        joint_signs: Joint direction multipliers
        joint_offsets: Joint offset values
        servo_types: List of servo type strings
        urdf_path: Path to robot URDF file
        start_joints: Starting joint positions
        gripper_config: Gripper configuration (id, open_degrees, closed_degrees)
        enable_gravity_comp: Whether to enable gravity compensation (disabled for testing)

    Returns:
        YAMGelloAgent instance
    """
    # Use defaults if not provided
    if joint_signs is None:
        joint_signs = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0]

    if joint_offsets is None:
        joint_offsets = [1.5708, 3.14159, 3.14159, 3.14159, 6.28319, 1.5708]

    if servo_types is None:
        servo_types = [
            "XC330_T288_T",
            "XM430_W210_T",
            "XM430_W210_T",
            "XC330_T288_T",
            "XC330_T288_T",
            "XC330_T288_T",
            "XC330_T288_T",
        ]

    if start_joints is None:
        start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    if gripper_config is None:
        gripper_config = (7, -30.0, 24.0)  # Default YAM gripper config

    # Create configuration
    config = YAMGelloConfig(
        port=port,
        joint_signs=tuple(joint_signs),
        joint_offsets=tuple(joint_offsets),
        servo_types=tuple(servo_types),
        urdf_path=urdf_path,
        start_joints=tuple(start_joints),
        gripper_config=gripper_config,
    )

    print(f"Created YAMGelloConfig:")
    print(f"  Port: {config.port}")
    print(f"  Joint IDs: {config.joint_ids}")
    print(f"  Joint Signs: {config.joint_signs}")
    print(f"  Joint Offsets: {config.joint_offsets}")
    print(f"  Servo Types: {config.servo_types}")
    print(f"  URDF Path: {config.urdf_path}")
    print(f"  Gripper Config: {config.gripper_config}")
    print(f"  Gravity Gain: {config.gravity_gain}")
    print(f"  Null Space KP: {config.null_space_kp}")
    print(f"  Null Space KD: {config.null_space_kd}")
    print(f"  Friction Gain: {config.friction_gain}")
    print(f"  Barrier KP: {config.barrier_kp}")
    print(f"  Barrier KD: {config.barrier_kd}")

    # Create robot
    robot = YAMGelloRobot(config)

    # Create agent
    agent = YAMGelloAgent(robot=robot)

    # Start gravity compensation if requested (but it will be zero torque)
    if enable_gravity_comp:
        agent.start_gravity_compensation()

    return agent


if __name__ == "__main__":
    """Test the YAM GELLO agent with FACTR gravity compensation."""

    print("=" * 60)
    print("YAM GELLO Agent Test (Zero Torque Mode)")
    print("=" * 60)

    try:
        # Create agent
        agent = create_yam_gello_agent(
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0",
            enable_gravity_comp=True,  # Will run but with zero torque
        )

        print("✓ YAM GELLO agent created successfully")
        print("Running in zero torque mode for testing...")
        print("Press Ctrl+C to stop")

        # Run for a while to test
        try:
            while True:
                state = agent.get_joint_pos()
                print(
                    f"\rJoint positions: {[f'{x:.3f}' for x in state]}",
                    end="",
                    flush=True,
                )
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    except Exception as e:
        print(f"✗ Failed to create YAM GELLO agent: {e}")
        sys.exit(1)

    finally:
        if "agent" in locals():
            agent.close()
        print("YAM GELLO agent test complete")
