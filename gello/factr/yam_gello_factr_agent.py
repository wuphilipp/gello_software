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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import os 

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gello.agents.agent import Agent
from gello.dynamixel.driver import DynamixelDriver
from gello.factr.gravity_compensation import FACTRGravityCompensation


@dataclass
class YAMGelloConfig:
    """Configuration for YAM GELLO agent with FACTR gravity compensation."""
    
    # Hardware configuration
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0"
    baudrate: int = 57600
    joint_ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    
    # Joint configuration
    joint_offsets: Tuple[float, ...] = (1.5708, 3.14159, 3.14159, 3.14159, 6.28319, 1.5708)
    joint_signs: Tuple[float, ...] = (1.0, -1.0, -1.0, -1.0, 1.0, 1.0)
    
    # Servo types for torque conversion
    servo_types: Tuple[str, ...] = ("XC330_T288_T", "XM430_W210_T", "XM430_W210_T", 
                                   "XC330_T288_T", "XC330_T288_T", "XC330_T288_T")
    
    # URDF configuration
    urdf_path: str = "gello/factr/urdf/yam_active_gello/robot.urdf"
    
    # FACTR gravity compensation parameters (non-zero for testing)
    control_frequency: float = 100.0  # Hz
    gravity_gain: float = 0.3  # Non-zero for testing (30% gravity compensation)
    null_space_kp: float = 0.05  # Non-zero for testing
    null_space_kd: float = 0.005  # Non-zero for testing
    friction_gain: float = 0.2  # Non-zero for testing
    barrier_kp: float = 0.3  # Non-zero for testing
    barrier_kd: float = 0.01  # Non-zero for testing
    
    # Gripper configuration
    gripper_config: Optional[Tuple[int, float, float]] = (7, -30.0, 24.0)  # ID: 7, open: -30°, closed: 24°
    
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
        self._sim_mode = False # Track simulation mode
        
        # Gripper configuration
        self.gripper_open_close = None
        if self.config.gripper_config is not None:
            # Convert degrees to radians for gripper open/close positions
            self.gripper_open_close = (
                self.config.gripper_config[1] * np.pi / 180,  # open position in radians
                self.config.gripper_config[2] * np.pi / 180,  # closed position in radians
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
                servo_types = list(self.config.servo_types)
                
                if self.config.gripper_config is not None:
                    joint_ids.append(self.config.gripper_config[0])
                    # Use XC330_T288_T for gripper (common gripper servo type)
                    servo_types.append("XC330_T288_T")
                
                self.driver = DynamixelDriver(
                    ids=joint_ids,
                    port=self.config.port,
                    servo_types=servo_types
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
                print("Setting up FACTR gravity compensation system in SIMULATION mode...")
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
            
            # Create FACTR gravity compensation system
            # Resolve URDF path to be absolute
            urdf_path = self.config.urdf_path
            if not os.path.isabs(urdf_path):
                # Resolve relative to gello_software directory
                gello_dir = Path(__file__).parent.parent.parent
                urdf_path = str(gello_dir / urdf_path)
            
            try:
                self.factr_system = FACTRGravityCompensation(
                    driver=self.driver,
                    servo_types=list(self.config.servo_types),
                    joint_signs=list(self.config.joint_signs),
                    urdf_path=urdf_path,
                    joint_offsets=list(self.config.joint_offsets),
                    gravity_gain=self.config.gravity_gain,
                    null_space_kp=self.config.null_space_kp,
                    null_space_kd=self.config.null_space_kd,
                    friction_gain=self.config.friction_gain,
                    barrier_kp=self.config.barrier_kp,
                    barrier_kd=self.config.barrier_kd
                )
            except Exception as e:
                print(f"⚠️  Failed to create FACTR system with URDF: {e}")
                print("  Falling back to simple model...")
                
                # Fallback to simple model if URDF fails
                self.factr_system = FACTRGravityCompensation(
                    driver=self.driver,
                    servo_types=list(self.config.servo_types),
                    joint_signs=list(self.config.joint_signs),
                    use_simple_model=True,  # Use simple model instead of URDF
                    joint_offsets=list(self.config.joint_offsets),
                    gravity_gain=self.config.gravity_gain,
                    null_space_kp=self.config.null_space_kp,
                    null_space_kd=self.config.null_space_kd,
                    friction_gain=self.config.friction_gain,
                    barrier_kp=self.config.barrier_kp,
                    barrier_kd=self.config.barrier_kd
                )
            
            # Debug: Check what model was loaded
            if self.factr_system:
                print(f"  FACTR System Details:")
                print(f"    URDF Path: {self.config.urdf_path}")
                print(f"    Use Simple Model: {getattr(self.factr_system, 'use_simple_model', 'Unknown')}")
                print(f"    Model DOF: {getattr(self.factr_system, 'model', None) is not None}")
                print(f"    Gravity Gain: {self.factr_system.gravity_gain}")
                print(f"    Null Space KP: {self.factr_system.null_space_kp}")
                print(f"    Null Space KD: {self.factr_system.null_space_kd}")
                print(f"    Friction Gain: {self.factr_system.friction_gain}")
                print(f"    Barrier KP: {self.factr_system.barrier_kp}")
                print(f"    Barrier KD: {self.factr_system.barrier_kd}")
            
            # Set YAM-specific joint limits (more conservative than Franka defaults)
            if self.factr_system:
                # YAM robot joint limits (conservative values in radians)
                yam_joint_limits_max = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]  # Conservative upper limits
                yam_joint_limits_min = [-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]  # Conservative lower limits
                
                # Update the FACTR system with YAM-specific limits
                self.factr_system.joint_limits_max = np.array(yam_joint_limits_max)
                self.factr_system.joint_limits_min = np.array(yam_joint_limits_min)
                self.factr_system.joint_limits_margin = 0.2  # Larger margin for safety
                
                # Set YAM-specific null space target (home position)
                yam_null_space_target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # YAM home position
                self.factr_system.null_space_target = np.array(yam_null_space_target)
            
            print("✓ FACTR system setup complete")
            
        except Exception as e:
            print(f"✗ Failed to setup FACTR system: {e}")
            self.factr_system = None
    
    def start_control_loop(self):
        """Start the control loop."""
        if self._running:
            print("Control loop already running")
            return
        
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
        """Main control loop for FACTR gravity compensation."""
        dt = 1.0 / self.config.control_frequency
        
        while self._running:
            start_time = time.time()
            
            try:
                # Get current joint states
                if self._sim_mode:
                    # In simulation mode, we can't control torques directly
                    # Just monitor the simulation
                    time.sleep(dt)
                    continue
                
                joint_pos_raw, joint_vel_raw = self.driver.get_positions_and_velocities()
                
                if joint_pos_raw is None or joint_vel_raw is None:
                    time.sleep(0.01)
                    continue
                
                # Separate arm and gripper data
                arm_pos_raw = joint_pos_raw[:len(self.config.joint_ids)]
                arm_vel_raw = joint_vel_raw[:len(self.config.joint_ids)]
                
                # Calculate torques using FACTR system (currently zero for testing)
                if self.factr_system and not self._sim_mode:
                    # Ensure we're in current control mode for FACTR
                    if hasattr(self.driver, 'get_operating_mode'):
                        try:
                            current_mode = self.driver.get_operating_mode()
                            if current_mode != 0:  # Not in current control mode
                                print(f"⚠️  Switching to current control mode (was mode {current_mode})")
                                self._switch_to_current_control()
                        except:
                            pass  # Ignore if get_operating_mode is not available
                    
                    # Use FACTR system for torque calculation (real robot only)
                    # Pass raw positions/velocities - FACTR system handles offsets and signs internally
                    torques = self.factr_system.compute_torques(arm_pos_raw, arm_vel_raw)
                    
                    # Debug: Print torque components to see what's happening
                    if hasattr(self.factr_system, 'gravity_gain') and self.factr_system.gravity_gain > 0:
                        print(f"\r🔍 FACTR Debug: gravity_gain={self.factr_system.gravity_gain:.3f}, "
                              f"null_kp={self.factr_system.null_space_kp:.3f}, "
                              f"null_kd={self.factr_system.null_space_kd:.3f}, "
                              f"friction_gain={self.factr_system.friction_gain:.3f}, "
                              f"barrier_kp={self.factr_system.barrier_kp:.3f}, "
                              f"barrier_kd={self.factr_system.barrier_kd:.3f}, "
                              f"torques={[f'{t:.4f}' for t in torques[:3]]}...", end="", flush=True)
                else:
                    # Zero torque mode for testing or simulation
                    torques = np.zeros(len(self.config.joint_ids))
                
                # Apply torques (will be zero in testing mode)
                try:
                    # Convert torques to driver format (including gripper if configured)
                    if self.config.gripper_config is not None:
                        # Add zero torque for gripper
                        gripper_torques = np.zeros(1)  # Zero torque for gripper
                        # FACTR system already applies joint signs, so don't apply them again
                        all_torques = np.concatenate([torques, gripper_torques])
                    else:
                        # FACTR system already applies joint signs, so don't apply them again
                        all_torques = torques
                    
                    self.driver.set_torque(all_torques.tolist())
                except Exception as e:
                    print(f"Warning: Failed to set torque: {e}")
                
                # Maintain loop timing
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"⚠️  Loop overrun: {elapsed - dt:.4f}s")
                    
            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.1)
                continue
    
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
                "sim_mode": True
            }
        
        if self.driver is None:
            return {"pos": np.zeros(len(self.config.joint_ids)), "vel": np.zeros(len(self.config.joint_ids))}
        
        try:
            pos, vel = self.driver.get_positions_and_velocities()
            if pos is None or vel is None:
                return {"pos": np.zeros(len(self.config.joint_ids)), "vel": np.zeros(len(self.config.joint_ids))}
            
            # Separate arm and gripper positions
            arm_pos = pos[:len(self.config.joint_ids)]
            arm_vel = vel[:len(self.config.joint_ids)]
            
            # Apply calibration for arm joints
            calibrated_pos = (arm_pos - np.array(self.config.joint_offsets)) * np.array(self.config.joint_signs)
            calibrated_vel = arm_vel * np.array(self.config.joint_signs)
            
            # Handle gripper position if configured
            if self.config.gripper_config is not None and len(pos) > len(self.config.joint_ids):
                gripper_pos = pos[len(self.config.joint_ids)]  # Raw gripper position
                # Map gripper position to normalized [0, 1] range
                if self.gripper_open_close is not None:
                    g_pos = (gripper_pos - self.gripper_open_close[0]) / (
                        self.gripper_open_close[1] - self.gripper_open_close[0]
                    )
                    g_pos = min(max(0, g_pos), 1)  # Clamp to [0, 1]
                    calibrated_pos = np.concatenate([calibrated_pos, [g_pos]])
                else:
                    calibrated_pos = np.concatenate([calibrated_pos, [gripper_pos]])
            elif self.config.gripper_config is not None:
                # Add default gripper position if gripper is configured but not available
                calibrated_pos = np.concatenate([calibrated_pos, [0.0]])
            
            return {
                "pos": calibrated_pos,
                "vel": calibrated_vel,
                "raw_pos": pos,
                "raw_vel": vel,
                "sim_mode": False
            }
        except Exception as e:
            print(f"Error getting joint state: {e}")
            return {"pos": np.zeros(len(self.config.joint_ids)), "vel": np.zeros(len(self.config.joint_ids))}
    
    def get_joint_pos(self) -> np.ndarray:
        """Get current joint positions."""
        state = self.get_joint_state()
        return state["pos"]
    
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
                print(f"Warning: Could not switch to current control mode: {e}")
                return False
        return False
    
    def set_gripper_position(self, position: float) -> None:
        """Set gripper position (0.0 = open, 1.0 = closed).
        
        Args:
            position: Normalized gripper position in range [0.0, 1.0]
        """
        if self._sim_mode or self.driver is None:
            print("Cannot set gripper position in simulation mode or when driver is not available")
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
                raw_position = (position * (self.gripper_open_close[1] - self.gripper_open_close[0]) + 
                              self.gripper_open_close[0])
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
            print("Cannot command joint state in simulation mode or when driver is not available")
            return
        
        try:
            # Separate arm and gripper commands
            arm_joints = joint_state[:len(self.config.joint_ids)]
            gripper_pos = joint_state[len(self.config.joint_ids):] if len(joint_state) > len(self.config.joint_ids) else None
            
            # Command arm joints
            if len(arm_joints) == len(self.config.joint_ids):
                # Convert calibrated positions back to raw positions
                raw_positions = (arm_joints / np.array(self.config.joint_signs)) + np.array(self.config.joint_offsets)
                self.driver.set_joints(raw_positions.tolist())
            
            # Command gripper if provided
            if gripper_pos is not None and len(gripper_pos) > 0 and self.config.gripper_config is not None:
                self.set_gripper_position(gripper_pos[0])
                
        except Exception as e:
            print(f"Error commanding joint state: {e}")
    
    def set_torque_mode(self, torque_mode: bool) -> None:
        """Set torque control mode."""
        if self.driver:
            self.driver.set_torque_mode(torque_mode)
    
    def num_dofs(self) -> int:
        """Get number of degrees of freedom."""
        return len(self.config.joint_ids) + (1 if self.config.gripper_config is not None else 0)
    
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
    
    def __post_init__(self):
        """Post-initialization setup."""
        if isinstance(self.robot, YAMGelloRobot):
            self.robot.set_torque_mode(False)  # Start in position mode for safety
    
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
        """Get current joint positions."""
        return self.robot.get_joint_pos()
    
    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Get current joint state."""
        return self.robot.get_joint_state()
    
    def start_gravity_compensation(self) -> None:
        """Start FACTR gravity compensation control loop."""
        self.robot.start_control_loop()
    
    def stop_gravity_compensation(self) -> None:
        """Stop FACTR gravity compensation control loop."""
        self.robot.stop_control_loop()
    
    def close(self) -> None:
        """Close the agent and cleanup."""
        self.robot.close()


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
        joint_signs = [1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
    
    if joint_offsets is None:
        joint_offsets = [1.5708, 3.14159, 3.14159, 3.14159, 6.28319, 1.5708]
    
    if servo_types is None:
        servo_types = ["XC330_T288_T", "XM430_W210_T", "XM430_W210_T", 
                      "XC330_T288_T", "XC330_T288_T", "XC330_T288_T"]
    
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
        gripper_config=gripper_config
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
    
    print("="*60)
    print("YAM GELLO Agent Test (Zero Torque Mode)")
    print("="*60)
    
    try:
        # Create agent
        agent = create_yam_gello_agent(
            port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0",
            enable_gravity_comp=True  # Will run but with zero torque
        )
        
        print("✓ YAM GELLO agent created successfully")
        print("Running in zero torque mode for testing...")
        print("Press Ctrl+C to stop")
        
        # Run for a while to test
        try:
            while True:
                state = agent.get_joint_pos()
                print(f"\rJoint positions: {[f'{x:.3f}' for x in state]}", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        
    except Exception as e:
        print(f"✗ Failed to create YAM GELLO agent: {e}")
        sys.exit(1)
    
    finally:
        if 'agent' in locals():
            agent.close()
        print("YAM GELLO agent test complete") 