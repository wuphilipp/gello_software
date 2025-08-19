"""
FACTR Gravity Compensation for GELLO Software

This module provides gravity compensation for GELLO devices using FACTR's
inverse dynamics approach. It integrates with the existing gello_software
architecture to provide active force feedback.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from ..dynamixel.driver import (
    CURRENT_CONTROL_MODE,
    POSITION_CONTROL_MODE,
    SERVO_CURRENT_LIMITS,
    TORQUE_TO_CURRENT_MAPPING,
    DynamixelDriver,
)


class FACTRGravityCompensation:
    """
    FACTR gravity compensation system for GELLO devices.
    
    This class provides:
    - Gravity compensation using inverse dynamics
    - Null-space regulation
    - Joint limit barriers
    - Static friction compensation
    """

    def __init__(
        self,
        driver: DynamixelDriver,
        servo_types: Optional[List[str]] = None,
        joint_signs: List[int] = None,
        urdf_path: str = "gello/factr/urdf/factr_teleop_franka.urdf",
        frequency: float = 500.0,
        gravity_gain: float = 0.8,
        null_space_kp: float = 0.1,
        null_space_kd: float = 0.01,
        friction_gain: float = 0.6,
        friction_enable_speed: float = 0.9,
        barrier_kp: float = 1.0,
        barrier_kd: float = 0.04,
        use_simple_model: bool = False,
        joint_offsets: Optional[List[float]] = None,
        auto_zero_on_start: bool = True,  # capture current pose as zero offsets on start
        simple_k: Optional[List[float]] = None,  # Simple model gains
        simple_theta0: Optional[List[float]] = None,  # Simple model reference angles
    ):
        """Initialize the FACTR gravity compensation system.
        
        Args:
            driver: DynamixelDriver instance
            servo_types: List of servo type strings (e.g., "XC330_T288_T")
            joint_signs: Joint direction multipliers
            urdf_path: Path to robot URDF file
            frequency: Control loop frequency (Hz)
            gravity_gain: Gravity compensation strength (0.0-1.0)
            null_space_kp: Null space position gain
            null_space_kd: Null space damping gain
            friction_gain: Static friction compensation gain
            friction_enable_speed: Velocity threshold for friction compensation
            barrier_kp: Joint limit barrier stiffness
            barrier_kd: Joint limit barrier damping
        """
        self.driver = driver
        self.servo_types = servo_types
        self.joint_signs = np.array(joint_signs if joint_signs is not None else [])
        self.running = False
        
        # Determine DOF
        self.num_arm_joints = len(self.servo_types) if self.servo_types is not None else len(self.joint_signs)
        
        # Joint offsets to zero current pose if provided
        self.joint_offsets = np.array(joint_offsets[: self.num_arm_joints], dtype=float) if joint_offsets is not None else None
        
        # Control parameters
        self.dt = 1.0 / frequency
        self.enable_gravity_comp = True
        # Conservative defaults for safety
        self.gravity_gain = min(gravity_gain, 0.5)
        self.null_space_kp = min(null_space_kp, 0.05)
        self.null_space_kd = min(null_space_kd, 0.005)
        self.friction_gain = min(friction_gain, 0.3)
        self.friction_enable_speed = friction_enable_speed
        self.barrier_kp = min(barrier_kp, 0.5)
        self.barrier_kd = min(barrier_kd, 0.02)
        
        # Default null space target (Franka home position)
        self.null_space_target = np.array([0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0])
        
        # Default joint limits (Franka limits)
        self.joint_limits_max = np.array([2.8973, 1.7628, 2.8973, -0.8698, 2.8973, 3.7525, 2.8973])
        self.joint_limits_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_limits_margin = 0.1
        
        # Initialize model options
        self.use_simple_model = use_simple_model
        if self.use_simple_model:
            # Initialize simple model params
            if simple_k is None:
                # conservative default per-joint Nm scale
                self.simple_k = np.full(self.num_arm_joints, 0.2, dtype=float)
            else:
                self.simple_k = np.array(simple_k[: self.num_arm_joints], dtype=float)
            if simple_theta0 is None:
                self.simple_theta0 = np.zeros(self.num_arm_joints, dtype=float)
            else:
                self.simple_theta0 = np.array(simple_theta0[: self.num_arm_joints], dtype=float)
        
        # Setup inverse dynamics if not using simple model
        if not self.use_simple_model:
            self._setup_inverse_dynamics(urdf_path)
        
        # Initialize state tracking
        # self.num_arm_joints already set above
        
        # Adjust arrays to match actual number of joints
        if len(self.null_space_target) > self.num_arm_joints:
            self.null_space_target = self.null_space_target[:self.num_arm_joints]
        if len(self.joint_limits_max) > self.num_arm_joints:
            self.joint_limits_max = self.joint_limits_max[:self.num_arm_joints]
        if len(self.joint_limits_min) > self.num_arm_joints:
            self.joint_limits_min = self.joint_limits_min[:self.num_arm_joints]
            
        self._last_positions = None
        self._last_velocities = None
        self._velocity_filter_alpha = 0.8
        
        print(f"FACTR Gravity Compensation initialized for {self.num_arm_joints} joints")

    def _setup_inverse_dynamics(self, urdf_path: str):
        """Setup Pinocchio model for inverse dynamics."""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
            
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        print(f"Loaded robot model with {self.model.nq} DOF")

    def get_joint_states(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get current joint positions and velocities (FACTR-style combined read)."""
        pos, vel = self.driver.get_positions_and_velocities()
        if self.joint_offsets is None:
            offsets = np.zeros(self.num_arm_joints)
        else:
            offsets = self.joint_offsets
        positions = (pos[:self.num_arm_joints] - offsets) * self.joint_signs
        raw_velocities = vel[:self.num_arm_joints] * self.joint_signs
        
        # Apply velocity filtering
        if self._last_velocities is not None:
            velocities = (
                self._velocity_filter_alpha * self._last_velocities
                + (1 - self._velocity_filter_alpha) * raw_velocities
            )
        else:
            velocities = raw_velocities
            
        self._last_positions = positions
        self._last_velocities = velocities
        
        return positions, velocities

    def gravity_compensation(
        self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute gravity compensation torques."""
        if self.use_simple_model:
            # Per-joint sinusoidal gravity model (URDF-free)
            tau = self.simple_k * np.sin(positions - self.simple_theta0)
            return self.gravity_gain * tau
        
        # Pinocchio inverse dynamics model
        q = np.zeros(self.model.nq)
        q[:len(positions)] = positions
        
        v = np.zeros(self.model.nv)
        v[:len(velocities)] = velocities
        
        tau_gravity = pin.computeGeneralizedGravity(self.model, self.data, q)
        return self.gravity_gain * tau_gravity[:self.num_arm_joints]

    def friction_compensation(self, velocities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute static friction compensation."""
        friction_torques = np.zeros_like(velocities)
        
        for i, vel in enumerate(velocities):
            if abs(vel) < self.friction_enable_speed:
                friction_torques[i] = self.friction_gain * np.sign(vel)
                
        return friction_torques

    def null_space_regulation(
        self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute null space regulation torques."""
        pos_error = self.null_space_target - positions
        return self.null_space_kp * pos_error - self.null_space_kd * velocities

    def joint_limit_barrier(
        self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute joint limit barrier torques."""
        torques = np.zeros_like(positions)
        
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            # Upper limit barrier
            if pos > self.joint_limits_max[i] - self.joint_limits_margin:
                if vel > 0:  # Moving towards limit
                    torques[i] -= self.barrier_kp * (pos - (self.joint_limits_max[i] - self.joint_limits_margin))
                    torques[i] -= self.barrier_kd * vel
                    
            # Lower limit barrier  
            if pos < self.joint_limits_min[i] + self.joint_limits_margin:
                if vel < 0:  # Moving towards limit
                    torques[i] -= self.barrier_kp * (pos - (self.joint_limits_min[i] + self.joint_limits_margin))
                    torques[i] -= self.barrier_kd * vel
                    
        return torques

    def compute_torques(
        self, positions: npt.NDArray[np.float64], velocities: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute combined torques for all compensation types.
        
        Args:
            positions: Current joint positions
            velocities: Current joint velocities
            
        Returns:
            Combined torque command for all joints
        """
        # Initialize torque command
        torques = np.zeros(self.num_arm_joints)
        
        # 1. Gravity compensation
        if self.enable_gravity_comp:
            torques += self.gravity_compensation(positions, velocities)
        
        # 2. Null space regulation
        torques += self.null_space_regulation(positions, velocities)
        
        # 3. Joint limit barriers
        torques += self.joint_limit_barrier(positions, velocities)
        
        # 4. Friction compensation
        torques += self.friction_compensation(velocities)
        
        # Apply joint signs to torques
        torques = torques * self.joint_signs
        
        # Clamp torques before returning (safety)
        torque_limit = 0.2  # Nm-equivalent budget per joint; adjust up later
        torques = np.clip(torques, -torque_limit, torque_limit)
        
        return torques

    def torque_to_current(self, torques: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert torques to servo currents."""
        currents = np.zeros_like(torques)
        
        if self.servo_types is None:
            # Default to XC330 conversion for all joints if types unknown
            default_factor = TORQUE_TO_CURRENT_MAPPING.get("XC330_T288_T", 1000.0)
            default_limit = SERVO_CURRENT_LIMITS.get("XC330_T288_T", 1000.0)
            currents = np.clip(torques * default_factor, -default_limit, default_limit)
            return currents
        
        for i, (torque, servo_type) in enumerate(zip(torques, self.servo_types)):
            if servo_type in TORQUE_TO_CURRENT_MAPPING:
                current = torque * TORQUE_TO_CURRENT_MAPPING[servo_type]
                
                # Apply current limits
                if servo_type in SERVO_CURRENT_LIMITS:
                    limit = SERVO_CURRENT_LIMITS[servo_type]
                    current = np.clip(current, -limit, limit)
                    
                currents[i] = current
            else:
                # Fallback to default factor if unknown type
                default_factor = TORQUE_TO_CURRENT_MAPPING.get("XC330_T288_T", 1000.0)
                default_limit = SERVO_CURRENT_LIMITS.get("XC330_T288_T", 1000.0)
                currents[i] = np.clip(torque * default_factor, -default_limit, default_limit)
        
        return currents

    def control_step(self):
        """Execute one control step."""
        try:
            # Get current joint states
            positions, velocities = self.get_joint_states()
            
            # Initialize torque command
            torques = np.zeros(self.num_arm_joints)
            
            # Gravity only for initial safety
            if self.enable_gravity_comp:
                torques += self.gravity_compensation(positions, velocities)
            
            # Apply small nullspace and joint-limit softly
            torques += self.null_space_regulation(positions, velocities)
            torques += self.joint_limit_barrier(positions, velocities)
            
            # Apply joint signs to torques
            torques = torques * self.joint_signs
            
            # Clamp torques before sending (safety)
            torque_limit = 0.2  # Nm-equivalent budget per joint; adjust up later
            torques = np.clip(torques, -torque_limit, torque_limit)
            
            # FACTR-style: send torques directly; driver converts to current
            self.driver.set_torque(torques)
            
        except Exception as e:
            print(f"Control step error: {e}")

    def enable_gravity_compensation(self):
        """Enable gravity compensation mode."""
        print("Enabling gravity compensation mode...")
        
        # Capture current pose as zero offsets if requested
        if getattr(self, "joint_offsets", None) is None:
            pos, _ = self.driver.get_positions_and_velocities()
            self.joint_offsets = pos[: self.num_arm_joints]
            print(f"Auto-zeroed joint offsets: {[f'{x:.3f}' for x in self.joint_offsets]}")
        
        # Switch to current control with synchronous I/O to avoid bus contention
        if hasattr(self.driver, "enter_factr_mode"):
            self.driver.enter_factr_mode()
        else:
            self.driver.set_operating_mode(CURRENT_CONTROL_MODE)
            self.driver.set_torque_mode(True)
        
        print("Gravity compensation enabled")

    def disable_gravity_compensation(self):
        """Disable gravity compensation mode."""
        print("Disabling gravity compensation mode...")
        
        # Switch back to position control mode
        self.driver.set_operating_mode(POSITION_CONTROL_MODE)
        
        print("Gravity compensation disabled")

    def run(self):
        """Run the gravity compensation control loop."""
        print(f"Starting FACTR gravity compensation at {1/self.dt:.1f} Hz")
        print("Press Ctrl+C to stop")
        
        self.enable_gravity_compensation()
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                self.control_step()
                
                # Maintain timing
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

    def shutdown(self):
        """Safely shutdown the system."""
        self.running = False
        try:
            self.disable_gravity_compensation()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        print("FACTR gravity compensation shutdown complete")


 