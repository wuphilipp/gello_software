import queue
from functools import partial
from typing import Callable, Dict, Tuple

import numpy as np


class JointMapper:
    def __init__(self, index_range_map: Dict[int, Tuple[float, float]], total_dofs: int):
        """_summary_
        This class is used to map the joint positions from the command space to the robot joint space.

        Args:
            index_range_map (Dict[int, Tuple[float, float]]): 0 indexed
            total_dofs (int): num of joints in the robot including the gripper if the girpper is the second robot
        """
        self.empty = len(index_range_map) == 0
        if not self.empty:
            self.joints_one_hot = np.zeros(total_dofs).astype(bool)
            self.joint_limits = []
            for idx, (start, end) in index_range_map.items():
                self.joints_one_hot[idx] = True
                self.joint_limits.append((start, end))
            self.joint_limits = np.array(self.joint_limits)
            self.joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]

    def to_robot_joint_pos_space(self, command_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_pos
        result = command_joint_pos.copy()
        needs_remapping = command_joint_pos[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range + self.joint_limits[:, 0]
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_robot_joint_vel_space(self, command_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_vel
        result = command_joint_vel.copy()
        needs_remapping = command_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_vel_space(self, robot_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_vel
        result = robot_joint_vel.copy()
        needs_remapping = robot_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_pos_space(self, robot_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_pos
        result = robot_joint_pos.copy()
        needs_remapping = robot_joint_pos[self.joints_one_hot]
        needs_remapping = (needs_remapping - self.joint_limits[:, 0]) / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result


def linear_gripper_force_torque_map(
    motor_stroke: float, gripper_stroke: float, gripper_force: float, current_angle: float
) -> float:
    """Maps the motor stroke required to achieve a given gripper force.

    Args:
        motor_stroke (float): in rad
        gripper_stroke (float): in meter
        gripper_force (float): in newton
    """
    # force = torque * motor_stroke / gripper_stroke
    return gripper_force * gripper_stroke / motor_stroke


def zero_linkage_crank_gripper_force_torque_map(
    gripper_close_angle: float,
    gripper_open_angle: float,
    motor_reading_to_crank_angle: Callable[[float], float],
    gripper_stroke: float,
    current_angle: float,
    gripper_force: float,
) -> float:
    """Maps the motor crank torque required to achieve a given gripper force. For Yam style gripper (zero linkage crank)

    Args:
        gripper_close_angle (float): Angle of the crank in radians at the closed position.
        gripper_open_angle (float): Angle of the crank in radians at the open position.
        gripper_stroke (float): Linear displacement of the gripper in meters.
        current_angle (float): Current crank angle in radians (relative to the closed position).
        gripper_force (float): Required gripping force in Newtons (N).

    Returns:
        float: Required motor torque in Newton-meters (Nm).
    """
    current_angle = motor_reading_to_crank_angle(current_angle)
    # Compute crank radius based on the total stroke and angle change
    crank_radius = gripper_stroke / (2 * (np.cos(gripper_close_angle) - np.cos(gripper_open_angle)))
    # gripper_position = crank_radius * (np.cos(gripper_close_angle) - np.cos(current_angle))
    grad_gripper_position = crank_radius * np.sin(current_angle)

    # Compute the required torque
    target_torque = gripper_force * grad_gripper_position
    return target_torque


class GripperForceLimiter:
    def __init__(
        self,
        max_force: float,
        gripper_type: str,
        kp: float,
        debug: bool = False,
    ):
        self.max_force = max_force
        self.gripper_type = gripper_type
        self._is_clogged = False
        self._gripper_adjusted_qpos = None
        self._kp = kp
        self._past_gripper_effort_queue = queue.Queue(maxsize=50)
        self.debug = debug
        if self.gripper_type == "arx_92mm_linear":
            self.clog_force_threshold = 0.5
            self.clog_speed_threshold = 0.2
            self.sign = 1.0
            self.gripper_force_torque_map = partial(
                linear_gripper_force_torque_map,
                motor_stroke=4.93,
                gripper_stroke=0.092,
                gripper_force=self.max_force,
            )
        elif self.gripper_type == "yam_small":
            self.clog_force_threshold = 0.5
            self.clog_speed_threshold = 0.3
            self.sign = 1.0
            self.gripper_force_torque_map = partial(
                zero_linkage_crank_gripper_force_torque_map,
                motor_reading_to_crank_angle=lambda x: (-x + 0.174),
                gripper_close_angle=8 / 180.0 * np.pi,
                gripper_open_angle=170 / 180.0 * np.pi,
                gripper_stroke=0.071,  # unit in meter
                gripper_force=self.max_force,
            )
        else:
            raise ValueError(f"Unknown gripper type: {self.gripper_type}")

    def compute_target_gripper_torque(self, gripper_state: Dict[str, float]) -> float:
        current_speed = gripper_state["current_qvel"]
        average_effort = np.abs(np.mean(self._past_gripper_effort_queue.queue))
        if self.debug:
            print(f"average_effort: {average_effort}")

        if self._is_clogged:
            normalized_current_qpos = gripper_state["current_normalized_qpos"]
            normalized_target_qpos = gripper_state["target_normalized_qpos"]
            # 0 close 1 open
            if (normalized_current_qpos < normalized_target_qpos) or average_effort < 0.2:  # want to open
                self._is_clogged = False
        else:
            if average_effort > self.clog_force_threshold and np.abs(current_speed) < self.clog_speed_threshold:
                self._is_clogged = True

        if self._is_clogged:
            target_eff = self.gripper_force_torque_map(current_angle=gripper_state["current_qpos"])
            self._is_clogged = True
            return target_eff + 0.3  # this is to compensate the friction
        else:
            return None

    def update(self, gripper_state: Dict[str, float]) -> None:
        if self._past_gripper_effort_queue.full():
            self._past_gripper_effort_queue.get()
        self._past_gripper_effort_queue.put(gripper_state["current_eff"])
        target_eff = self.compute_target_gripper_torque(gripper_state)

        if target_eff is not None:
            command_sign = np.sign(gripper_state["target_qpos"] - gripper_state["current_qpos"]) * self.sign

            current_zero_eff_pos = (
                gripper_state["last_command_qpos"] - command_sign * np.abs(gripper_state["current_eff"]) / self._kp
            )
            target_gripper_raw_pos = current_zero_eff_pos + command_sign * np.abs(target_eff) / self._kp
            if self.debug:
                print("clogged")
                print(f"gripper_state: {gripper_state}")
                print("current zero eff")
                print(current_zero_eff_pos)
                print(f"target_gripper_raw_pos: {target_gripper_raw_pos}")
            # Update gripper target position
            a = 0.1
            self._gripper_adjusted_qpos = (1 - a) * self._gripper_adjusted_qpos + a * target_gripper_raw_pos
            return self._gripper_adjusted_qpos
        else:
            if self.debug:
                print("unclogged")
            self._gripper_adjusted_qpos = gripper_state["current_qpos"]
            return gripper_state["target_qpos"]
