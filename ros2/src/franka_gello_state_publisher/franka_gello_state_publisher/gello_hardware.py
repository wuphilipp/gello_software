import numpy as np
from ament_index_python.packages import get_package_prefix
from os import path as os_path
from sys import path as sys_path
from dataclasses import dataclass, field
from typing import List, TypedDict, Iterator, Tuple


class GelloHardwareParams(TypedDict):
    """Type-safe parameter dictionary for GelloHardware initialization."""

    com_port: str
    gello_name: str
    num_arm_joints: int
    joint_signs: List[int]
    gripper: bool
    gripper_range_rad: List[float]
    best_offsets: List[float]
    dynamixel_kp_p: List[int]
    dynamixel_kp_i: List[int]
    dynamixel_kp_d: List[int]
    dynamixel_torque_enable: List[int]
    dynamixel_goal_position: List[float]


@dataclass
class DynamixelControlConfig:
    """Tracks current dynamixel parameter state."""

    kp_p: List[int] = field(default_factory=list)
    kp_i: List[int] = field(default_factory=list)
    kp_d: List[int] = field(default_factory=list)
    torque_enable: List[int] = field(default_factory=list)
    goal_position: List[int] = field(default_factory=list)
    goal_current: List[int] = field(default_factory=list)
    operating_mode: List[int] = field(default_factory=list)

    # Define the order for parameter updates
    _UPDATE_ORDER = [
        "operating_mode",  # resets kp_p, kp_i, kp_d, goal_current, goal_position
        "goal_current",
        "kp_p",
        "kp_i",
        "kp_d",
        "torque_enable",  # resets goal_position
        "goal_position",
    ]

    def __contains__(self, param_name: str) -> bool:
        """Check if parameter exists in this configuration."""
        return hasattr(self, param_name)

    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
        """Iterate through parameters in correct update order."""
        for param_name in self._UPDATE_ORDER:
            if hasattr(self, param_name):
                yield param_name, getattr(self, param_name)

    def __getitem__(self, param_name: str) -> List[int]:
        """Enable dictionary-style access for getting values."""
        if not hasattr(self, param_name):
            raise KeyError(f"Parameter '{param_name}' not found")
        return getattr(self, param_name)

    def __setitem__(self, param_name: str, value: List[int]) -> None:
        """Enable dictionary-style access for setting values."""
        if not hasattr(self, param_name):
            raise KeyError(f"Parameter '{param_name}' not found")
        setattr(self, param_name, value)


class GelloHardware:
    """Hardware interface for GELLO teleoperation device."""

    def __init__(
        self,
        hardware_config: GelloHardwareParams,
    ) -> None:
        self._com_port = hardware_config["com_port"]
        self._gello_name = hardware_config["gello_name"]
        self._num_arm_joints = hardware_config["num_arm_joints"]
        self._joint_signs = np.array(hardware_config["joint_signs"])
        self._gripper = hardware_config["gripper"]
        self._num_total_joints = self._num_arm_joints + (1 if self._gripper else 0)
        self._gripper_range_rad = hardware_config["gripper_range_rad"]
        self._best_offsets = np.array(hardware_config["best_offsets"])

        self._initialize_driver()

        OPERATING_MODE = 5  # CURRENT_BASED_POSITION_MODE
        CURRENT_LIMIT = 600  # mA
        self._dynamixel_control_config = DynamixelControlConfig(
            kp_p=hardware_config["dynamixel_kp_p"].copy(),
            kp_i=hardware_config["dynamixel_kp_i"].copy(),
            kp_d=hardware_config["dynamixel_kp_d"].copy(),
            torque_enable=hardware_config["dynamixel_torque_enable"].copy(),
            goal_position=self._goal_position_to_pulses(
                hardware_config["dynamixel_goal_position"]
            ).copy(),
            goal_current=[CURRENT_LIMIT] * self._num_total_joints,
            operating_mode=[OPERATING_MODE] * self._num_total_joints,
        )

        self._initialize_parameters()

    def _add_dynamixel_driver_path(self) -> None:
        """Add dynamixel driver to Python path."""
        gello_path = os_path.abspath(
            os_path.join(get_package_prefix("franka_gello_state_publisher"), "../../../")
        )
        sys_path.insert(0, gello_path)

    def _initialize_driver(self) -> None:
        """Initialize dynamixel driver with joint IDs and port."""
        joint_ids = list(range(1, self._num_total_joints + 1))
        self._add_dynamixel_driver_path()
        from gello.dynamixel.driver import DynamixelDriver

        self._driver = DynamixelDriver(joint_ids, port=self._com_port, baudrate=57600)

    def _initialize_parameters(self) -> None:
        """Write all dynamixel configuration parameters to hardware."""
        for param_name, param_value in self._dynamixel_control_config:
            self._driver.write_value_by_name(param_name, param_value)

    def update_dynamixel_control_parameter(
        self, param_name: str, param_value: list[float] | list[int]
    ) -> None:
        """Update a single dynamixel parameter and handle dependencies."""
        clean_name = param_name.replace("dynamixel_", "")

        if clean_name == "goal_position":
            param_value = self._goal_position_to_pulses(param_value)

        self._dynamixel_control_config[clean_name] = param_value
        self._driver.write_value_by_name(clean_name, self._dynamixel_control_config[clean_name])
        if clean_name == "torque_enable":
            self._driver.write_value_by_name(
                "goal_position", self._dynamixel_control_config["goal_position"]
            )

    def read_joint_states(self) -> tuple[np.ndarray, float]:
        """Read current joint positions and gripper state."""
        gello_joints_raw = self._driver.get_joints()
        gello_arm_joints_raw = gello_joints_raw[: self._num_arm_joints]
        gello_arm_joints = (gello_arm_joints_raw - self._best_offsets) * self._joint_signs

        gripper_position = 0.0
        if self._gripper:
            gripper_position = self._gripper_readout_to_percent(gello_joints_raw[-1])

        return gello_arm_joints, gripper_position

    def disable_torque(self) -> None:
        """Disable torque on all joints."""
        self._driver.set_torque_mode(False)

    def _goal_position_to_pulses(self, goals: list[float]) -> list[int]:
        """Convert goal positions from radians to dynamixel pulses."""
        arm_goals_raw = (goals[: self._num_arm_joints] * self._joint_signs) + self._best_offsets
        goals_raw = np.append(arm_goals_raw, goals[-1]) if self._gripper else arm_goals_raw
        return [self._driver._rad_to_pulses(rad) for rad in goals_raw]

    def _gripper_readout_to_percent(self, gripper_position: float) -> float:
        """Convert gripper position to percentage (0-1)."""
        gripper_position_percent = (gripper_position - self._gripper_range_rad[0]) / (
            self._gripper_range_rad[1] - self._gripper_range_rad[0]
        )
        return max(0.0, min(1.0, gripper_position_percent))
