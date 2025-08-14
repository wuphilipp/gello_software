from dataclasses import dataclass
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from typing import Any, Iterator


@dataclass
class ParameterConfig:
    descriptor: ParameterDescriptor
    default: Any


class GelloParameterConfig:
    """Configuration class for GELLO ROS2 parameters."""

    DEFAULT_NUM_JOINTS = 7
    DEFAULT_JOINT_SIGNS = [1] * DEFAULT_NUM_JOINTS
    DEFAULT_BEST_OFFSETS = [0.0] * DEFAULT_NUM_JOINTS
    DEFAULT_GRIPPER_RANGE_RAD = (0.0, 0.0)
    DEFAULT_CONTROL_GAINS = [0] * DEFAULT_NUM_JOINTS
    DEFAULT_GOAL_POSITION = [0.0] * DEFAULT_NUM_JOINTS

    def __init__(self, default_com_port: str):
        self.hardware_params = [
            ParameterConfig(
                ParameterDescriptor(
                    name="com_port",
                    type=ParameterType.PARAMETER_STRING,
                    description="USB serial port path",
                    read_only=True,
                ),
                default_com_port,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="gello_name",
                    type=ParameterType.PARAMETER_STRING,
                    description="GELLO device identifier",
                    read_only=True,
                ),
                default_com_port,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="num_arm_joints",
                    type=ParameterType.PARAMETER_INTEGER,
                    description="Number of arm joints",
                    read_only=True,
                ),
                self.DEFAULT_NUM_JOINTS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="joint_signs",
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    description="Joint direction signs",
                    read_only=True,
                ),
                self.DEFAULT_JOINT_SIGNS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="gripper",
                    type=ParameterType.PARAMETER_BOOL,
                    description="Enable gripper control",
                    read_only=True,
                ),
                True,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="gripper_range_rad",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description="Gripper range in radians",
                    read_only=True,
                ),
                self.DEFAULT_GRIPPER_RANGE_RAD,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="best_offsets",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description="Joint offset calibration",
                    read_only=True,
                ),
                self.DEFAULT_BEST_OFFSETS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="dynamixel_kp_p",
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    description="Proportional gains",
                    additional_constraints="0 to ~1000",
                    read_only=False,
                ),
                self.DEFAULT_CONTROL_GAINS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="dynamixel_kp_i",
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    description="Integral gains",
                    additional_constraints="0 to ~1000",
                    read_only=False,
                ),
                self.DEFAULT_CONTROL_GAINS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="dynamixel_kp_d",
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    description="Derivative gains",
                    additional_constraints="0 to ~1000",
                    read_only=False,
                ),
                self.DEFAULT_CONTROL_GAINS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="dynamixel_torque_enable",
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    description="GELLO Torque enabled",
                    additional_constraints="0 (disabled), 1 (enabled)",
                    read_only=False,
                ),
                [0] * self.DEFAULT_NUM_JOINTS,
            ),
            ParameterConfig(
                ParameterDescriptor(
                    name="dynamixel_goal_position",
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    description="Goal positions",
                    read_only=False,
                ),
                self.DEFAULT_GOAL_POSITION,
            ),
        ]

    def __iter__(self) -> Iterator[ParameterConfig]:
        """Return an iterator over the parameter configurations."""
        return iter(self.hardware_params)
