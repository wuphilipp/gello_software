import os
from dataclasses import dataclass
from typing import Tuple
from textwrap import indent

import numpy as np
import tyro

from franka_gello_state_publisher.dynamixel.driver import DynamixelDriver
from franka_gello_state_publisher.gello_hardware import GelloHardware

# The offset in radians from the gripper open position to the closed position.
GRIPPER_OPEN_TO_CLOSED_RAD = -1.22


@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = (0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0)
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[int, ...] = (1, -1, 1, -1, 1, 1, 1)
    """Sign multipliers for each joint to account for motor direction."""

    gripper: bool = True
    """Whether or not the gripper is attached."""

    baudrate: int = 57600
    """Host serial baudrate matching the Dynamixel chain (57600 factory, 115200 after provision)."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert j == -1 or j == 1, f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_arm_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_total_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_arm_joints + extra_joints


def determine_offsets(
    arm_joints_raw: np.ndarray, start_joints: np.ndarray, joint_signs: np.ndarray
) -> np.ndarray:
    """
    Calculate assembly offsets by comparing current pose to expected initialization pose and rounding to nearest 90 degrees.

    Parameters
    ----------
    arm_joints_raw : np.ndarray
        Raw joint positions read from the driver
    start_joints : np.ndarray
        Expected joint positions in radians for the initialization pose
    joint_signs : np.ndarray
        Signs to apply to each joint angle to account for motor direction

    Returns
    -------
    np.ndarray
        Assembly offsets in radians, normalized to positive values.

    """
    arm_joints_normalized = GelloHardware.normalize_joint_positions(
        arm_joints_raw,
        np.zeros(len(arm_joints_raw)),  # zero offsets since we are calculating them here
        joint_signs,
    )
    pose_differences = arm_joints_normalized - start_joints
    offsets = np.round(pose_differences / (np.pi / 2)) * (np.pi / 2)
    offsets_normalized = np.mod(offsets, 2 * np.pi)
    return offsets_normalized


def main(args: Args) -> None:
    joint_ids = list(range(1, args.num_total_joints + 1))
    driver = DynamixelDriver(joint_ids, port=args.port, baudrate=args.baudrate)
    joints_raw = driver.get_joints()
    arm_joints_raw = np.array(joints_raw[: args.num_arm_joints])
    assembly_offsets = determine_offsets(
        arm_joints_raw, np.array(args.start_joints), np.array(args.joint_signs)
    )

    gripper_range_rad = None
    if args.gripper and len(joints_raw) > args.num_arm_joints:
        gripper_open = joints_raw[-1]
        gripper_range_rad = [gripper_open + GRIPPER_OPEN_TO_CLOSED_RAD, gripper_open]

    print("Update your config files with the following values:\n")
    print(indent(f'com_port: "{os.path.basename(args.port)}"', "  "))
    print(indent(f"num_arm_joints: {args.num_arm_joints}", "  "))
    print(indent(f"joint_signs: {list(args.joint_signs)}", "  "))
    print(indent(f"gripper: {str(args.gripper).lower()}", "  "))
    print(indent(f"assembly_offsets: {list(np.round(assembly_offsets, 3))} # rad", "  "))
    if args.gripper:
        print(indent(f"gripper_range_rad: {list(np.round(gripper_range_rad, 3))}", "  "))
    print()


if __name__ == "__main__":
    main(tyro.cli(Args))
