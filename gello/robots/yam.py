from typing import Dict

import numpy as np

from gello.robots.robot import Robot


class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    def __init__(self, channel="can0"):
        from i2rt.robots.get_robot import get_yam_robot

        self.robot = get_yam_robot(channel=channel)

        # YAM has 7 joints (6 arm joints + 1 gripper)
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = np.zeros(7)  # 7 joints
        self._joint_velocities = np.zeros(7)  # 7 joints
        self._gripper_state = 0.0

    def num_dofs(self) -> int:
        return 7  # YAM has 7 DOFs

    def get_joint_state(self) -> np.ndarray:
        # Get actual joint positions from I2RT robot (7 joints total)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we have exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")

        self._joint_state = joint_pos
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        dt = 0.01
        self._joint_velocities = (joint_state - self._joint_state) / dt
        self._joint_state = joint_state

        # Command the I2RT robot with all 7 joints (6 arm + 1 gripper)
        self.command_joint_pos(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }

    def get_joint_pos(self):
        # Get 7 joints from I2RT robot (6 arm + 1 gripper)
        joint_pos = self.robot.get_joint_pos()
        # Ensure we return exactly 7 joints
        if len(joint_pos) > 7:
            joint_pos = joint_pos[:7]
        elif len(joint_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            joint_pos = np.pad(joint_pos, (0, 7 - len(joint_pos)), "constant")
        return joint_pos

    def command_joint_pos(self, target_pos):
        # Ensure we send exactly 7 joints to the I2RT robot
        if len(target_pos) > 7:
            target_pos = target_pos[:7]
        elif len(target_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            target_pos = np.pad(target_pos, (0, 7 - len(target_pos)), "constant")
        self.robot.command_joint_pos(np.array(target_pos))


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
