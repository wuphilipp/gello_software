from typing import Dict
import numpy as np
from gello.robots.robot import Robot

class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    def __init__(self, no_gripper: bool = False):
        self._use_gripper = not no_gripper
        self._joint_state = np.zeros(self.num_dofs())
        self._joint_velocities = np.zeros(self.num_dofs())
        self._gripper_state = 0.0

    def num_dofs(self) -> int:
        return 7 if self._use_gripper else 6

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs(), (
            f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"
        )
        dt = 0.01
        self._joint_velocities = (joint_state - self._joint_state) / dt
        self._joint_state = joint_state
        if self._use_gripper:
            self._gripper_state = joint_state[-1]

    def get_observations(self) -> Dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }


def main():
    robot = YAMRobot()
    print(robot.get_observations())

if __name__ == "__main__":
    main()