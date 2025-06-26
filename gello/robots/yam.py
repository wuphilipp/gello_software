from dm_control import mjcf
from pathlib import Path
import numpy as np
from typing import Dict
from gello.robots.robot import Robot
from gello.dm_control_tasks.mjcf_utils import MENAGERIE_ROOT

class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    def __init__(self, no_gripper: bool = False):
        self._use_gripper = not no_gripper

        # Load MJCF model and filter joint names
        mjcf_model = mjcf.from_path(str(MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"))
        all_joints = [j.name for j in mjcf_model.find_all("joint")]

        if self._use_gripper:
            # Use all joints except right_finger (passive)
            self._joint_names = [j for j in all_joints if j != "right_finger"]
        else:
            # Use only joint1–6
            self._joint_names = [j for j in all_joints if j.startswith("joint")]

        self._joint_state = np.zeros(len(self._joint_names))
        self._joint_velocities = np.zeros(len(self._joint_names))
        self._gripper_state = 0.0

    def num_dofs(self) -> int:
        return len(self._joint_names)

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