import time
from typing import Dict

import numpy as np

from gello.robots.robot import Robot

# MAX_OPEN = 0.09

class PandaFrankyRobot(Robot):
    """A class representing a Panda robot."""
    def __init__(self, robot_ip: str = "172.16.0.2"):
        from franky import Robot as Robot_Franky
        from franky import Gripper, JointMotion, RelativeDynamicsFactor

        self.robot = Robot_Franky(
            robot_ip,
        )
        self.robot.recover_from_errors()
        self.gripper = Gripper(
            robot_ip
        )
        joints = [-0.0, -0.0, 0.0, -1.57, 0.0, 1.57, 0.785037]
        reset_joints = JointMotion(joints)
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=0.1, acceleration=0.05, jerk=0.05
        )
        self.gripper.open(speed=0.1)
        self.robot.move(reset_joints)
        self.gripper_state = {
            "current_state": 0, 
            "target_state": 0,
        }
        time.sleep(1)

def num_dofs(self) -> int:
    """Get the number of joints of the robot.

    Returns:
    int: The number of joints of the robot.
    """
    return 8

def get_joint_state(self) -> np.ndarray:
    """Get the current state of the leader robot.

    Returns:
    T: The current state of the leader robot.
    """
    robot_joints = self.robot.current_joint_state.position
    gripper_pos = 0 # franky has a blocking issue
    pos = np.append(robot_joints, gripper_pos / MAX_OPEN)
    return pos

def command_joint_state(self, joint_state: np.ndarray) -> None:
    """Command the leader robot to a given state.

    Args:
    joint_state (np.ndarray): The state to command the leader robot to.
    """
    from franky import JointWaypoint, JointWaypointMotion
    self.robot.move(JointWaypointMotion([JointWaypoint(joint_state[:-1])]), asynchronous=True)
    speed = 0.10 # [m/s]
    force = 20.0 # [N]
    if joint_state[-1] < 0.5:
        self.gripper_state['target_state'] = 0
    else:
        self.gripper_state['target_state'] = 1
    if self.gripper_state['current_state'] != self.gripper_state['target_state']:
        if self.gripper_state['target_state'] == 0:
            self.gripper.open_async(speed=speed)
        else:
            self.gripper.grasp_async(0.0, speed, force, epsilon_outer=1.0)
        self.gripper_state['current_state'] = self.gripper_state['target_state']

def get_observations(self) -> Dict[str, np.ndarray]:
    joints = self.get_joint_state()
    pos_quat = np.zeros(7)
    gripper_pos = np.array([joints[-1]])
    return {
    "joint_positions": joints,
    "joint_velocities": joints,
    "ee_pos_quat": pos_quat,
    "gripper_position": gripper_pos,
    }


def main():
    robot = PandaFrankyRobot()
    current_joints = robot.get_joint_state()
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.1
    time.sleep(1)
    m = 0.09
    robot.command_joint_state(move_joints)
    time.sleep(1)

if __name__ == "__main__":
    main()

