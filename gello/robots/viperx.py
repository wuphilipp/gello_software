import numpy as np
from pyquaternion import Quaternion

from gello.robots.robot import Robot
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

GRIPPER_POSITION_OPEN = 0.05800
GRIPPER_POSITION_CLOSE = 0.01844

GRIPPER_JOINT_OPEN = 1.4910
GRIPPER_JOINT_CLOSE = -0.6213

class ViperXRobot(Robot):
    def __init__(self, robot_ip):
        print('ViperXRobot __init__')
        super().__init__()
        self.bot = InterbotixManipulatorXS(robot_model='vx300s', group_name='arm', gripper_name='gripper')
        self._gripper_cmd = JointSingleCommand(name="gripper")

        self.bot.core.robot_reboot_motors("single", "gripper", True)
        self.bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")

        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Velocity', 100)
        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Acceleration', 0)

    def stop(self):
        self.bot.core.robot_set_operating_modes("single", "gripper", "pwm")

        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Velocity', 2000)
        self.bot.core.robot_set_motor_registers("group", "arm", 'Profile_Acceleration', 300)

    def num_dofs(self) -> int:
        return 7

    def get_joint_state(self) -> np.ndarray:
        state = np.concatenate([self.bot.arm.get_joint_commands(), 0])
        print(f'get_joint_state: {state}')
        return state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == (self.num_dofs()), (
            f"Expected joint state of length {self.num_dofs()}, "
            f"got {len(joint_state)}."
        )

        self.bot.arm.set_joint_positions(joint_state[:6], blocking=False)

        gripper_angle = ((1 - joint_state[6]) * (GRIPPER_JOINT_OPEN - GRIPPER_JOINT_CLOSE) + GRIPPER_JOINT_CLOSE)
        self._gripper_cmd.cmd = gripper_angle
        self.bot.gripper.core.pub_single.publish(self._gripper_cmd)

    def get_observations(self):
        gripper_angle = self.bot.core.joint_states.position[-2]
        gripper_pos = 1 - ((gripper_angle - GRIPPER_POSITION_CLOSE) / (GRIPPER_POSITION_OPEN - GRIPPER_POSITION_CLOSE))

        joints = np.concatenate([self.bot.arm.get_joint_commands(), [gripper_pos]])
        ee_pos_matrix = self.bot.arm.get_ee_pose_command()
        ee_pos = np.array([ee_pos_matrix[0][3], ee_pos_matrix[1][3], ee_pos_matrix[2][3]])
        ee_quat = Quaternion(matrix=ee_pos_matrix[:3, :3])

        obs =  {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat.elements]),
            "gripper_position": np.array([gripper_pos]),
        }
        return obs
