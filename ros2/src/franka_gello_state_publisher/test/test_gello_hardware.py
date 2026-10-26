from unittest import mock

import numpy as np
import pytest

from franka_gello_state_publisher.gello_hardware import GelloHardware


def make_config(num_arm_joints=7, gripper=True, **overrides):
    num_motors = num_arm_joints + (1 if gripper else 0)
    config = {
        "com_port": "/dev/null",
        "gello_name": "test",
        "num_arm_joints": num_arm_joints,
        "joint_signs": [1] * num_arm_joints,
        "gripper": gripper,
        "gripper_range_rad": [2.0, 3.2],
        "assembly_offsets": [0.0] * num_arm_joints,
        "baudrate": 57600,
        "dynamixel_kp_p": [0] * num_motors,
        "dynamixel_kp_i": [0] * num_motors,
        "dynamixel_kp_d": [0] * num_motors,
        "dynamixel_torque_enable": [0] * num_motors,
        "dynamixel_goal_position": [0.0] * num_motors,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize("num_arm_joints, gripper", [(0, True), (7, True), (7, False)])
def test_hardware_reads_configured_number_of_joints(fake_driver, num_arm_joints, gripper):
    hardware = GelloHardware(make_config(num_arm_joints, gripper), mock.Mock())

    arm_joints, gripper_position = hardware.get_joint_and_gripper_positions()

    assert hardware._driver._ids == list(range(1, num_arm_joints + (2 if gripper else 1)))
    assert arm_joints.shape == (num_arm_joints,)
    limits = GelloHardware.JOINT_POSITION_LIMITS[:num_arm_joints]
    assert np.all(arm_joints >= limits[:, 0])
    assert np.all(arm_joints <= limits[:, 1])
    assert 0.0 <= gripper_position <= 1.0
