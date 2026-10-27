from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import rclpy
from rclpy.clock import Clock
from rclpy.parameter import Parameter

from franka_gello_state_publisher.gello_publisher import GelloPublisher

ARM_PARAMETERS = {
    "num_arm_joints": "7",
    "joint_signs": "[1, -1, 1, -1, 1, 1, 1]",
    "assembly_offsets": "[0.0, 0.0, 3.142, 3.142, 3.142, 4.712, 0.0]",
    "gripper_range_rad": "[2.0, 3.2]",
    "dynamixel_kp_p": "[0, 0, 0, 0, 0, 0, 0, 0]",
    "dynamixel_kp_i": "[0, 0, 0, 0, 0, 0, 0, 0]",
    "dynamixel_kp_d": "[0, 0, 0, 0, 0, 0, 0, 0]",
    "dynamixel_torque_enable": "[0, 0, 0, 0, 0, 0, 0, 0]",
    "dynamixel_goal_position": "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
}
GRIPPER_ONLY_PARAMETERS = {
    "num_arm_joints": "0",
    "gripper_range_rad": "[2.0, 3.2]",
    "dynamixel_kp_p": "[0]",
    "dynamixel_kp_i": "[0]",
    "dynamixel_kp_d": "[0]",
    "dynamixel_torque_enable": "[0]",
    "dynamixel_goal_position": "[0.0]",
}


@contextmanager
def running_publisher(parameters):
    args = ["--ros-args"]
    for name, value in parameters.items():
        args += ["-p", f"{name}:={value}"]
    rclpy.init(args=args)
    try:
        publisher = GelloPublisher()
        try:
            yield publisher
        finally:
            publisher.destroy_node()
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize(
    "parameters", [ARM_PARAMETERS, GRIPPER_ONLY_PARAMETERS], ids=["arm", "gripper_only"]
)
def test_publisher_starts_with_configured_parameters(fake_driver, parameters):
    with running_publisher(parameters) as publisher:
        num_arm_joints = int(parameters["num_arm_joints"])
        assert len(publisher.gello_hardware._joint_signs) == num_arm_joints
        assert len(publisher.gello_hardware._assembly_offsets) == num_arm_joints


@pytest.mark.parametrize("num_arm_joints", [0, 7])
def test_publisher_publishes_names_for_configured_joints_only(num_arm_joints):
    publisher = SimpleNamespace(
        gello_hardware=mock.Mock(),
        arm_joint_publisher=mock.Mock(),
        gripper_joint_publisher=mock.Mock(),
        get_clock=Clock,
    )
    publisher.gello_hardware.get_joint_and_gripper_positions.return_value = (
        np.zeros(num_arm_joints),
        0.5,
    )

    GelloPublisher.publish_joint_jog(publisher)

    gripper_msg = publisher.gripper_joint_publisher.publish.call_args.args[0]
    assert gripper_msg.data == 0.5
    if num_arm_joints == 0:
        publisher.arm_joint_publisher.publish.assert_not_called()
    else:
        joint_states = publisher.arm_joint_publisher.publish.call_args.args[0]
        assert list(joint_states.name) == [f"fr3_joint{i}" for i in range(1, num_arm_joints + 1)]
        assert len(joint_states.position) == num_arm_joints


def make_publisher(num_arm_joints=7, gripper=True):
    parameters = {"num_arm_joints": num_arm_joints, "gripper": gripper}
    return SimpleNamespace(get_parameter=lambda name: SimpleNamespace(value=parameters[name]))


def test_runtime_update_with_correct_length_is_accepted():
    result = GelloPublisher.validate_parameters_callback(
        make_publisher(), [Parameter("dynamixel_kp_p", value=[0] * 8)]
    )

    assert result.successful


def test_runtime_update_with_wrong_length_is_rejected():
    result = GelloPublisher.validate_parameters_callback(
        make_publisher(), [Parameter("dynamixel_kp_p", value=[0] * 7)]
    )

    assert not result.successful
    assert "dynamixel_kp_p must have 8 entries" in result.reason


def test_runtime_update_of_other_parameters_is_not_checked():
    result = GelloPublisher.validate_parameters_callback(
        make_publisher(), [Parameter("publishing_rate", value=50)]
    )

    assert result.successful
