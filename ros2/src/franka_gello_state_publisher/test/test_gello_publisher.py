from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from rclpy.clock import Clock

from franka_gello_state_publisher.gello_publisher import GelloPublisher


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
