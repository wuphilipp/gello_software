import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import rclpy
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterEvent, ParameterType, ParameterValue
from rclpy.clock import Clock
from rclpy.exceptions import ParameterNotDeclaredException
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter

from franka_gello_state_publisher.gello_publisher import GelloPublisher

NUM_MOTORS = 8
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


@pytest.fixture
def publisher(fake_driver):
    with running_publisher(ARM_PARAMETERS) as node:
        driver = node.gello_hardware._driver
        driver.write_value_by_name = mock.Mock(wraps=driver.write_value_by_name)
        yield node


def driver_writes(publisher):
    return publisher.gello_hardware._driver.write_value_by_name


def test_valid_update_is_written_to_driver(publisher):
    [result] = publisher.set_parameters([Parameter("dynamixel_kp_p", value=[5] * NUM_MOTORS)])

    assert result.successful
    driver_writes(publisher).assert_called_once_with("kp_p", [5] * NUM_MOTORS)
    assert list(publisher.get_parameter("dynamixel_kp_p").value) == [5] * NUM_MOTORS


def test_torque_enable_update_rewrites_goal_position(publisher):
    goal_position = publisher.gello_hardware._dynamixel_control_config["goal_position"]

    [result] = publisher.set_parameters(
        [Parameter("dynamixel_torque_enable", value=[1] * NUM_MOTORS)]
    )

    assert result.successful
    assert driver_writes(publisher).call_args_list == [
        mock.call("torque_enable", [1] * NUM_MOTORS),
        mock.call("goal_position", goal_position),
    ]


def test_update_with_wrong_length_is_not_written_to_driver(publisher):
    [result] = publisher.set_parameters(
        [Parameter("dynamixel_kp_p", value=[5] * (NUM_MOTORS - 1))]
    )

    assert not result.successful
    driver_writes(publisher).assert_not_called()
    assert list(publisher.get_parameter("dynamixel_kp_p").value) == [0] * NUM_MOTORS


def test_failed_driver_write_rejects_update(publisher):
    driver_writes(publisher).side_effect = RuntimeError("communication error")

    [result] = publisher.set_parameters([Parameter("dynamixel_kp_p", value=[5] * NUM_MOTORS)])
    driver_writes(publisher).side_effect = None

    assert not result.successful
    assert "communication error" in result.reason
    assert list(publisher.get_parameter("dynamixel_kp_p").value) == [0] * NUM_MOTORS


def test_undeclared_dynamixel_parameter_cannot_be_set(publisher):
    with pytest.raises(ParameterNotDeclaredException):
        publisher.set_parameters([Parameter("dynamixel_goal_current", value=[2000] * NUM_MOTORS)])

    result = publisher.set_parameters_callback(
        [Parameter("dynamixel_goal_current", value=[2000] * NUM_MOTORS)]
    )

    assert not result.successful
    driver_writes(publisher).assert_not_called()


def test_parameter_event_is_not_written_to_driver(publisher):
    sender = rclpy.create_node("parameter_event_sender")
    event = ParameterEvent(
        node=publisher.get_fully_qualified_name(),
        changed_parameters=[
            ParameterMsg(
                name=name,
                value=ParameterValue(
                    type=ParameterType.PARAMETER_INTEGER_ARRAY,
                    integer_array_value=[1] * NUM_MOTORS,
                ),
            )
            for name in ("dynamixel_torque_enable", "dynamixel_goal_current")
        ],
    )
    received = []
    sender.create_subscription(
        ParameterEvent,
        "/parameter_events",
        lambda msg: received.append(msg) if msg.node == event.node else None,
        10,
    )
    event_publisher = sender.create_publisher(ParameterEvent, "/parameter_events", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(sender)

    deadline = time.monotonic() + 5.0
    while not received and time.monotonic() < deadline:
        event_publisher.publish(event)
        executor.spin_once(timeout_sec=0.1)
    for _ in range(5):
        executor.spin_once(timeout_sec=0.05)

    executor.shutdown()
    sender.destroy_node()
    assert received, "parameter event was not delivered, test is inconclusive"
    driver_writes(publisher).assert_not_called()
