from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from franka_gello_state_publisher.gello_hardware import GelloHardware

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


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


@pytest.mark.parametrize("num_arm_joints, gripper", [(7, True), (7, False), (0, True)])
def test_valid_config_passes(num_arm_joints, gripper):
    GelloHardware.validate_parameters(make_config(num_arm_joints, gripper))


@pytest.mark.parametrize("num_arm_joints", [-1, 3, 8])
def test_unsupported_num_arm_joints_fails(num_arm_joints):
    with pytest.raises(ValueError, match=r"num_arm_joints must be 0 \(gripper only\) or 7"):
        GelloHardware.validate_parameters(make_config(num_arm_joints))


def test_no_motors_fails():
    with pytest.raises(ValueError, match="gripper must be true if num_arm_joints is 0"):
        GelloHardware.validate_parameters(make_config(num_arm_joints=0, gripper=False))


@pytest.mark.parametrize("param_name", GelloHardware.ARM_JOINT_PARAMS)
@pytest.mark.parametrize(
    "num_arm_joints, num_values",
    [(7, 8), (7, 6), (7, 1), (0, 1)],
    ids=["too_long", "too_short", "single_entry", "not_empty_without_arm"],
)
def test_arm_joint_param_with_wrong_length_fails(param_name, num_arm_joints, num_values):
    config = make_config(num_arm_joints, **{param_name: [1] * num_values})

    with pytest.raises(ValueError, match=f"{param_name} must have {num_arm_joints} entries"):
        GelloHardware.validate_parameters(config)


@pytest.mark.parametrize("param_name", GelloHardware.DYNAMIXEL_CONTROL_PARAMS)
@pytest.mark.parametrize(
    "gripper, num_values, expected",
    [(True, 7, 8), (False, 8, 7)],
    ids=["gripper_entry_missing", "extra_entry_without_gripper"],
)
def test_dynamixel_param_with_wrong_length_fails(param_name, gripper, num_values, expected):
    config = make_config(gripper=gripper, **{param_name: [0] * num_values})

    with pytest.raises(ValueError, match=f"{param_name} must have {expected} entries"):
        GelloHardware.validate_parameters(config)


@pytest.mark.parametrize(
    "gripper_range_rad",
    [[2.0], [2.0, 3.0, 4.0], [2.0, 2.0]],
    ids=["too_short", "too_long", "equal_values"],
)
def test_invalid_gripper_range_fails_with_gripper(gripper_range_rad):
    config = make_config(gripper=True, gripper_range_rad=gripper_range_rad)

    with pytest.raises(ValueError, match="gripper_range_rad must have 2 different entries"):
        GelloHardware.validate_parameters(config)


def test_invalid_gripper_range_is_ignored_without_gripper():
    GelloHardware.validate_parameters(make_config(gripper=False, gripper_range_rad=[0.0, 0.0]))


@pytest.mark.parametrize("config_file", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_pass(config_file):
    for gello_config in yaml.safe_load(config_file.read_text()).values():
        GelloHardware.validate_parameters(
            {"joint_signs": [], "assembly_offsets": [], **gello_config}
        )
