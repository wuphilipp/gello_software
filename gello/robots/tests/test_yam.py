import sys
import types

import numpy as np
import pytest

from gello.robots.yam import YAMRobot


class FakeI2RTRobot:
    """Stands in for i2rt's robot: it reports where the arm is, not what it was told."""

    def __init__(self, start):
        self.measured = np.array(start, dtype=float)
        self.commands = []

    def get_joint_pos(self):
        return self.measured.copy()

    def command_joint_pos(self, pos):
        self.commands.append(np.array(pos, dtype=float))


@pytest.fixture
def make_yam(monkeypatch):
    def make(start):
        fake = FakeI2RTRobot(start)
        module = types.ModuleType("i2rt.robots.get_robot")
        module.get_yam_robot = lambda channel="can0": fake
        for name in ("i2rt", "i2rt.robots"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", module)
        return YAMRobot(channel="can0"), fake

    return make


def test_observations_report_the_arm_before_any_command(make_yam):
    start = [1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.8]
    robot, _ = make_yam(start)
    obs = robot.get_observations()
    assert np.allclose(obs["joint_positions"], start)
    assert np.allclose(obs["gripper_position"], [0.8])


def test_observations_report_the_arm_not_the_command(make_yam):
    robot, fake = make_yam(np.zeros(7))
    command = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.command_joint_state(command)
    assert np.allclose(fake.commands[-1], command)
    # The arm sags under load and an object stops the gripper part way.
    fake.measured = np.array([0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.38])
    obs = robot.get_observations()
    assert np.allclose(obs["joint_positions"], fake.measured)
    assert np.allclose(obs["gripper_position"], [0.38])
