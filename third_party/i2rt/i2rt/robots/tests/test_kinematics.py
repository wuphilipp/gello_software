import numpy as np
import pytest

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.motor_chain_robot import YAM_XML_PATH


@pytest.fixture
def kinematics_yam() -> Kinematics:
    return Kinematics(YAM_XML_PATH, "grasp_site")


def test_fk(kinematics_yam: Kinematics) -> None:
    q = np.zeros(6)
    pose = kinematics_yam.fk(q)
    assert pose.shape == (4, 4), "FK should return a 4x4 matrix"

    # Add more assertions based on expected pose values
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    start_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    start_trans = np.array([0.245, 0.0, 0.164])
    np.testing.assert_allclose(rotation, start_rot, atol=1e-5)
    np.testing.assert_allclose(translation, start_trans, atol=1e-5)


def test_ik_smoke(kinematics_yam: Kinematics) -> None:
    q = np.zeros(6)
    pose = kinematics_yam.fk(q)
    pose[0, 3] -= 0.1
    pose[2, 3] += 0.1
    success, q_ik = kinematics_yam.ik(pose, "grasp_site")
    assert success, "IK should succeed"
    assert q_ik.shape == (6,), "IK should return a joint configuration of size 6"


def test_cycle(kinematics_yam: Kinematics) -> None:
    q = np.zeros(6)
    pose = kinematics_yam.fk(q)
    success, q_ik = kinematics_yam.ik(pose, "grasp_site")
    assert success, "IK should succeed"
    pose_reconstructed = kinematics_yam.fk(q_ik)
    np.testing.assert_allclose(pose, pose_reconstructed, atol=1e-5)
