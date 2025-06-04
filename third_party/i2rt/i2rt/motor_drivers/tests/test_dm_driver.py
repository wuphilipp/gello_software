from unittest.mock import Mock, patch

import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import (
    DMChainCanInterface,
    DMSingleMotorCanInterface,
    FeedbackFrameInfo,
    MotorInfo,
    ReceiveMode,
)


@pytest.fixture
def mock_single_motor_interface() -> DMSingleMotorCanInterface:
    with patch("i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface", autospec=True) as MockInterface:
        mock_interface = MockInterface.return_value
        # Setup default responses
        mock_interface.motor_on = Mock()
        mock_interface.clean_error = Mock()
        mock_interface.set_control = Mock()
        mock_interface._send_message_get_response = Mock()
        yield mock_interface


@pytest.fixture
def dm_chain_interface(mock_single_motor_interface: DMSingleMotorCanInterface) -> DMChainCanInterface:
    """Create a DMChainCanInterface with mocked components."""
    motor_list = [[0x01, "DM4340"], [0x02, "DM4310"]]
    motor_offsets = [0, 0]
    motor_directions = [1, 1]

    with patch("i2rt.motor_drivers.dm_driver.DMSingleMotorCanInterface", return_value=mock_single_motor_interface):
        interface = DMChainCanInterface(
            motor_list=motor_list,
            motor_offset=motor_offsets,
            motor_direction=motor_directions,
            channel="test_channel",
            start_thread=False,
        )
        # Initialize state
        interface.state = [
            FeedbackFrameInfo(
                id=1,
                error_code=0,
                error_message="",
                position=0.0,
                velocity=0.0,
                torque=0.0,
                temperature_mos=30,
                temperature_rotor=35,
            ),
            FeedbackFrameInfo(
                id=2,
                error_code=0,
                error_message="",
                position=0.0,
                velocity=0.0,
                torque=0.0,
                temperature_mos=30,
                temperature_rotor=35,
            ),
        ]
        interface.absolute_positions = np.zeros(len(motor_list))
        yield interface


def test_dm_chain_initialization(dm_chain_interface: DMChainCanInterface) -> None:
    """Test DMChainCanInterface initialization."""
    assert len(dm_chain_interface.motor_list) == 2
    assert dm_chain_interface.motor_list == [[0x01, "DM4340"], [0x02, "DM4310"]]
    assert np.array_equal(dm_chain_interface.motor_offset, np.array([0, 0]))
    assert np.array_equal(dm_chain_interface.motor_direction, np.array([1, 1]))


def test_set_commands(
    dm_chain_interface: DMChainCanInterface, mock_single_motor_interface: DMSingleMotorCanInterface
) -> None:
    """Test setting commands to motors."""
    # Prepare test data
    torques = np.array([0.5, -0.5])
    pos = np.array([1.0, 2.0])
    vel = np.array([0.1, 0.2])
    kp = np.array([10.0, 20.0])
    kd = np.array([1.0, 2.0])

    # Mock feedback response
    mock_feedback = FeedbackFrameInfo(
        id=1,
        error_code=0,
        error_message="",
        position=1.0,
        velocity=0.1,
        torque=0.5,
        temperature_mos=30,
        temperature_rotor=35,
    )

    mock_single_motor_interface.set_control.return_value = mock_feedback

    # Test command setting
    result = dm_chain_interface.set_commands(torques, pos, vel, kp, kd)

    assert len(result) == 2
    assert isinstance(result[0], MotorInfo)
    assert result[0].target_torque == pytest.approx(0.5)


def test_receive_mode() -> None:
    """Test ReceiveMode functionality."""
    mode = ReceiveMode.p16
    assert mode.get_receive_id(1) == 17
    assert mode.to_motor_id(17) == 1

    mode = ReceiveMode.same
    assert mode.get_receive_id(1) == 1
    assert mode.to_motor_id(1) == 1


def test_thread_safety(dm_chain_interface: DMChainCanInterface) -> None:
    """Test thread safety mechanisms."""
    assert hasattr(dm_chain_interface, "command_lock")
    assert hasattr(dm_chain_interface, "state_lock")

    with dm_chain_interface.command_lock:
        assert True  # Lock acquired successfully

    with dm_chain_interface.state_lock:
        assert True  # Lock acquired successfully
