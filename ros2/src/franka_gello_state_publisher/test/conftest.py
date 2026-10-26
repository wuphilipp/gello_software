import pytest

from franka_gello_state_publisher import gello_hardware
from franka_gello_state_publisher.dynamixel.driver import FakeDynamixelDriver


@pytest.fixture
def fake_driver(monkeypatch):
    """Replace the Dynamixel driver of GelloHardware with the fake driver."""
    monkeypatch.setattr(
        gello_hardware,
        "DynamixelDriver",
        lambda ids, port, baudrate: FakeDynamixelDriver(list(ids)),
    )
    monkeypatch.setattr(gello_hardware.time, "sleep", lambda _: None)
