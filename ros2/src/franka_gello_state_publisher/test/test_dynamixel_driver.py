from threading import Lock
from unittest import mock

import pytest
from dynamixel_sdk import COMM_SUCCESS, COMM_TX_FAIL, GroupSyncWrite

from franka_gello_state_publisher.dynamixel.driver import DynamixelDriver


def test_write_succeeds_after_failed_write():
    driver = DynamixelDriver.__new__(DynamixelDriver)
    driver._ids = [1, 2]
    driver._lock = Lock()
    driver._packetHandler = mock.Mock()
    driver._packetHandler.syncWriteTxOnly.side_effect = [COMM_TX_FAIL, COMM_SUCCESS]
    group_sync_write = GroupSyncWrite(mock.Mock(), driver._packetHandler, 84, 2)
    driver._groupSyncWriteHandlers = {"kp_p": group_sync_write}
    driver._ctrl_table = {"kp_p": {"addr": 84, "len": 2}}

    with pytest.raises(RuntimeError, match="Failed to syncwrite kp_p"):
        driver.write_value_by_name("kp_p", [5, 5])
    driver.write_value_by_name("kp_p", [5, 5])

    assert driver._packetHandler.syncWriteTxOnly.call_count == 2
