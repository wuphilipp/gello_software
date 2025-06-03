import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
)

# Constants for XL330-M288-T and XL330-M077-T Dynamixel servos
CTRL_TABLE = {
    "torque_enable": {"addr": 64, "len": 1, "min": 0, "max": 1},
    "goal_position": {"addr": 116, "len": 4},
    "present_position": {"addr": 132, "len": 4, "read_only": True},
    "goal_current": {"addr": 102, "len": 2, "min": -1750, "max": 1750},
    "kp_d": {"addr": 80, "len": 2, "min": 0, "max": 16383},
    "kp_i": {"addr": 82, "len": 2, "min": -1750, "max": 1750},
    "kp_p": {"addr": 84, "len": 2, "min": -1750, "max": 1750},
    "operating_mode": {"addr": 11, "len": 1, "min": 0, "max": 6},
}
OPERATING_MODES = {
    0: "Current Control Mode",
    1: "Velocity Control Mode",
    3: "Position Control Mode (Default)",
    4: "Extended Position Control Mode",
    5: "Current-based Position Control Mode",
    6: "PWM Control Mode",
}
PULSES_PER_REVOLUTION = 4095


def pulses_to_rad(pulses) -> float:
    return pulses / PULSES_PER_REVOLUTION * 2 * np.pi


def rad_to_pulses(rad: float) -> int:
    return int(rad / (2 * np.pi) * PULSES_PER_REVOLUTION)


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float | None]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float | None]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def set_by_name(self, name: str, values: Sequence[int | None]):
        """Set a group of values by name.

        Args:
            name (str): The name of the control parameter.
            values (Sequence[int | None]): A list of values to set for each servo.
        """
        ...

    def read_by_name(self, name: str) -> Sequence[int]:
        """Read a group of values by name.

        Args:
            name (str): The name of the control parameter to read.

        Returns:
            Sequence[int]: A list of values read from the servos.
        """
        ...

    def close(self):
        """Close the driver."""
        ...


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._storage_map = {
            "goal_position": np.zeros(len(ids), dtype=int),
            "goal_current": np.zeros(len(ids), dtype=int),
            "kp_d": np.zeros(len(ids), dtype=int),
            "kp_i": np.zeros(len(ids), dtype=int),
            "kp_p": np.zeros(len(ids), dtype=int),
            "operating_mode": np.zeros(len(ids), dtype=int),
            "torque_enable": np.zeros(len(ids), dtype=int),
        }

    def _set_group(
        self,
        name: str,
        values: Sequence[int | None],
        value_min: int = None,
        value_max: int = None,
    ):
        if len(values) != len(self._ids):
            raise ValueError(f"The length of {name} must match the number of servos")
        for dxl_id, value in zip(self._ids, values):
            if value is None:
                continue
            if value_min is not None:
                value = max(value, value_min)
            if value_max is not None:
                value = min(value, value_max)
            self._storage_map[name][self._ids.index(dxl_id)] = value
            print(f"Set {name} {value} for ID {dxl_id}", flush=True)

    def set_by_name(self, name: str, values: Sequence[int | None]):
        self._set_group(
            name=name,
            values=values,
            value_min=CTRL_TABLE[name].get("min", None),
            value_max=CTRL_TABLE[name].get("max", None),
        )

    def _read_group(self, name: str) -> Sequence[int]:
        if name == "present_position":
            name = "goal_position"
        elif name == "present_current":
            name = "goal_current"
        if name not in self._storage_map:
            raise ValueError(f"Register {name} not found.")
        return self._storage_map[name].copy()

    def read_by_name(self, name: str) -> Sequence[int]:
        return self._read_group(name)

    def set_joints(self, joint_angles: Sequence[float | None]):
        self.set_by_name(
            "goal_position",
            [
                rad_to_pulses(angle) if angle is not None else None
                for angle in joint_angles
            ],
        )

    def get_joints(self) -> np.ndarray:
        return pulses_to_rad(self._storage_map["goal_position"].copy())

    def torque_enabled(self) -> bool:
        return bool(np.any(self._storage_map["torque_enable"]))

    def set_torque_mode(self, enable: bool):
        torque_value = 1 if enable else 0
        self.set_by_name("torque_enable", [torque_value] * len(self._ids))

    def close(self):
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self, ids: Sequence[int], port: str = "/dev/ttyUSB0", baudrate: int = 57600
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._ids = ids
        self._joint_angles = None
        self._lock = Lock()

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)

        # Create group sync read/write handlers for each CTRL_TABLE entry
        self._groupSyncReadHandlers = {}
        self._groupSyncWriteHandlers = {}
        for key, entry in CTRL_TABLE.items():
            self._groupSyncReadHandlers[key] = GroupSyncRead(
                self._portHandler,
                self._packetHandler,
                entry["addr"],
                entry["len"],
            )
            # Add parameters for each Dynamixel servo to the group sync read
            for dxl_id in self._ids:
                if not self._groupSyncReadHandlers[key].addParam(dxl_id):
                    raise RuntimeError(
                        f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                    )
            if "read_only" not in entry or not entry["read_only"]:
                self._groupSyncWriteHandlers[key] = GroupSyncWrite(
                    self._portHandler,
                    self._packetHandler,
                    entry["addr"],
                    entry["len"],
                )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # Disable torque for all Dynamixel servos
        try:
            self.set_torque_mode(False)
        except Exception as e:
            print(f"port: {port}, {e}")

        self._stop_thread = Event()
        self._start_reading_thread()

    def _set_group(
        self,
        name: str,
        values: Sequence[int | None],
        groupSyncWriteHandler,
        value_length: int,
        value_min: int = None,
        value_max: int = None,
    ):
        if len(values) != len(self._ids):
            raise ValueError(f"The length of {name} must match the number of servos")
        with self._lock:
            for dxl_id, value in zip(self._ids, values):
                if value is None:
                    continue
                if value_min is not None:
                    value = max(value, value_min)
                if value_max is not None:
                    value = min(value, value_max)
                # Convert value to little-endian byte array of length value_length
                param = [(value >> (8 * i)) & 0xFF for i in range(value_length)]
                result = groupSyncWriteHandler.addParam(dxl_id, param)
                if not result:
                    raise RuntimeError(
                        f"Failed to set {name} for Dynamixel with ID {dxl_id}"
                    )
            comm_result = groupSyncWriteHandler.txPacket()
            if comm_result != COMM_SUCCESS:
                raise RuntimeError(f"Failed to syncwrite {name}")
            groupSyncWriteHandler.clearParam()

    def set_by_name(self, name: str, values: Sequence[int | None]):
        self._set_group(
            name=name,
            values=values,
            groupSyncWriteHandler=self._groupSyncWriteHandlers[name],
            value_length=CTRL_TABLE[name]["len"],
            value_min=CTRL_TABLE[name].get("min", None),
            value_max=CTRL_TABLE[name].get("max", None),
        )

    def _read_group(
        self, name: str, groupSyncReadHandler, value_length: int
    ) -> list[int]:
        with self._lock:
            result = groupSyncReadHandler.txRxPacket()
            if result != COMM_SUCCESS:
                raise RuntimeError(f"Failed to sync read {name}, comm result: {result}")
            values = []
            for dxl_id in self._ids:
                if groupSyncReadHandler.isAvailable(
                    dxl_id, CTRL_TABLE[name]["addr"], value_length
                ):
                    value = groupSyncReadHandler.getData(
                        dxl_id, CTRL_TABLE[name]["addr"], value_length
                    )
                    value = int(np.int32(np.uint32(value)))
                    values.append(value)
                else:
                    raise RuntimeError(
                        f"Failed to get {name} for Dynamixel with ID {dxl_id}"
                    )
            return values

    def read_by_name(self, name: str) -> list[int]:
        return self._read_group(
            name=name,
            groupSyncReadHandler=self._groupSyncReadHandlers[name],
            value_length=CTRL_TABLE[name]["len"],
        )

    def set_joints(self, joint_angles: Sequence[float | None]):
        joint_angles_pulses = [
            rad_to_pulses(rad) if rad is not None else None for rad in joint_angles
        ]
        self.set_by_name("goal_position", joint_angles_pulses)

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles_loop)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_angles_loop(self):
        # Continuously read joint angles and update the joint_angles array
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            try:
                _joint_angles = self._read_group(
                    "present_position",
                    self._groupSyncReadHandlers["present_position"],
                    CTRL_TABLE["present_position"]["len"],
                )
                _joint_angles = np.array(_joint_angles, dtype=int)
                self._joint_angles = _joint_angles
            except RuntimeError as e:
                print(f"warning, comm failed: {e}")
                continue

    def get_joints(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        while self._joint_angles is None:
            time.sleep(0.1)
        _j = self._joint_angles.copy()
        return pulses_to_rad(_j)

    def torque_enabled(self) -> bool:
        try:
            torque_values = self.read_by_name("torque_enable")
            return any(val == 1 for val in torque_values)
        except RuntimeError:
            return False

    def set_torque_mode(self, enable: bool):
        torque_value = 1 if enable else 0
        self.set_by_name("torque_enable", [torque_value] * len(self._ids))

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self._portHandler.closePort()


def main():
    # Set the port, baudrate, and servo IDs
    ids = [1]

    # Create a DynamixelDriver instance
    try:
        driver = DynamixelDriver(ids)
    except FileNotFoundError:
        driver = DynamixelDriver(ids, port="/dev/cu.usbserial-FT7WBMUB")

    # Test setting torque mode
    driver.set_torque_mode(True)
    driver.set_torque_mode(False)

    # Test reading the joint angles
    try:
        while True:
            joint_angles = driver.get_joints()
            print(f"Joint angles for IDs {ids}: {joint_angles}")
            # print(f"Joint angles for IDs {ids[1]}: {joint_angles[1]}")
    except KeyboardInterrupt:
        driver.close()


if __name__ == "__main__":
    main()  # Test the driver
