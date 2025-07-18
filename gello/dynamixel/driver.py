import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Protocol, Sequence

import numpy as np
import yaml
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
)

# Configuration loader for motor types
def load_motor_config(motor_type: str = "xl330") -> dict:
    """Load motor configuration from YAML file.
    
    Args:
        motor_type (str): The type of motor to load configuration for.
        
    Returns:
        dict: The motor configuration dictionary.
    """
    config_dir = Path(__file__).parent / "motor_configs"
    config_path = config_dir / f"{motor_type}.yaml"
    
    if not config_path.exists():
        available_types = [f.stem for f in config_dir.glob("*.yaml")]
        raise ValueError(f"Motor type '{motor_type}' not found. Available types: {available_types}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float | None]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float | None]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for at least one of the Dynamixel servos.
        If you want to check which servos have torque enabled, you can use the read_value_by_name method.

        Returns:
            bool: True if torque is enabled for one of the servos, False if it is disabled for all.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Enable or disable torque for all Dynamixel servos.
        If you want to enable torque for a specific servo, you can use the write_value_by_name method.

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

    def write_value_by_name(self, name: str, values: Sequence[int | None]):
        """Set a group of values by name.

        Args:
            name (str): The name of the control parameter.
            values (Sequence[int | None]): A list of values to set for each servo.
        """
        ...

    def read_value_by_name(self, name: str) -> Sequence[int]:
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
    def __init__(self, ids: Sequence[int], motor_type: str = "xl330"):
        self._ids = ids
        
        # Load motor configuration
        self._motor_config = load_motor_config(motor_type)
        self._ctrl_table = self._motor_config["control_table"]
        self._pulses_per_revolution = self._motor_config["pulses_per_revolution"]
        
        self._storage_map = {
            "goal_position": np.zeros(len(ids), dtype=int),
            "goal_current": np.zeros(len(ids), dtype=int),
            "kp_d": np.zeros(len(ids), dtype=int),
            "kp_i": np.zeros(len(ids), dtype=int),
            "kp_p": np.zeros(len(ids), dtype=int),
            "operating_mode": np.zeros(len(ids), dtype=int),
            "torque_enable": np.zeros(len(ids), dtype=int),
        }

    def _pulses_to_rad(self, pulses) -> np.ndarray:
        """Convert pulses to radians using motor-specific configuration."""
        return np.array(pulses) / self._pulses_per_revolution * 2 * np.pi

    def _rad_to_pulses(self, rad: float) -> int:
        """Convert radians to pulses using motor-specific configuration."""
        return int(rad / (2 * np.pi) * self._pulses_per_revolution)

    def _set_group(
        self,
        name: str,
        values: Sequence[int | None],
        value_min: int | None = None,
        value_max: int | None = None,
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

    def write_value_by_name(self, name: str, values: Sequence[int | None]):
        self._set_group(
            name=name,
            values=values,
            value_min=self._ctrl_table[name].get("min", None),
            value_max=self._ctrl_table[name].get("max", None),
        )

    def _read_group(self, name: str) -> Sequence[int]:
        if name == "present_position":
            name = "goal_position"
        elif name == "present_current":
            name = "goal_current"
        if name not in self._storage_map:
            raise ValueError(f"Register {name} not found.")
        return self._storage_map[name].copy().tolist()

    def read_value_by_name(self, name: str) -> Sequence[int]:
        return self._read_group(name)

    def set_joints(self, joint_angles: Sequence[float | None]):
        self.write_value_by_name(
            "goal_position",
            [
                self._rad_to_pulses(angle) if angle is not None else None
                for angle in joint_angles
            ],
        )

    def get_joints(self) -> np.ndarray:
        return self._pulses_to_rad(self._storage_map["goal_position"].copy())

    def torque_enabled(self) -> bool:
        return bool(np.any(self._storage_map["torque_enable"]))

    def set_torque_mode(self, enable: bool):
        torque_value = 1 if enable else 0
        self.write_value_by_name("torque_enable", [torque_value] * len(self._ids))

    def close(self):
        pass

class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self, 
        ids: Sequence[int], 
        port: str = "/dev/ttyUSB0", 
        baudrate: int = 57600,
        motor_type: str = "xl330"
    ):
        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
            motor_type (str): The type of motor to use (e.g., "xl330").
        """
        self._ids = ids
        self._joint_angles = None
        self._lock = Lock()
        
        # Load motor configuration
        self._motor_config = load_motor_config(motor_type)
        self._ctrl_table = self._motor_config["control_table"]
        self._operating_modes = self._motor_config["operating_modes"]
        self._pulses_per_revolution = self._motor_config["pulses_per_revolution"]

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)

        # Create group sync read/write handlers for each control table entry
        self._groupSyncReadHandlers = {}
        self._groupSyncWriteHandlers = {}
        for key, entry in self._ctrl_table.items():
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
        value_min: int | None = None,
        value_max: int | None = None,
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

    def write_value_by_name(self, name: str, values: Sequence[int | None]):
        self._set_group(
            name=name,
            values=values,
            groupSyncWriteHandler=self._groupSyncWriteHandlers[name],
            value_length=self._ctrl_table[name]["len"],
            value_min=self._ctrl_table[name].get("min", None),
            value_max=self._ctrl_table[name].get("max", None),
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
                    dxl_id, self._ctrl_table[name]["addr"], value_length
                ):
                    value = groupSyncReadHandler.getData(
                        dxl_id, self._ctrl_table[name]["addr"], value_length
                    )
                    value = int(np.int32(np.uint32(value)))
                    values.append(value)
                else:
                    raise RuntimeError(
                        f"Failed to get {name} for Dynamixel with ID {dxl_id}"
                    )
            return values

    def read_value_by_name(self, name: str) -> list[int]:
        return self._read_group(
            name=name,
            groupSyncReadHandler=self._groupSyncReadHandlers[name],
            value_length=self._ctrl_table[name]["len"],
        )

    def set_joints(self, joint_angles: Sequence[float | None]):
        joint_angles_pulses = [
            self._rad_to_pulses(rad) if rad is not None else None for rad in joint_angles
        ]
        self.write_value_by_name("goal_position", joint_angles_pulses)

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
                    self._ctrl_table["present_position"]["len"],
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
        return self._pulses_to_rad(_j)

    def torque_enabled(self) -> bool:
        try:
            torque_values = self.read_value_by_name("torque_enable")
            return any(val == 1 for val in torque_values)
        except RuntimeError:
            return False

    def set_torque_mode(self, enable: bool):
        torque_value = 1 if enable else 0
        self.write_value_by_name("torque_enable", [torque_value] * len(self._ids))

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self._portHandler.closePort()

    def _pulses_to_rad(self, pulses) -> np.ndarray:
        """Convert pulses to radians using motor-specific configuration."""
        return np.array(pulses) / self._pulses_per_revolution * 2 * np.pi

    def _rad_to_pulses(self, rad: float) -> int:
        """Convert radians to pulses using motor-specific configuration."""
        return int(rad / (2 * np.pi) * self._pulses_per_revolution)


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
