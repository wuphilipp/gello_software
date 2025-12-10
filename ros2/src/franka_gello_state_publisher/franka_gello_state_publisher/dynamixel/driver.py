import time
from glob import glob
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Protocol, Sequence
from serial import SerialException

import numpy as np
import yaml
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS


# Configuration loader for motor types
def load_motor_config(motor_type: str = "xl330") -> dict:
    """
    Load motor configuration from YAML file.

    Parameters
    ----------
    motor_type : str, optional
        The type of motor to load configuration for (default is "xl330").

    Returns
    -------
    dict
        The motor configuration dictionary.

    """
    config_dir = Path(__file__).parent / "motor_configs"
    config_path = config_dir / f"{motor_type}.yaml"

    if not config_path.exists():
        available_types = [f.stem for f in config_dir.glob("*.yaml")]
        raise ValueError(
            f"Motor type '{motor_type}' not found. Available types: {available_types}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class DynamixelDriverProtocol(Protocol):
    def get_joints(self) -> np.ndarray:
        """
        Get the current joint angles in radians.

        Returns
        -------
        np.ndarray
            An array of joint angles.

        """
        ...

    def write_value_by_name(self, name: str, values: Sequence[int | None]):
        """
        Set a group of values by name.

        Parameters
        ----------
        name : str
            The name of the control parameter.
        values : Sequence[int | None]
            A list of values to set for each servo.

        """
        ...

    def read_value_by_name(self, name: str) -> Sequence[int]:
        """
        Read a group of values by name.

        Parameters
        ----------
        name : str
            The name of the control parameter to read.

        Returns
        -------
        Sequence[int]
            A list of values read from the servos.

        """
        ...

    def start_joint_polling(self) -> None:
        """Start the background polling thread."""
        ...

    def stop_joint_polling(self) -> None:
        """Stop the background polling thread."""
        ...

    def close(self) -> None:
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

    def get_joints(self) -> np.ndarray:
        return self._pulses_to_rad(self._storage_map["goal_position"].copy())

    def start_joint_polling(self) -> None:
        pass

    def stop_joint_polling(self) -> None:
        pass

    def close(self) -> None:
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self,
        ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        motor_type: str = "xl330",
        autostart_polling: bool = False,
    ):
        """
        Initialize the DynamixelDriver class.

        Parameters
        ----------
        ids : Sequence[int]
            A list of IDs for the Dynamixel servos.
        port : str, optional
            The USB port to connect to the arm (default is "/dev/ttyUSB0").
        baudrate : int, optional
            The baudrate for communication (default is 57600).
        motor_type : str, optional
            The type of motor to use, e.g., "xl330" (default is "xl330").
        autostart_polling : bool, optional
            Whether to automatically start the joint polling thread (default is False).

        Raises
        ------
        ConnectionError
            If the port cannot be opened or is already in use.
        RuntimeError
            If there is an error reading or writing to the servos.

        """
        self._ids = ids
        self._port = port
        self._baudrate = baudrate
        self._buffered_joint_positions = None
        self._lock = Lock()

        # Load motor configuration
        self._motor_config = load_motor_config(motor_type)
        self._ctrl_table = self._motor_config["control_table"]
        self._operating_modes = self._motor_config["operating_modes"]
        self._pulses_per_revolution = self._motor_config["pulses_per_revolution"]

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(self._port)
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
                    raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")
            if "read_only" not in entry or not entry["read_only"]:
                self._groupSyncWriteHandlers[key] = GroupSyncWrite(
                    self._portHandler,
                    self._packetHandler,
                    entry["addr"],
                    entry["len"],
                )

        # Open the port and set the baudrate
        try:
            self._portHandler.openPort()
            self._portHandler.setBaudRate(self._baudrate)
        except SerialException:
            detected_ports = self._detect_com_ports()
            if self._port in detected_ports:
                msg = "Check that you have permissions to access it."
            elif detected_ports:
                ports_list = ", ".join(detected_ports)
                msg = f"Did you specify the correct port? Detected ports: {ports_list}"
            else:
                msg = "Is the device connected? No supported devices were detected."
            raise ConnectionError(f"Could not open port {self._port}. {msg}") from None

        # Verify connection by attempting to read the model number
        try:
            self.read_value_by_name("model_number")
        except (RuntimeError, SerialException):
            raise ConnectionError(
                f"Port {self._port} opened but could not read from motors. Make sure no other "
                "process is using the same port, that all motors are wired correctly and get power."
            ) from None

        # Disable torque for all Dynamixel servos
        self.write_value_by_name("torque_enable", [0] * len(self._ids))

        self._stop_thread = Event()
        self._polling_thread = None
        self._is_polling = False

        if autostart_polling:
            self.start_joint_polling()

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
                    raise RuntimeError(f"Failed to set {name} for Dynamixel with ID {dxl_id}")
            comm_result = groupSyncWriteHandler.txPacket()
            if comm_result != COMM_SUCCESS:
                result_string = self._packetHandler.getTxRxResult(comm_result)
                raise RuntimeError(f"Failed to syncwrite {name}, {result_string}")
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

    def _read_group(self, name: str, groupSyncReadHandler, value_length: int) -> list[int]:
        with self._lock:
            result = groupSyncReadHandler.txRxPacket()
            if result != COMM_SUCCESS:
                result_string = self._packetHandler.getTxRxResult(result)
                raise RuntimeError(f"Failed to sync read {name}, {result_string}")
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
                    raise RuntimeError(f"Failed to get {name} for Dynamixel with ID {dxl_id}")
            return values

    def read_value_by_name(self, name: str) -> list[int]:
        return self._read_group(
            name=name,
            groupSyncReadHandler=self._groupSyncReadHandlers[name],
            value_length=self._ctrl_table[name]["len"],
        )

    def start_joint_polling(self) -> None:
        if self._is_polling:
            return
        self._stop_thread.clear()
        self._polling_thread = Thread(target=self._joint_polling_loop)
        self._polling_thread.daemon = True
        self._polling_thread.start()
        self._is_polling = True

    def stop_joint_polling(self) -> None:
        if not self._is_polling:
            return
        self._stop_thread.set()
        if self._polling_thread is not None:
            self._polling_thread.join()
        self._is_polling = False

    def _joint_polling_loop(self):
        # Continuously read joint positions and update the _buffered_joint_positions array

        while not self._stop_thread.is_set():
            time.sleep(0.001)
            try:
                self._buffered_joint_positions = np.array(
                    self.read_value_by_name("present_position"), dtype=int
                )
            except RuntimeError as e:
                print(f"Warning: {e}")
                continue

    def get_joints(self) -> np.ndarray:
        if self._is_polling:
            # If polling, return _buffered_joint_positions array, use copy to avoid race conditions
            while self._buffered_joint_positions is None:
                time.sleep(0.01)
            return self._pulses_to_rad(self._buffered_joint_positions.copy())
        else:
            joint_positions = np.array(self.read_value_by_name("present_position"), dtype=int)
            return self._pulses_to_rad(joint_positions)

    def close(self) -> None:
        self.stop_joint_polling()
        self._portHandler.closePort()

    def _pulses_to_rad(self, pulses) -> np.ndarray:
        """Convert pulses to radians using motor-specific configuration."""
        return np.array(pulses) / self._pulses_per_revolution * 2 * np.pi

    def _rad_to_pulses(self, rad: float) -> int:
        """Convert radians to pulses using motor-specific configuration."""
        return int(rad / (2 * np.pi) * self._pulses_per_revolution)

    def _detect_com_ports(self) -> list[str]:
        """Detect available com_ports of supported communication converters."""
        SUPPORTED_CONVERTERS = [
            "usb-FTDI_USB__-__Serial_Converter",
            "usb-ROBOTIS_OpenRB-150",
        ]
        matches = []
        for converter in SUPPORTED_CONVERTERS:
            matches.extend(glob(f"/dev/serial/by-id/{converter}*"))
        return matches
