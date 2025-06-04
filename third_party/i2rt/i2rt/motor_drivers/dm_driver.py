import enum
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Protocol, Tuple

import can
import numpy as np

log_level = os.getenv("LOGLEVEL", "ERROR").upper()

# if no log_level is set, set it to WARNING
logging.basicConfig(level=log_level)

# set control frequence
CONTROL_FREQ = 250
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms


@dataclass
class MotorConstants:
    POSITION_MAX: float = 12.5
    POSITION_MIN: float = -12.5

    VELOCITY_MAX: float = 45
    VELOCITY_MIN: float = -45

    TORQUE_MAX: float = 54
    TORQUE_MIN: float = -54

    KP_MAX: float = 500.0
    KP_MIN: float = 0.0
    KD_MAX: float = 5.0
    KD_MIN: float = 0.0


class MotorType:
    DM8009 = "DM8009"
    DM4310 = "DM4310"
    DM4310V = "DM4310V"
    DM4340 = "DM4340"
    DMH6215 = "DMH6215"
    DM3507 = "DM3507"

    @classmethod
    def get_motor_constants(cls, motor_type: str) -> MotorConstants:
        if motor_type == cls.DM8009:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=45,
                VELOCITY_MIN=-45,
                TORQUE_MAX=54,
                TORQUE_MIN=-54,
            )
        elif motor_type == cls.DM4310:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=30,
                VELOCITY_MIN=-30,
                TORQUE_MAX=10,
                TORQUE_MIN=-10,
                # max kp 500
                # max kd 5
            )
        elif motor_type == cls.DM4310V:
            return MotorConstants(
                POSITION_MAX=3.1415926,
                POSITION_MIN=-3.1415926,
                VELOCITY_MAX=30,
                VELOCITY_MIN=-30,
                TORQUE_MAX=10,
                TORQUE_MIN=-10,
            )
        elif motor_type == cls.DM4340:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=10,
                VELOCITY_MIN=-10,
                TORQUE_MAX=28,
                TORQUE_MIN=-28,
                # max kp 500
                # max kd 5
            )
        elif motor_type == cls.DMH6215:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=45,
                VELOCITY_MIN=-45,
                TORQUE_MAX=10,
                TORQUE_MIN=-10,
            )
        elif motor_type == cls.DM3507:
            return MotorConstants(
                POSITION_MAX=12.5,
                POSITION_MIN=-12.5,
                VELOCITY_MAX=50,
                VELOCITY_MIN=-50,
                TORQUE_MAX=5,
                TORQUE_MIN=-5,
            )
        else:
            raise ValueError(f"Motor type '{motor_type}' not recognized.")


def uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
    """Converts unsigned int to float, given range and number of bits."""
    span = x_max - x_min
    offset = x_min
    return (x_int * span / ((1 << bits) - 1)) + offset


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    """Converts a float to an unsigned int, given range and number of bits."""
    span = x_max - x_min
    offset = x_min
    x = min(x, x_max)
    x = max(x, x_min)
    return int((x - offset) * ((1 << bits) - 1) / span)


def split_int16_to_uint8(data: int) -> Tuple[int, int]:
    """Split a signed 16-bit integer into two unsigned 8-bit integers.

    Args:
        data (int): The 16-bit integer to split.

    Returns:
        Tuple[int, int]: The high and low bytes as two 8-bit unsigned integers.

    """
    # Ensure the data is within the int16 range
    data = max(min(data, 32767), -32768)
    data_int16 = np.int16(data)

    # Split the int16 into two uint8s
    low_byte = data_int16 & 0xFF
    high_byte = (data_int16 >> 8) & 0xFF
    return high_byte, low_byte


class AutoNameEnum(Enum):
    def _generate_next_value_(name: str, start: int, count: int, last_values: List[str]) -> str:
        return name


# Print the version of the 'can' library
print(can.__version__)


@dataclass
class MotorInfo:
    """Class to represent motor information.

    Attributes:
        id (int): Motor ID.
        target_torque (int): Target torque value.
        vel (float): Motor speed.
        eff (float): Motor current.
        pos (float): Encoder value.
        voltage (float): Motor voltage.
        temperature (float): Motor temperature.

    """

    id: int
    target_torque: int = 0
    vel: float = 0.0
    eff: float = 0
    pos: float = 0
    voltage: float = -1
    temp: float = -1


@dataclass
class FeedbackFrameInfo:
    id: int
    error_code: int
    error_message: str
    position: float
    velocity: float
    torque: float
    temperature_mos: float
    temperature_rotor: float


@dataclass
class EncoderInfo:
    encoder = -1
    encoder_raw = -1
    encoder_offset = -1


class MotorErrorCode:
    disabled = 0x0
    normal = 0x1
    over_voltage = 0x8
    under_voltage = 0x9
    over_current = 0xA
    mosfet_over_temperature = 0xB
    motor_over_temperature = 0xC
    loss_communication = 0xD
    overload = 0xE

    # create a dict map error code to error message
    motor_error_code_dict = {
        normal: "normal",
        disabled: "disabled",
        over_voltage: "over voltage",
        under_voltage: "under voltage",
        over_current: "over current",
        mosfet_over_temperature: "mosfet over temperature",
        motor_over_temperature: "motor over temperature",
        loss_communication: "loss communication",
        overload: "overload",
    }
    # covert to decimal
    motor_error_code_dict = {int(k): v for k, v in motor_error_code_dict.items()}

    @classmethod
    def get_error_message(cls, error_code: int) -> str:
        return cls.motor_error_code_dict.get(int(error_code), f"Unknown error code: {error_code}")


class ReceiveMode(AutoNameEnum):
    p16 = enum.auto()
    same = enum.auto()
    zero = enum.auto()
    plus_one = enum.auto()

    def get_receive_id(self, motor_id: int) -> int:
        if self == ReceiveMode.p16:
            return motor_id + 16
        elif self == ReceiveMode.same:
            return motor_id
        elif self == ReceiveMode.zero:
            return 0
        elif self == ReceiveMode.plus_one:
            return motor_id + 1
        else:
            raise NotImplementedError(f"receive_mode: {self} not recognized")

    def to_motor_id(self, receive_id: int) -> int:
        if self == ReceiveMode.p16:
            return receive_id - 16
        elif self == ReceiveMode.same:
            return receive_id
        elif self == ReceiveMode.zero:
            return 0
        else:
            raise NotImplementedError(f"receive_mode: {self} not recognized")


class ControlMode:
    MIT = "MIT"
    POS_VEL = "POS_VEL"
    VEL = "VEL"

    @classmethod
    def get_id_offset(cls, control_mode: str) -> int:
        if control_mode == cls.MIT:
            return 0x000
        elif control_mode == cls.POS_VEL:
            return 0x100
        elif control_mode == cls.VEL:
            return 0x200
        else:
            raise ValueError(f"Control mode '{control_mode}' not recognized.")


######### for passive encoder #########
@dataclass
class PassiveEncoderInfo:
    """The encoder report."""

    id: int
    """The device number, uint8."""
    position: float
    """Position, in radian."""
    velocity: float
    """Velocity, in radian/s."""
    io_inputs: List[bool]
    """The discrete inputs, list of boolean."""


class CanInterface:
    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "pcan",
        bitrate: int = 1000000,
        name: str = "default_can_interface",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        use_buffered_reader: bool = False,
    ):
        self.bus = can.interface.Bus(bustype=bustype, channel=channel, bitrate=bitrate)
        self.busstate = self.bus.state
        self.name = name
        self.receive_mode = receive_mode
        self.use_buffered_reader = use_buffered_reader
        logging.info(f"Can interface {self.name} use_buffered_reader: {use_buffered_reader}")
        if use_buffered_reader:
            # Initialize BufferedReader for asynchronous message handling
            self.buffered_reader = can.BufferedReader()
            self.notifier = can.Notifier(self.bus, [self.buffered_reader])

    def close(self) -> None:
        """Shut down the CAN bus."""
        if self.use_buffered_reader:
            self.notifier.stop()
        self.bus.shutdown()

    def _send_message_get_response(
        self, id: int, motor_id: int, data: List[int], max_retry: int = 5, expected_id: Optional[int] = None
    ) -> can.Message:
        """Send a message over the CAN bus.

        Args:
            id (int): The arbitration ID of the message.
            data (List[int]): The data payload of the message.

        Returns:
            can.Message: The message that was sent.
        """
        message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        for _ in range(max_retry):
            try:
                self.bus.send(message)
                response = self._receive_message(motor_id)

                if expected_id is None:
                    expected_id = self.receive_mode.get_receive_id(motor_id)
                if response and (expected_id == response.arbitration_id):
                    return response
                self.try_receive_message(id)
            except (can.CanError, AssertionError) as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m"
                    + f"CAN Error {self.name}: Failed to communicate with motor {id} over can bus. Retrying..."
                    + "\033[0m"
                )
                time.sleep(0.001)
        raise AssertionError(
            f"fail to communicate with the motor {id} on {self.name} at can channel {self.bus.channel_info}"
        )

    def try_receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> Optional[can.Message]:
        """Try to receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message, or None if no message is received.
        """
        try:
            return self._receive_message(motor_id, timeout)
        except AssertionError:
            return None

    def _receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> Optional[can.Message]:
        """Receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message.

        Raises:
            AssertionError: If no message is received within the timeout.
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.use_buffered_reader:
                # Use BufferedReader to get the message
                message = self.buffered_reader.get_message(timeout=0.0008)
            else:
                message = self.bus.recv(timeout=0.002)
            if message:
                return message
            else:
                message = self.bus.recv(timeout=0.0008)
                if message:
                    return message
        logging.warning(
            "\033[91m"
            + f"Failed to receive message, {self.name} motor id {motor_id} motor timeout. Check if the motor is powered on or if the motor ID exists."
            + "\033[0m"
        )


class PassiveEncoderReader:
    def __init__(self, can_interface: CanInterface, receive_mode: ReceiveMode = ReceiveMode.plus_one):
        self.can_interface = can_interface
        assert self.can_interface.use_buffered_reader, "Passive encoder reader must use buffered reader"

        self.receive_mode = receive_mode

    def read_encoder(self, encoder_id: int) -> PassiveEncoderInfo:
        # this encoder's trigger message is 0x02
        data = [0xFF, 0x02]
        message = self.can_interface._send_message_get_response(
            encoder_id, encoder_id, data, expected_id=self.receive_mode.get_receive_id(0x50E)
        )
        pos, vel, button_state = self._parse_encoder_message(message)
        result = PassiveEncoderInfo(id=encoder_id, position=pos, velocity=vel, io_inputs=button_state)
        return result

    def _parse_encoder_message(self, message: can.Message) -> PassiveEncoderInfo:
        # Standard format
        struct_format = "!B h h B"
        device_id, position, velocity, digital_inputs = struct.unpack(struct_format, message.data)

        # Convert position and velocity to radians
        position_rad = position * 2 * np.pi / 4096
        velocity_rad = velocity * 2 * np.pi / 4096
        button_state = [digital_inputs % 2, digital_inputs // 2]

        return position_rad, velocity_rad, button_state


class EncoderChain:
    def __init__(self, encoder_ids: List[int], encoder_interface: CanInterface):
        self.encoder_ids = encoder_ids
        self.encoder_interface = encoder_interface

    def read_states(self) -> List[PassiveEncoderInfo]:
        return [self.encoder_interface.read_encoder(encoder_id) for encoder_id in self.encoder_ids]


class DMSingleMotorCanInterface(CanInterface):
    """Class for CAN interface with a single motor."""

    def __init__(
        self,
        control_mode: ControlMode = ControlMode.MIT,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "pcan",
        bitrate: int = 1000000,
        receive_mode: ReceiveMode = ReceiveMode.p16,
        name: str = "default_can_DM_interface",
        use_buffered_reader: bool = False,
    ):
        super().__init__(
            channel, bustype, bitrate, receive_mode=receive_mode, name=name, use_buffered_reader=use_buffered_reader
        )
        self.control_mode = control_mode
        self.cmd_idoffset = ControlMode.get_id_offset(self.control_mode)
        self.receive_mode = receive_mode

    def _get_frame_id(self, motor_id: int) -> int:
        """Calculate the Control Frame ID for a given motor."""
        return self.cmd_idoffset + motor_id

    def motor_on(self, motor_id: int, motor_type: str) -> None:
        """Turn on the motor.

        Args:
            motor_id (int): The ID of the motor to turn on.
        """
        current_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)
        for _ in range(2):
            self.try_receive_message()

        id = motor_id  # self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFC]

        message = self._send_message_get_response(id, motor_id, data)

        # dummy motor type just check motor status
        motor_info = self.parse_recv_message(message, MotorType.DM4310)
        if int(motor_info.error_code, 16) != MotorErrorCode.normal:
            logging.info(f"motor {motor_id} error: {motor_info.error_message}")
            self.clean_error(motor_id=motor_id)
            self.try_receive_message()
            logging.info(f"motor {motor_id} error cleaned")
            # enable again
            message = self._send_message_get_response(id, motor_id, data)
        else:
            logging.info(f"motor {motor_id} is already on")
        logging.getLogger().setLevel(current_level)
        motor_info = self.parse_recv_message(message, motor_type)
        return motor_info

    def clean_error(self, motor_id: int) -> None:
        # self.try_receive_message()
        id = motor_id  # self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFB]
        logging.info("clear error")
        message = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        for _ in range(3):
            try:
                self.bus.send(message)
            except Exception as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m" + "CAN Error: Failed to communicate with motor over can bus. Retrying..." + "\033[0m"
                )
        # message = self._send_message_get_response(id, data)

    def motor_off(self, motor_id: int) -> None:
        """Turn off the motor.

        Args:
            motor_id (int): The ID of the motor to turn off.
        """
        id = self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFD]
        message = self._send_message_get_response(id, motor_id, data)

    def save_zero_position(self, motor_id: int) -> None:
        """Save the current position as zero position.

        Args:
            motor_id (int): The ID of the motor to save zero position.
        """
        id = self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFE]
        try:
            message = self._send_message_get_response(id, motor_id, data, 2)
        except AssertionError:
            pass
        # check if set zero position success
        current_state = self.set_control(id, MotorType.DM4310, 0, 0, 0, 0, 0)
        diff = abs(current_state.position)
        if diff < 0.01:
            logging.warning(f"motor {motor_id} set zero position success, current position: {current_state.position}")
        # message = self._receive_message(timeout=0.5)

    def set_control(
        self,
        motor_id: int,
        motor_type: str,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        torque: float,
    ) -> FeedbackFrameInfo:
        """Set the control of the motor and return its status.

        Args:
            motor_id (int): The ID of the motor. Check GUI for the CAN ID.
            motor_type (str): The type of the motor. Check MotorType class for available motor types.
            pos (float): The target position value.
            vel (float): The target velocity value.
            kp (float): The proportional gain value.
            kd (float): The derivative gain value.
            torque (float): The target torque value.

        Returns:
            FeedbackFrameInfo: The current state of the motor, including motor id, error code, position, velocity, torque, temperature.
        """
        frame_id = self._get_frame_id(motor_id)
        # Prepare the CAN message
        data = bytearray(8)
        if self.control_mode == ControlMode.MIT:
            const = MotorType.get_motor_constants(motor_type)

            pos_tmp = float_to_uint(pos, const.POSITION_MIN, const.POSITION_MAX, 16)
            vel_tmp = float_to_uint(vel, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
            kp_tmp = float_to_uint(kp, const.KP_MIN, const.KP_MAX, 12)
            kd_tmp = float_to_uint(kd, const.KD_MIN, const.KD_MAX, 12)
            tor_tmp = float_to_uint(torque, const.TORQUE_MIN, const.TORQUE_MAX, 12)

            # "& 0xFF" (bitwise AND with 0xFF) is used to ensure that only the lowest 8 bits (one byte) of a value are kept,
            # and any higher bits are discarded.
            data[0] = (pos_tmp >> 8) & 0xFF
            data[1] = pos_tmp & 0xFF
            data[2] = (vel_tmp >> 4) & 0xFF
            data[3] = ((vel_tmp & 0xF) << 4) | (kp_tmp >> 8)
            data[4] = kp_tmp & 0xFF
            data[5] = (kd_tmp >> 4) & 0xFF
            data[6] = ((kd_tmp & 0xF) << 4) | (tor_tmp >> 8)
            data[7] = tor_tmp & 0xFF
        elif self.control_mode == ControlMode.VEL:
            # system will only response to vel command
            can_data = struct.pack("<f", vel)
            data[0:4] = can_data[0:4]

        # Send the CAN message
        message = self._send_message_get_response(frame_id, motor_id, data)

        # Parse the received message to extract motor information
        motor_info = self.parse_recv_message(message, motor_type)
        return motor_info

    def parse_recv_message(self, message: can.Message, motor_type: str) -> FeedbackFrameInfo:
        """Parse the received message to extract motor information.

        Args:
            message (can.Message): The received message.

        Returns:
            FeedbackFrameInfo: The current state of the motor.
        """
        data = message.data
        error_int = (data[0] & 0xF0) >> 4  # TODO: error code seems incorrect, double check

        # convert error into hex
        error_hex = hex(error_int)
        error_message = MotorErrorCode.get_error_message(error_int)

        motor_id_of_this_response = self.receive_mode.to_motor_id(message.arbitration_id)
        if error_hex != "0x1":
            # print motor id and error
            logging.warning(f"motor id: {motor_id_of_this_response}, error: {error_message}")
        p_int = (data[1] << 8) | data[2]
        v_int = (data[3] << 4) | (data[4] >> 4)
        t_int = ((data[4] & 0xF) << 8) | data[5]
        temporature_mos = data[6]
        temperature_rotor = data[7]

        const = MotorType.get_motor_constants(motor_type)
        position = uint_to_float(p_int, const.POSITION_MIN, const.POSITION_MAX, 16)
        velocity = uint_to_float(v_int, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
        torque = uint_to_float(t_int, const.TORQUE_MIN, const.TORQUE_MAX, 12)
        temperature_mos = float(temporature_mos)
        temperature_rotor = float(temperature_rotor)

        return FeedbackFrameInfo(
            id=motor_id_of_this_response,
            error_code=error_hex,
            error_message=error_message,
            position=position,
            velocity=velocity,
            torque=torque,
            temperature_mos=temperature_mos,
            temperature_rotor=temperature_rotor,
        )


@dataclass
class MotorCmd:
    type: str = "pos_vel_torque"
    pos: float = 0.0
    vel: float = 0.0
    torque: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


class MotorChain(Protocol):
    """Class for CAN interface with multiple motors."""

    def __len__(self) -> int:
        """Get the number of motors in the chain."""
        raise NotImplementedError

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        """Set commands to the motors in the chain."""
        raise NotImplementedError


class DMChainCanInterface(MotorChain):
    def __init__(
        self,
        motor_list: List[Tuple[int, str]],
        motor_offset: np.ndarray,
        motor_direction: np.ndarray,
        channel: str = "PCAN_USBBUS1",
        bitrate: int = 1000000,
        start_thread: bool = True,
        motor_chain_name: str = "default_motor_chain",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        control_mode: ControlMode = ControlMode.MIT,
        # assume this driver shares the same bus interface with the motor interface
        get_same_bus_device_driver: Optional[Callable] = None,
        use_buffered_reader: bool = False,
    ):
        assert len(motor_list) > 0
        assert (
            len(motor_list) == len(motor_offset) == len(motor_direction)
        ), f"len{len(motor_list)}, len{len(motor_offset)}, len{len(motor_direction)}"
        self.motor_list = motor_list
        self.motor_offset = np.array(motor_offset)
        self.motor_direction = np.array(motor_direction)
        print(channel, bitrate)
        if "can" in channel:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bustype="socketcan",
                receive_mode=receive_mode,
                name=motor_chain_name,
                control_mode=control_mode,
                use_buffered_reader=use_buffered_reader,
            )
        else:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bitrate=bitrate,
                name=motor_chain_name,
                use_buffered_reader=use_buffered_reader,
            )
        self.state = None
        self.state_lock = threading.Lock()
        self.commands = [MotorCmd() for _ in range(len(motor_list))]
        self.command_lock = threading.Lock()

        self.same_bus_device_states = None
        self.same_bus_device_lock = threading.Lock()
        if get_same_bus_device_driver is not None:
            self.same_bus_device_driver = get_same_bus_device_driver(self.motor_interface)
        else:
            self.same_bus_device_driver = None

        self.absolute_positions = None
        self._motor_on()
        self.start_thread_flag = start_thread
        if start_thread:
            self.start_thread()

    def _update_absolute_positions(self, motor_feedback: List[MotorInfo]) -> None:
        init_mode = False
        if self.absolute_positions is None:
            self.absolute_positions = np.zeros(len(self.motor_list))
            init_mode = True

        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            const = MotorType.get_motor_constants(motor_type)
            position_min = const.POSITION_MIN
            position_max = const.POSITION_MAX
            position_range = position_max - position_min

            # Current position from feedback
            current_position = motor_feedback[idx].position

            # Previous absolute position
            previous_position = self.absolute_positions[idx]

            # Handle wrap-around
            delta_position = current_position - (previous_position % position_range)
            if delta_position > position_range / 2:  # Wrap backward
                delta_position -= position_range
            elif delta_position < -position_range / 2:  # Wrap forward
                delta_position += position_range

            if init_mode:
                self.absolute_positions[idx] = current_position
            else:
                self.absolute_positions[idx] += delta_position

    def __len__(self):
        return len(self.motor_list)

    def _joint_position_real_to_sim(self, joint_position_real: float) -> float:
        return (joint_position_real - self.motor_offset) * self.motor_direction

    def _joint_position_real_to_sim_idx(self, joint_position_real: float, idx: int) -> float:
        return (joint_position_real - self.motor_offset[idx]) * self.motor_direction[idx]

    def _joint_position_sim_to_real_idx(self, joint_position_sim: float, idx: int) -> float:
        return joint_position_sim * self.motor_direction[idx] + self.motor_offset[idx]

    def _motor_on(self) -> None:
        motor_feedback = []
        for motor_id, motor_type in self.motor_list:
            print(motor_id, motor_type)
            time.sleep(0.001)
            motor_feedback.append(self.motor_interface.motor_on(motor_id, motor_type))
        self._update_absolute_positions(motor_feedback)
        self.state = motor_feedback
        self.running = True
        print("starting separate thread for control loop")

    def start_thread(self) -> None:
        # clean error again for motor with timeout enabled
        self._motor_on()
        thread = threading.Thread(target=self._set_torques_and_update_state)
        thread.start()
        time.sleep(0.1)
        while self.state is None:
            time.sleep(0.1)
            print("waiting for the first state")

    def _set_torques_and_update_state(self) -> None:
        last_step_time = time.time()

        while self.running:
            try:
                # Maintain desired control frequency.
                while time.time() - last_step_time < CONTROL_PERIOD - 0.001:
                    time.sleep(0.001)
                curr_time = time.time()
                step_time = curr_time - last_step_time
                last_step_time = curr_time
                if step_time > 0.007:  # 7 ms
                    logging.warning(
                        f"Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop"
                    )

                # Update state.
                with self.command_lock:
                    motor_feedback = self._set_commands(self.commands)
                    errors = np.array(
                        [True if motor_feedback[i].error_code != "0x1" else False for i in range(len(motor_feedback))]
                    )
                    if np.any(errors):
                        self.running = False
                        logging.error(f"motor errors: {errors}")
                        raise Exception("motors have errors, stopping control loop")

                with self.state_lock:
                    self.state = motor_feedback
                    self._update_absolute_positions(motor_feedback)
                if self.same_bus_device_driver is not None:
                    time.sleep(0.01)  # TODO: check if this is necessary
                    with self.same_bus_device_lock:
                        # assume the same bus device is a passive input device (no commands to send) for now.
                        self.same_bus_device_states = self.same_bus_device_driver.read_states()
                        time.sleep(0.001)
                time.sleep(0.0005)  # this is necessary, else the locks will not be released
            except Exception as e:
                print(f"DM Error in control loop: {e}")
                raise e

    def _set_commands(self, commands: List[MotorCmd]) -> List[MotorInfo]:
        motor_feedback = []
        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            torque = commands[idx].torque * self.motor_direction[idx]
            pos = self._joint_position_sim_to_real_idx(commands[idx].pos, idx)

            vel = commands[idx].vel * self.motor_direction[idx]
            kp = commands[idx].kp
            kd = commands[idx].kd
            try:
                fd_back = self.motor_interface.set_control(
                    motor_id=motor_id,
                    motor_type=motor_type,
                    pos=pos,
                    vel=vel,
                    kp=kp,
                    kd=kd,
                    torque=torque,
                )
            except Exception as e:
                print(f"{idx}th motor failed with info {motor_info}")
                raise e

            motor_feedback.append(fd_back)
        return motor_feedback

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        motor_infos = []
        with self.state_lock:
            for idx in range(len(self.motor_list)):
                state = self.state[idx]
                motor_infos.append(
                    MotorInfo(
                        id=state.id,
                        target_torque=torques[idx] if torques is not None else 0.0,
                        vel=state.velocity * self.motor_direction[idx],
                        eff=state.torque * self.motor_direction[idx],
                        pos=self._joint_position_real_to_sim_idx(self.absolute_positions[idx], idx),
                        temp=state.temperature_rotor,
                    )
                )
        return motor_infos

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> List[MotorInfo]:
        command = []
        for idx in range(len(self.motor_list)):
            command.append(
                MotorCmd(
                    torque=torques[idx],
                    pos=pos[idx] if pos is not None else 0.0,
                    vel=vel[idx] if vel is not None else 0.0,
                    kp=kp[idx] if kp is not None else 0.0,
                    kd=kd[idx] if kd is not None else 0.0,
                )
            )
        with self.command_lock:
            self.commands = command
        if get_state:
            return self.read_states(torques=torques)

    def get_same_bus_device_states(self) -> Any:
        with self.same_bus_device_lock:
            return self.same_bus_device_states

    def close(self) -> None:
        self.running = False


class MultiDMChainCanInterface(MotorChain):
    """Class for interfacing with multiple asynchronous CAN interfaces."""

    def __init__(
        self,
        interfaces: List[DMChainCanInterface],
    ):
        self.interfaces = interfaces

    def __len__(self):
        return sum([len(inter) for inter in self.interfaces])

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        start_idx = 0
        motor_infos = []
        for inter in self.interfaces:
            inter_len = len(inter)
            end_idx = start_idx + inter_len
            inter_torques = torques[start_idx:end_idx]
            inter_pos = pos[start_idx:end_idx] if pos is not None else None
            inter_vel = vel[start_idx:end_idx] if vel is not None else None
            inter_kp = kp[start_idx:end_idx] if kp is not None else None
            inter_kd = kd[start_idx:end_idx] if kd is not None else None
            infos = inter.set_commands(inter_torques, inter_pos, inter_vel, inter_kp, inter_kd)
            motor_infos.extend(infos)
            start_idx = end_idx
        return motor_infos


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--channel", type=str, default="can0")
    args = args.parse_args()
    channel = args.channel
    motor_chain_name = "yam_real"
    motor_list = [
        [0x01, "DM4310"],
        [0x02, "DM4310"],
        [0x03, "DM4310"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
    ]
    motor_offsets = [0] * len(motor_list)
    motor_directions = [1] * len(motor_list)
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name,
        receive_mode=ReceiveMode.p16,
    )
    while True:
        # for _ in range(10):
        motor_chain.set_commands(np.zeros(len(motor_list)))
        # print(motor_chain.read_states())
        time.sleep(0.001)
