import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, TypedDict, Iterator, Tuple
from franka_gello_state_publisher.dynamixel.driver import DynamixelDriver


class GelloHardwareParams(TypedDict):
    """Type-safe parameter dictionary for GelloHardware initialization."""

    com_port: str
    gello_name: str
    num_arm_joints: int
    joint_signs: List[int]
    gripper: bool
    gripper_range_rad: List[float]
    assembly_offsets: List[float]
    baudrate: int
    dynamixel_kp_p: List[int]
    dynamixel_kp_i: List[int]
    dynamixel_kp_d: List[int]
    dynamixel_torque_enable: List[int]
    dynamixel_goal_position: List[float]


@dataclass
class DynamixelControlConfig:
    """Tracks current dynamixel parameter state."""

    kp_p: List[int] = field(default_factory=list)
    kp_i: List[int] = field(default_factory=list)
    kp_d: List[int] = field(default_factory=list)
    torque_enable: List[int] = field(default_factory=list)
    goal_position: List[int] = field(default_factory=list)
    goal_current: List[int] = field(default_factory=list)
    operating_mode: List[int] = field(default_factory=list)

    # Define the order for parameter updates
    _UPDATE_ORDER = [
        "operating_mode",  # resets kp_p, kp_i, kp_d, goal_current, goal_position
        "goal_current",
        "kp_p",
        "kp_i",
        "kp_d",
        "torque_enable",  # resets goal_position
        "goal_position",
    ]

    def __contains__(self, param_name: str) -> bool:
        """Check if parameter exists in this configuration."""
        return hasattr(self, param_name)

    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
        """Iterate through parameters in correct update order."""
        for param_name in self._UPDATE_ORDER:
            if hasattr(self, param_name):
                yield param_name, getattr(self, param_name)

    def __getitem__(self, param_name: str) -> List[int]:
        """Enable dictionary-style access for getting values."""
        if not hasattr(self, param_name):
            raise KeyError(f"Parameter '{param_name}' not found")
        return getattr(self, param_name)

    def __setitem__(self, param_name: str, value: List[int]) -> None:
        """Enable dictionary-style access for setting values."""
        if not hasattr(self, param_name):
            raise KeyError(f"Parameter '{param_name}' not found")
        setattr(self, param_name, value)


class GelloHardware:
    """Hardware interface for GELLO teleoperation device."""

    # From https://frankarobotics.github.io/docs/robot_specifications.html#limits-for-franka-research-3-fr3
    JOINT_POSITION_LIMITS = np.array(
        [
            [-2.9007, 2.9007],  # -166/166 deg
            [-1.8361, 1.8361],  # -105/105 deg
            [-2.9007, 2.9007],  # -166/166 deg
            [-3.0770, -0.1169],  # -176/-7 deg
            [-2.8763, 2.8763],  # -165/165 deg
            [0.4398, 4.6216],  # -25/265 deg
            [-3.0508, 3.0508],  # -175/175 deg
        ]
    )
    MID_JOINT_POSITIONS = JOINT_POSITION_LIMITS.mean(axis=1)

    OPERATING_MODE = 5  # CURRENT_BASED_POSITION_MODE
    CURRENT_LIMIT = 600  # mA
    ALLOWED_BAUDRATES = (57600, 115200, 1000000)
    BPS_TO_REG = {57600: 1, 115200: 2, 1000000: 3}
    SCAN_BAUDS = (115200, 57600, 1000000)
    BAUD_WRITE_SETTLE_S = 0.1

    @staticmethod
    def normalize_joint_positions(
        raw_positions: np.ndarray,
        assembly_offsets: np.ndarray,
        joint_signs: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize joint positions to working range centered on mid_positions.

        Raw joint positions (in rad) are based on the motor's internal position register.
        On power up, this register resets to [0, 2*Pi], losing tracking of full rotations (multi-turn).
        Furthermore, these raw values are offset by the physical assembly position and may need to be
        inverted based on motor direction.

        This function converts raw motor positions to absolute positions by:
        1. Removing physical assembly offsets
        2. Applying joint direction signs
        3. Wrapping to range [mid-pi, mid+pi) to resolve multi-turn ambiguity from the motor's power-on reset

        Parameters
        ----------
        raw_positions : np.ndarray
            Raw motor positions in radians
        assembly_offsets : np.ndarray
            Physical assembly position offsets
        joint_signs : np.ndarray
            Direction multipliers for each joint

        Returns
        -------
        np.ndarray
            Normalized joint positions in radians

        """
        return (
            np.mod(
                (raw_positions - assembly_offsets) * joint_signs
                - GelloHardware.MID_JOINT_POSITIONS,
                2 * np.pi,
            )
            - np.pi
            + GelloHardware.MID_JOINT_POSITIONS
        )

    def __init__(
        self,
        hardware_config: GelloHardwareParams,
        logger,
    ) -> None:
        self._logger = logger
        self._com_port = hardware_config["com_port"]
        self._gello_name = hardware_config["gello_name"]
        self._num_arm_joints = hardware_config["num_arm_joints"]
        self._joint_signs = np.array(hardware_config["joint_signs"])
        self._gripper = hardware_config["gripper"]
        self._num_total_joints = self._num_arm_joints + (1 if self._gripper else 0)
        self._gripper_range_rad = hardware_config["gripper_range_rad"]
        self._assembly_offsets = np.array(hardware_config["assembly_offsets"])
        self._desired_baudrate = int(hardware_config["baudrate"])

        self._initialize_driver()

        self._initial_arm_joints_raw = self._driver.get_joints()[: self._num_arm_joints]

        # Normalize the raw joint positions initially. After this, all position updates will be done
        # incrementally to maintain continuity.
        initial_arm_joints = self.normalize_joint_positions(
            self._initial_arm_joints_raw,
            self._assembly_offsets,
            self._joint_signs,
        )

        # Store raw initial joint positions to compute joint position deltas on update
        self._prev_arm_joints_raw = self._initial_arm_joints_raw.copy()
        # Store processed initial joint positions for updating the processed position with the deltas
        self._prev_arm_joints = initial_arm_joints.copy()

        self._dynamixel_control_config = DynamixelControlConfig(
            kp_p=hardware_config["dynamixel_kp_p"].copy(),
            kp_i=hardware_config["dynamixel_kp_i"].copy(),
            kp_d=hardware_config["dynamixel_kp_d"].copy(),
            torque_enable=hardware_config["dynamixel_torque_enable"].copy(),
            goal_position=self._goal_position_to_pulses(
                hardware_config["dynamixel_goal_position"]
            ).copy(),
            goal_current=[self.CURRENT_LIMIT] * self._num_total_joints,
            operating_mode=[self.OPERATING_MODE] * self._num_total_joints,
        )

        self._initialize_parameters()
        self._driver.start_joint_polling()

    def _initialize_driver(self) -> None:
        """Open the chain at the live baud, optionally write EEPROM, then reinit at desired."""
        desired = self._desired_baudrate
        if desired not in self.ALLOWED_BAUDRATES:
            raise ValueError(
                f"Unsupported baudrate {desired}. Allowed: {self.ALLOWED_BAUDRATES}"
            )

        joint_ids = list(range(1, self._num_total_joints + 1))
        driver = None
        live = None
        for candidate in self._baud_scan_order(desired):
            driver = self._try_open_driver(joint_ids, candidate)
            if driver is not None:
                live = candidate
                break

        if driver is None or live is None:
            raise ConnectionError(
                f"No Dynamixels on {self._com_port} at known bauds "
                f"{self._baud_scan_order(desired)}. Mixed-baud chains are not written; "
                "recover with Dynamixel Wizard 2.0."
            )

        if live != desired:
            self._logger.info(
                f"GELLO on {self._com_port} answered at {live} baud; "
                f"writing EEPROM baud_rate for {desired}."
            )
            driver.write_value_by_name(
                "baud_rate", [self.BPS_TO_REG[desired]] * len(joint_ids)
            )
            time.sleep(self.BAUD_WRITE_SETTLE_S)
            driver.close()
            try:
                driver = DynamixelDriver(
                    joint_ids, port=self._com_port, baudrate=desired
                )
            except ConnectionError as e:
                raise ConnectionError(
                    f"Wrote baud_rate={desired} on {self._com_port} but could not reopen. "
                    "Chain may be mixed-baud; recover with Dynamixel Wizard 2.0."
                ) from e

        self._driver = driver

    def _baud_scan_order(self, desired: int) -> list[int]:
        order: list[int] = []
        for baud in (desired, *self.SCAN_BAUDS):
            if baud not in order:
                order.append(baud)
        return order

    def _try_open_driver(self, joint_ids: List[int], baudrate: int) -> DynamixelDriver | None:
        try:
            return DynamixelDriver(joint_ids, port=self._com_port, baudrate=baudrate)
        except ConnectionError:
            return None

    def _initialize_parameters(self) -> None:
        """Write all dynamixel configuration parameters to hardware."""
        for param_name, param_value in self._dynamixel_control_config:
            self._driver.write_value_by_name(param_name, param_value)
            if (
                param_name == "torque_enable"
                and any(v == 1 for v in param_value)
                and "OpenRB-150" in self._com_port
            ):
                self._logger.warning(
                    "Enabling torque... Please make sure you have connected an external power "
                    "supply to the OpenRB-150 board and that the jumper is set to 'VIN(DXL)'. "
                    "Using the USB connection as a power source for torque operation may cause "
                    "damage to your PC."
                )
        time.sleep(0.1)  # Dynamixels are not immediately ready after these parameter writes

    def update_dynamixel_control_parameter(
        self, param_name: str, param_value: list[float] | list[int]
    ) -> None:
        """Update a single dynamixel parameter and handle dependencies."""
        clean_name = param_name.replace("dynamixel_", "")

        if clean_name == "goal_position":
            param_value = self._goal_position_to_pulses(param_value)

        self._dynamixel_control_config[clean_name] = param_value
        self._driver.write_value_by_name(clean_name, self._dynamixel_control_config[clean_name])
        if clean_name == "torque_enable":
            self._driver.write_value_by_name(
                "goal_position", self._dynamixel_control_config["goal_position"]
            )

    def get_joint_and_gripper_positions(self) -> tuple[np.ndarray, float]:
        """Return a tuple containing the processed joint positions and gripper position percentage."""
        joints_raw = self._driver.get_joints()
        arm_joints_raw = joints_raw[: self._num_arm_joints]
        gripper_position_raw = joints_raw[-1]
        return self.process_arm_joint_positions(arm_joints_raw), self.process_gripper_position(
            gripper_position_raw
        )

    def process_arm_joint_positions(self, arm_joints_raw: np.ndarray) -> np.ndarray:
        """
        Calculate arm joint positions from raw positions.

        Applies deltas to previous positions to maintain continuity and clamps
        to the robot's joint limits.
        """
        # Compute joint position deltas and apply to previous processed positions
        arm_joints_delta = (arm_joints_raw - self._prev_arm_joints_raw) * self._joint_signs
        arm_joints = self._prev_arm_joints + arm_joints_delta

        # Store for next update
        self._prev_arm_joints = arm_joints.copy()
        self._prev_arm_joints_raw = arm_joints_raw.copy()

        arm_joints_clipped = np.clip(
            arm_joints, self.JOINT_POSITION_LIMITS[:, 0], self.JOINT_POSITION_LIMITS[:, 1]
        )
        return arm_joints_clipped

    def process_gripper_position(self, gripper_position_raw: float) -> float:
        """Convert and clamp raw gripper position to percentage (0-1). Return 0.0 if no gripper is present."""
        if not self._gripper:
            return 0.0
        gripper_position_percent = (gripper_position_raw - self._gripper_range_rad[0]) / (
            self._gripper_range_rad[1] - self._gripper_range_rad[0]
        )
        gripper_position_clipped = max(0.0, min(1.0, gripper_position_percent))
        return gripper_position_clipped

    def disable_torque(self) -> None:
        """Disable torque on all joints."""
        self._driver.write_value_by_name("torque_enable", [0] * len(self._driver._ids))

    def _goal_position_to_pulses(self, goals: list[float]) -> list[int]:
        """Convert goal positions from radians to dynamixel pulses."""
        arm_goals = np.array(goals[: self._num_arm_joints])

        # Apply the inverse mapping of the initialization process to convert arm goals back to raw motor commands:
        # 1. Compute 'initial_rotations': The number of full 2*pi turns the motor was offset by at startup.
        # 2. Reconstruct 'arm_goals_raw': Combine the rotation offset, goal position, and assembly offsets,
        #    applying the correct joint signs to match the motor's coordinate system.
        initial_rotations = np.floor_divide(
            self._initial_arm_joints_raw - self._assembly_offsets - self.MID_JOINT_POSITIONS,
            2 * np.pi,
        )
        arm_goals_raw = (
            initial_rotations * 2 * np.pi + arm_goals + self._assembly_offsets
        ) * self._joint_signs + np.pi

        goals_raw = np.append(arm_goals_raw, goals[-1]) if self._gripper else arm_goals_raw
        goals_raw_pulses = [self._driver._rad_to_pulses(rad) for rad in goals_raw]
        return goals_raw_pulses
