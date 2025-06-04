import copy
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    DMChainCanInterface,
    MotorChain,
    MotorInfo,
    ReceiveMode,
)
from i2rt.robots.robot import Robot
from i2rt.robots.utils import GripperForceLimiter, JointMapper
from i2rt.utils.mujoco_utils import MuJoCoKDL

I2RT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/yam/yam.xml")
ARX_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arx_r5/arx.xml")

import logging
@dataclass
class JointStates:
    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray

    def asdict(self) -> Dict[str, Any]:
        return {
            "names": self.names,
            "pos": self.pos.flatten().tolist(),
            "vel": self.vel.flatten().tolist(),
            "eff": self.eff.flatten().tolist(),
        }


@dataclass
class JointCommands:
    torques: np.ndarray

    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    indices: Optional[List[int]] = None

    @classmethod
    def init_all_zero(cls, n_joints: int) -> "JointCommands":
        return cls(
            torques=np.zeros(n_joints),
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
        )


class MotorChainRobot(Robot):
    """A generic Robot protocol."""

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: Optional[str] = None,
        use_gravity_comp: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_comp_factor: float = 1.0,  # New parameter with default value
        gripper_index: Optional[int] = None,  # Zero starting index: if you have a 6 dof arm and last one is gripper: 6
        kp: Union[float, List[float]] = 10.0,
        kd: Union[float, List[float]] = 1.0,
        joint_limits: Optional[np.ndarray] = None,
        gripper_limits: Optional[np.ndarray] = None,  # [closed, open]
        limit_gripper_force: float = -1,  # whether to limit the gripper effort when it is blocked. -1 means no limit.
        clip_motor_torque: float = np.inf,  # clip the offset motor torque, real motor torque can still still be larger than this setting depending on the motor onboard PID loop
        gripper_type: str = "yam_small",
    ) -> None:
        if gripper_index is not None:
            assert gripper_limits is not None, "Gripper limits are required if gripper index is provided."

        if joint_limits is not None:
            joint_limits = np.array(joint_limits)
            assert np.all(
                joint_limits[:, 0] < joint_limits[:, 1]
            ), "Lower joint limits must be smaller than upper limits"
        self._last_gripper_command_qpos = None
        self._joint_limits = joint_limits
        assert clip_motor_torque >= 0.0
        self._clip_motor_torque = clip_motor_torque
        self.motor_chain = motor_chain
        self.use_gravity_comp = use_gravity_comp
        self.gravity_comp_factor = gravity_comp_factor  # Store the factor

        # variables for gripper effort limiting
        self._gripper_index = gripper_index
        self.remapper = JointMapper({}, len(motor_chain))  # so it works without gripper
        self._gripper_limits = gripper_limits
        self._gripper_adjusted_qpos = None
        if self._gripper_index is not None:
            self._gripper_force_limiter = GripperForceLimiter(
                max_force=limit_gripper_force, gripper_type=gripper_type, kp=kp[gripper_index]
            )  # force in newton
            self._limit_gripper_force = limit_gripper_force

            self.remapper = JointMapper(
                index_range_map={gripper_index: gripper_limits},
                total_dofs=len(motor_chain),
            )

        # make sure kp, kd are float number not int
        self._kp = (
            np.array(
                [
                    kp,
                ]
                * len(motor_chain)
            )
            if isinstance(kp, float)
            else np.array(kp)
        )
        self._kd = (
            np.array(
                [
                    kd,
                ]
                * len(motor_chain)
            )
            if isinstance(kd, float)
            else np.array(kd)
        )
        if gripper_index is not None:
            assert (
                gripper_index == len(motor_chain) - 1
            ), f"Gripper index should be the last one, but got {gripper_index}"
        if xml_path is not None:
            self.xml_path = os.path.expanduser(xml_path)
            self.kdl = MuJoCoKDL(self.xml_path)
            if gravity is not None:
                self.kdl.set_gravity(gravity)
        else:
            assert use_gravity_comp is False, "Gravity compensation requires a valid XML path."

        self._commands = JointCommands.init_all_zero(len(motor_chain))

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_state: Optional[JointStates] = None
        while self._joint_state is None:
            # wait to recive joint data
            time.sleep(0.05)
            self._joint_state = self._motor_state_to_joint_state(self.motor_chain.read_states())
        
        self._stop_event = threading.Event()  # Add a stop event
        self._server_thread = threading.Thread(target=self.start_server, name="robot_server")
        self._server_thread.start()

    def get_robot_info(self) -> Dict[str, Any]:
        """Get the robot information, such as kp, kd, joint limits, gripper limits, etc."""
        return {
            "kp": self._kp,
            "kd": self._kd,
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gravity_comp_factor": self.gravity_comp_factor,
            "limit_gripper_effort": self._limit_gripper_force,
            "gripper_index": self._gripper_index,
        }

    def start_server(self) -> None:
        """Start the server."""
        last_time = time.time()
        iteration_count = 0
        self.update()

        logging.info("initializing, ....")

        while not self._stop_event.is_set():  # Check the stop event
            current_time = time.time()
            elapsed_time = current_time - last_time

            self.update()
            if not self.motor_chain.running:
                logging.error("Motor chain is not running, exiting")
                exit()
            time.sleep(0.004)

            iteration_count += 1
            if elapsed_time >= 10.0:
                control_frequency = iteration_count / elapsed_time
                # Overwrite the current line with the new frequency information
                logging.info(f"Grav Comp Control Frequency: {control_frequency:.2f} Hz")
                if control_frequency < 100:
                    logging.warning("Gravity compensation control loop is slow, current frequency: {control_frequency:.2f} Hz")
                # Reset the counter and timer
                last_time = current_time
                iteration_count = 0

    def update(self) -> None:
        """Update the robot.

        Send Torques and update the joint state.
        """
        with self._command_lock:
            joint_commands = copy.deepcopy(self._commands)
        with self._state_lock:
            g = self._compute_gravity_compensation(self._joint_state)
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)

            if self._gripper_index is not None:
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    # Get current gripper state in raw robot joint pos space
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                            self._gripper_index
                        ],
                        "current_qvel": self._joint_state.vel[self._gripper_index],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(joint_commands.pos)[
                            self._gripper_index
                        ],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }

                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)

                # add final clip so the gripper won't be over-adjusted
                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]
            if not self.motor_chain.start_thread_flag:
                self.motor_chain.set_commands(
                    motor_torques,
                    pos=joint_commands.pos,
                    vel=joint_commands.vel,
                    kp=joint_commands.kp,
                    kd=joint_commands.kd,
                )
                self.motor_chain.start_thread()
                self.motor_chain.start_thread_flag = True
            # Send commands to motor chain and update joint state
            motor_state = self.motor_chain.set_commands(
                motor_torques,
                pos=joint_commands.pos,
                vel=joint_commands.vel,
                kp=joint_commands.kp,
                kd=joint_commands.kd,
            )
            self._joint_state = self._motor_state_to_joint_state(motor_state)

    def _motor_state_to_joint_state(self, motor_state: List[MotorInfo]) -> JointStates:
        """Convert motor state to joint state.

        Args:
            motor_state (List[Any]): The motor state.

        Returns:
            Dict[str, np.ndarray]: The joint state.
        """
        names = [str(i) for i in range(len(motor_state))]
        pos = np.array([motor.pos for motor in motor_state])
        pos = self.remapper.to_command_joint_pos_space(pos)
        vel = np.array([motor.vel for motor in motor_state])
        vel = self.remapper.to_command_joint_vel_space(vel)
        eff = np.array([motor.eff for motor in motor_state])
        return JointStates(names=names, pos=pos, vel=vel, eff=eff)

    def _compute_gravity_compensation(self, joint_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        if joint_state is None or not self.use_gravity_comp:
            return np.zeros(len(self.motor_chain))
        elif self.use_gravity_comp:
            q = joint_state.pos[: self._gripper_index] if self._gripper_index is not None else joint_state.pos
            t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            # print gravity torque to 2f
            if np.max(np.abs(t)) > 20.0:
                print([f"{s:.2f}" for s in t])
                raise RuntimeError("too large torques")
            if self._gripper_index is None:
                return self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            else:
                t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
                return np.append(t, 0.0)

    # ----------------- Server Functions ----------------- #

    def num_dofs(self) -> int:
        """Get the number of joints of the robot, including the gripper.

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.motor_chain)

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot, including the gripper in radian.

        Returns:
            T: The current state of the leader robot.
        """
        with self._state_lock:
            return self._joint_state.pos

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        if self._joint_limits is not None:
            if self._gripper_index is not None:
                pos[: self._gripper_index] = np.clip(
                    pos[: self._gripper_index],
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )
            else:
                pos = np.clip(pos, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_pos (np.ndarray): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_pos)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.kp = self._kp
            self._commands.kd = self._kd

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_state["pos"])
        vel = joint_state["vel"]
        self._commands = JointCommands.init_all_zero(len(self.motor_chain))
        kp = joint_state.get("kp", self._kp)
        kd = joint_state.get("kd", self._kd)
        with self._command_lock:
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.vel = self.remapper.to_robot_joint_vel_space(vel)
            self._commands.kp = kp
            self._commands.kd = kd

    def zero_torque_mode(self) -> None:
        print("in zero_torque_mode", self)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = np.zeros(len(self.motor_chain))

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        with self._state_lock:
            if self._gripper_index is None:
                return {
                    "joint_pos": self._joint_state.pos,
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            else:
                return {
                    "joint_pos": self._joint_state.pos[: self._gripper_index],
                    "gripper_pos": np.array([self._joint_state.pos[self._gripper_index]]),
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Move the robot to a given joint positions."""
        with self._state_lock:
            current_pos = self._joint_state.pos
        assert len(current_pos) == len(target_joint_positions)
        steps = 50  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor
            target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
            self.command_joint_pos(target_pos)
            time.sleep(time_interval_s / steps)

    def close(self) -> None:
        """Safely close the robot by setting all torques to zero."""
        # self.move_to_zero()
        self._stop_event.set()  # Signal the thread to stop
        self._server_thread.join()  # Wait for the thread to finish
        self.motor_chain.close()
        print("Robot closed with all torques set to zero.")


def get_yam_robot(channel: str = "can0", 
                  model_path: str = YAM_XML_PATH, 
                  motor_timeout_enabled: bool = True) -> MotorChainRobot:
    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
    ]
    motor_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    motor_directions = [1, 1, 1, 1, 1, 1, 1]
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        start_thread=motor_timeout_enabled,
    )
    motor_states = motor_chain.read_states()
    
    current_pos = [m.pos for m in motor_states]
    logging.info(f"current_pos: {current_pos}")

    for idx, motor_state in enumerate(motor_states):
        motor_position = motor_state.pos
        # if not within -pi to pi, set to the nearest equivalent position
        if motor_position < -np.pi:
            logging.info(f'motor {idx} is at {motor_position}, adding {2 * np.pi}')
            extra_offset = -2 * np.pi
        elif motor_position > np.pi:
            logging.info(f'motor {idx} is at {motor_position}, subtracting {2 * np.pi}')
            extra_offset = +2 * np.pi
        else:
            extra_offset = 0.0
        motor_offsets[idx] += extra_offset
    motor_chain.close()
    time.sleep(0.5)
    logging.info(f"adjusted motor_offsets: {motor_offsets}")
    
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        start_thread=motor_timeout_enabled,
        )
    motor_states = motor_chain.read_states()
    logging.info(f"YAN initial motor_states: {motor_states}")
    return MotorChainRobot(
        motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.3,
        gripper_index=6,
        gripper_limits=np.array([0.0, -2.7]),
        kp=np.array([80, 80, 80, 40, 10, 10, 20]),
        kd=np.array([5, 5, 5, 1.5, 1.5, 1.5, 0.5]),
        gripper_type="yam_small",
        limit_gripper_force=50.0,
    )


def get_arx_robot(channel: str = "can0", model_path: str = ARX_XML_PATH) -> MotorChainRobot:
    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x04, "DM4340"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
        [0x08, "DM4310"],
    ]
    motor_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    motor_directions = [1, 1, 1, 1, 1, 1, 1]
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="arx_real",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
    )
    return MotorChainRobot(
        motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.3,
        gripper_index=6,
        gripper_limits=np.array([0.0, 4.93]),
        kp=np.array([80, 80, 80, 40, 10, 10, 20]),
        kd=np.array([5, 5, 5, 1.5, 1.5, 1.5, 0.5]),
        limit_gripper_force=20.0,
        gripper_type="arx_92mm_linear",
    )


if __name__ == "__main__":
    import argparse
    import time

    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="yam")
    args.add_argument("--channel", type=str, default="can0")
    args.add_argument("--operation_mode", type=str, default="gravity_comp")
    args.add_argument("--motor_timeout_disabled", action='store_true')
    args = args.parse_args()

    if args.model == "arx":
        robot = get_arx_robot(args.channel)
    elif args.model == "yam":
        robot = get_yam_robot(args.channel, motor_timeout_enabled=not args.motor_timeout_disabled)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.operation_mode == "gravity_comp":
        while True:
            # print(robot.get_observations())
            time.sleep(1)
    elif args.operation_mode == "test_gripper":
        for _ in range(30):
            for gripper_pos in [0.8, 0.0]:
                print(f"gripper_pos: {gripper_pos}")
                robot.command_joint_pos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pos]))
                time.sleep(4)
                print(robot.get_observations())
