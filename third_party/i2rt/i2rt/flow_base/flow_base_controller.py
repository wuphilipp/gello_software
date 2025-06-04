# Modified from https://github.com/jimmyyhwu/tidybot2
import os

os.environ["CTR_TARGET"] = "Hardware"  # pylint: disable=wrong-import-position

import atexit
import math
import os
import queue
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig
from threadpoolctl import threadpool_limits

from i2rt.motor_drivers.dm_driver import ControlMode, DMChainCanInterface

POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
h_x, h_y = 0.2 * np.array([1.0, 1.0, -1.0, -1.0]), 0.2 * np.array([-1.0, 1.0, 1.0, -1.0])


def remove_pid_file(pid_file_path: str) -> None:
    # Remove PID file if it corresponds to the current process
    if pid_file_path.exists():
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if pid == os.getpid():
            pid_file_path.unlink()


def create_pid_file(name: str) -> None:
    # Check if PID file already exists
    pid_file_path = Path(f"/tmp/{name}.pid")
    if pid_file_path.exists():
        # Get PID of other process from lock file
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())

        # Check if PID matches current process
        if pid != os.getpid():
            # PID does not match current process, check if other process is still running
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Removing stale PID file (PID {pid})")
                pid_file_path.unlink()
            else:
                raise Exception(f"Another instance of the {name} is already running (PID {pid})")

    # Write PID of current process to the file
    pid_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file_path, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n")

    # Register cleanup function to remove PID file upon exit
    atexit.register(remove_pid_file, pid_file_path)


# Vehicle
CONTROL_FREQ = 200
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms
NUM_CASTERS = 4

# Caster
b_x = -0.020
b_y = -0.0  # Lateral caster offset (m)
r = 0.05  # Wheel radius (m)
N_s = 1  # Steer gear ratio
N_r1 = 1  # Drive gear ratio (1st stage)
N_r2 = 1  # Drive gear ratio (2nd stage)
N_w = 1  # Wheel gear ratio
N_r1_r2_w = N_r1 * N_r2 * N_w
N_s_r2_w = N_s * N_r2 * N_w
TWO_PI = 2 * math.pi


class VehicleMotorController:
    def __init__(
        self,
        steering_offset: List[float],
        steering_direction: List[int],
        channel: str = "can_flow_base",
        num_casters: int = 4,
    ):
        self.num_casters = num_casters
        motor_list = []
        motor_offsets = []

        motor_directions = []
        for caster_idx in [1, 2, 3, 0]:
            motor_offsets.append(steering_offset[caster_idx])
            motor_offsets.append(0)  # drive motor no need to set offset
            motor_directions.append(steering_direction[caster_idx])
            motor_directions.append((-1) ** (caster_idx))  # drive motor direction is always 1

            caster_idx = caster_idx + 1
            steering_motor_id = caster_idx * 2 - 1
            drive_motor_id = caster_idx * 2
            motor_list.append([steering_motor_id, "DM4310V"])
            motor_list.append([drive_motor_id, "DMH6215"])

        self.motor_interface = DMChainCanInterface(
            motor_list,
            motor_offsets,
            motor_directions,
            channel=channel,
            motor_chain_name="holonomic_base",
            control_mode=ControlMode.VEL,
        )
        self.motor_offsets = motor_offsets
        self.motor_directions = motor_directions
        self.kd = np.array(
            [
                0.5,
                0.5,
            ]
            * self.num_casters
        )

        print(f"dm chain can interface: {self.motor_interface} initialized")

    def get_state(self) -> Dict[str, Any]:
        motor_states = self.motor_interface.read_states()
        steer_pos, drive_pos = [], []
        steer_vel, drive_vel = [], []
        for idx in range(self.num_casters):
            steer_idx = idx * 2
            drive_idx = idx * 2 + 1
            steer_pos.append(motor_states[steer_idx].pos)
            drive_pos.append(motor_states[drive_idx].pos)
            steer_vel.append(motor_states[steer_idx].vel)
            drive_vel.append(motor_states[drive_idx].vel)
        result_dict = {
            "steer_pos": steer_pos,
            "drive_pos": drive_pos,
            "steer_vel": steer_vel,
            "drive_vel": drive_vel,
        }
        return result_dict

    def get_positions(self) -> List[float]:
        steer_pos, drive_pos, _, _ = self.get_state()
        return steer_pos + drive_pos

    def get_velocities(self) -> List[float]:
        _, _, steer_vel, drive_vel = self.get_state()
        return steer_vel + drive_vel

    def set_velocities(self, input_dict: Dict[str, Any]) -> None:
        steer_vel, drive_vel = input_dict["steer_vel"], input_dict["drive_vel"]
        vels = []
        for i in range(self.num_casters):
            vels.append(steer_vel[i])
            vels.append(drive_vel[i])

        self.motor_interface.set_commands(
            torques=np.zeros(2 * self.num_casters),
            pos=np.zeros(2 * self.num_casters),
            vel=vels,
            kp=np.zeros(2 * self.num_casters),
            kd=self.kd,
            get_state=False,
        )

    def set_neutral(self) -> None:
        self.motor_interface.set_commands(
            torques=np.zeros(2 * self.num_casters),
            pos=np.zeros(2 * self.num_casters),
            vel=np.zeros(2 * self.num_casters),
            kp=np.zeros(2 * self.num_casters),
            kd=0.5 * np.ones(2 * self.num_casters),
        )


class CommandType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"


# Currently only used for velocity commands
class FrameType(Enum):
    GLOBAL = "global"
    LOCAL = "local"


class Vehicle:
    def __init__(
        self,
        max_vel: Tuple[float, float, float] = (0.5, 0.5, 1.57),
        max_accel: Tuple[float, float, float] = (0.25, 0.25, 0.79),
    ):
        self.max_vel = np.array(max_vel)
        self.max_accel = np.array(max_accel)

        # Use PID file to enforce single instance
        create_pid_file("base-controller")

        # Initialize hardware module
        steering_offset = [0.0, 0.0, 0.0, 0.0]
        steering_direction = [1, 1, 1, 1]
        self.num_casters = len(steering_offset)

        self.caster_module_controller = VehicleMotorController(steering_offset, steering_direction)

        # Joint space
        num_motors = 2 * NUM_CASTERS
        self.q = np.zeros(num_motors)
        self.dq = np.zeros(num_motors)

        # Operational space (global frame)
        num_dofs = 3  # (x, y, theta)
        self.x = np.zeros(num_dofs)
        self.dx = np.zeros(num_dofs)

        # C matrix relating operational space velocities to joint velocities
        self.C = np.zeros((num_motors, num_dofs))
        self.C_steer = self.C[::2]
        self.C_drive = self.C[1::2]

        # C_p matrix relating operational space velocities to wheel velocities at the contact points
        self.C_p = np.zeros((num_motors, num_dofs))
        self.C_p_steer = self.C_p[::2]
        self.C_p_drive = self.C_p[1::2]
        self.C_p_steer[:, :2] = [1.0, 0.0]
        self.C_p_drive[:, :2] = [0.0, 1.0]

        # C_qp^# matrix relating joint velocities to operational space velocities
        self.C_pinv = np.zeros((num_motors, num_dofs))
        self.CpT_Cqinv = np.zeros((num_dofs, num_motors))
        self.CpT_Cqinv_steer = self.CpT_Cqinv[:, ::2]
        self.CpT_Cqinv_drive = self.CpT_Cqinv[:, 1::2]

        # OTG (online trajectory generation)
        # Note: It would be better to couple x and y using polar coordinates
        self.otg = Ruckig(num_dofs, CONTROL_PERIOD)
        self.otg_inp = InputParameter(num_dofs)
        self.otg_out = OutputParameter(num_dofs)
        self.otg_res = Result.Working
        self.otg_inp.max_velocity = self.max_vel
        self.otg_inp.max_acceleration = self.max_accel

        # Control loop
        self.command_queue = queue.Queue(1)
        self.control_loop_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False

    def update_state(self) -> None:
        # Joint positions and velocities
        now = time.time()
        state_dict = self.caster_module_controller.get_state()
        steer_pos, drive_pos, steer_vel, drive_vel = (
            state_dict["steer_pos"],
            state_dict["drive_pos"],
            state_dict["steer_vel"],
            state_dict["drive_vel"],
        )
        for i in range(self.num_casters):
            self.q[i * 2] = steer_pos[i]
            self.q[i * 2 + 1] = drive_pos[i]
            self.dq[i * 2] = steer_vel[i]
            self.dq[i * 2 + 1] = drive_vel[i]

        q_steer = self.q[::2]
        s = np.sin(q_steer)
        c = np.cos(q_steer)

        # C matrix
        self.C_steer[:, 0] = s / b_x
        self.C_steer[:, 1] = -c / b_x
        self.C_steer[:, 2] = (-h_x * c - h_y * s) / b_x - 1.0
        self.C_drive[:, 0] = c / r - b_y * s / (b_x * r)
        self.C_drive[:, 1] = s / r + b_y * c / (b_x * r)
        self.C_drive[:, 2] = (h_x * s - h_y * c) / r + b_y * (h_x * c + h_y * s) / (b_x * r)

        # C_p matrix
        self.C_p_steer[:, 2] = -b_x * s - b_y * c - h_y
        self.C_p_drive[:, 2] = b_x * c - b_y * s + h_x

        # C_qp^# matrix
        self.CpT_Cqinv_steer[0] = b_x * s + b_y * c
        self.CpT_Cqinv_steer[1] = -b_x * c + b_y * s
        self.CpT_Cqinv_steer[2] = b_x * (-h_x * c - h_y * s - b_x) + b_y * (h_x * s - h_y * c - b_y)
        self.CpT_Cqinv_drive[0] = r * c
        self.CpT_Cqinv_drive[1] = r * s
        self.CpT_Cqinv_drive[2] = r * (h_x * s - h_y * c - b_y)
        with threadpool_limits(limits=1, user_api="blas"):  # Prevent excessive CPU usage
            self.C_pinv = np.linalg.solve(self.C_p.T @ self.C_p, self.CpT_Cqinv)

        # Odometry
        dx_local = self.C_pinv @ self.dq
        theta_avg = self.x[2] + 0.5 * dx_local[2] * CONTROL_PERIOD
        R = np.array(
            [
                [math.cos(theta_avg), -math.sin(theta_avg), 0.0],
                [math.sin(theta_avg), math.cos(theta_avg), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.dx = R @ dx_local
        self.x += self.dx * CONTROL_PERIOD

    def start_control(self) -> None:
        if self.control_loop_thread is None:
            print("To initiate a new control loop, please create a new instance of Vehicle.")
            return
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self) -> None:
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None

    def control_loop(self) -> None:
        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(
                0,
                os.SCHED_FIFO,
                os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)),
            )
        except PermissionError:
            print("Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf")

        disable_motors = True
        last_command_time = time.time()
        last_step_time = time.time()

        while self.control_loop_running:
            # Maintain the desired control frequency
            while time.time() - last_step_time < CONTROL_PERIOD:
                time.sleep(0.0001)
            curr_time = time.time()
            step_time = curr_time - last_step_time
            last_step_time = curr_time
            if step_time > 0.01:  # 5 ms
                print(f"Warning: Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop")

            # Update state
            self.update_state()
            # Global to local frame conversion
            theta = self.x[2]
            R = np.array(
                [
                    [math.cos(theta), math.sin(theta), 0.0],
                    [-math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            # Check for new command
            if not self.command_queue.empty():
                command = self.command_queue.get()
                last_command_time = time.time()
                target = command["target"]

                # Velocity command
                if command["type"] == CommandType.VELOCITY:
                    if command["frame"] == FrameType.LOCAL:
                        target = R.T @ target
                    self.otg_inp.control_interface = ControlInterface.Velocity
                    self.otg_inp.target_velocity = np.clip(target, -self.max_vel, self.max_vel)

                # Position command
                elif command["type"] == CommandType.POSITION:
                    self.otg_inp.control_interface = ControlInterface.Position
                    self.otg_inp.target_position = target
                    self.otg_inp.target_velocity = np.zeros_like(self.dx)

                self.otg_res = Result.Working
                disable_motors = False
            # Maintain current pose if command stream is disrupted
            if time.time() - last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
                self.otg_inp.target_position = self.otg_out.new_position
                self.otg_inp.target_velocity = np.zeros_like(self.dx)
                self.otg_inp.current_velocity = self.dx  # Set this to prevent lurch when command stream resumes
                self.otg_res = Result.Working
                disable_motors = True

            # Slow down base during caster flip
            # Note: At low speeds, this section can be disabled for smoother movement
            if np.max(np.abs(self.dq[::2])) > 12.56:  # Steer joint speed > 720 deg/s
                if self.otg_inp.control_interface == ControlInterface.Position:
                    self.otg_inp.target_position = self.otg_out.new_position
                elif self.otg_inp.control_interface == ControlInterface.Velocity:
                    self.otg_inp.target_velocity = np.zeros_like(self.dx)

            # Update OTG
            if self.otg_res == Result.Working:
                self.otg_inp.current_position = self.x
                self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
                self.otg_out.pass_to_input(self.otg_inp)

            disable_motors = False
            if disable_motors:
                # Send motor neutral commands
                self.caster_module_controller.set_neutral()

            else:
                # Operational space velocity
                dx_d = self.otg_out.new_velocity

                dx_d_local = R @ dx_d

                # Joint velocities
                dq_d = self.C @ dx_d_local

                vel_dict = {
                    "steer_vel": np.asarray(dq_d[::2], order="C"),
                    "drive_vel": np.asarray(dq_d[1:][::2], order="C"),
                }
                self.caster_module_controller.set_velocities(vel_dict)

    def _enqueue_command(self, command_type: CommandType, target: Any, frame: Optional[FrameType] = None) -> None:
        if self.command_queue.full():
            print("Warning: Command queue is full. Is control loop running?")
        else:
            command = {"type": command_type, "target": target}
            if frame is not None:
                command["frame"] = FrameType(frame)
            self.command_queue.put(command, block=False)

    def set_target_velocity(self, velocity: Any, frame: str = "local") -> None:
        self._enqueue_command(CommandType.VELOCITY, velocity, frame)

    def set_target_position(self, position: Any) -> None:
        self._enqueue_command(CommandType.POSITION, position)


if __name__ == "__main__":
    import os

    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Force headless mode
    import time

    import pygame

    vehicle = Vehicle(max_vel=(1, 1, 1), max_accel=(1, 1, 1))
    vehicle.start_control()
    # Initialize pygame and joystick
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick/gamepad connected!")
        exit()
    else:
        print(f"Detected {pygame.joystick.get_count()} joystick(s).")

    # Initialize the joystick
    joy = pygame.joystick.Joystick(0)
    joy.init()

    print(f"Joystick Name: {joy.get_name()}")
    print(f"Number of Axes: {joy.get_numaxes()}")
    print(f"Number of Buttons: {joy.get_numbuttons()}")

    # Main loop to read joystick inputs

    try:
        while True:
            # for _ in range(100):
            pygame.event.pump()

            # Read inputs
            start = joy.get_button(7)  # Example button
            x = joy.get_axis(1)  # Left stick Y-axis
            y = joy.get_axis(0)  # Left stick X-axis
            th = joy.get_axis(3)  # Right stick X-axis

            user_cmd = np.array([-x, y, -th])
            # if < 0.05 force to zero
            user_cmd[np.abs(user_cmd) < 0.05] = 0
            print(f"user_cmd: {user_cmd}")
            # user_cmd = np.array([0.4,0,0])
            vehicle.set_target_velocity(user_cmd, frame="local")

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Exiting...")
        pygame.quit()
