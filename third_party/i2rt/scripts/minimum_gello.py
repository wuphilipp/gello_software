import time
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import portal
import tyro

from i2rt.robots.motor_chain_robot import ARX_XML_PATH, YAM_XML_PATH, get_arx_robot, get_yam_robot
from i2rt.robots.robot import Robot

DEFAULT_ROBOT_PORT = 11333


class ServerRobot:
    """A simple server for a leader robot."""

    def __init__(self, robot: Robot, port: str):
        self._robot = robot
        self._server = portal.Server(DEFAULT_ROBOT_PORT)
        print(f"Robot Sever Binding to {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Serve the leader robot."""
        self._server.start()


class ClientRobot(Robot):
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_pos (T): The state to command the leader robot to.
        """
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        return self._client.get_observations().result()


@dataclass
class Args:
    robot: Literal["yam", "arx"] = "yam"
    mode: Literal["follower", "leader", "visualizer"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"


def main(args: Args) -> None:
    if args.robot == "yam":
        xml_path = YAM_XML_PATH
        if args.mode != "visualizer":
            robot = get_yam_robot(channel=args.can_channel)
    elif args.robot == "arx":
        xml_path = ARX_XML_PATH
        if args.mode != "visualizer":
            robot = get_arx_robot(channel=args.can_channel)
    else:
        raise ValueError(f"Invalid robot: {args.robot}")

    if args.mode == "follower":
        server_robot = ServerRobot(robot, DEFAULT_ROBOT_PORT)
        server_robot.serve()
    elif args.mode == "leader":
        client_robot = ClientRobot(DEFAULT_ROBOT_PORT, host=args.server_host)

        # sync the robot state
        current_joint_pos = robot.get_joint_pos()
        current_follower_joint_pos = client_robot.get_joint_pos()
        print(f"Current leader joint pos: {current_joint_pos}")
        print(f"Current follower joint pos: {current_follower_joint_pos}")
        for i in range(100):
            current_joint_pos = robot.get_joint_pos()
            follower_command_joint_pos = current_joint_pos * i / 100 + current_follower_joint_pos * (1 - i / 100)
            follower_command_joint_pos[-1] = 0.0
            client_robot.command_joint_pos(follower_command_joint_pos)
            time.sleep(0.03)

        while True:
            current_joint_pos = robot.get_joint_pos()
            current_joint_pos[-1] = 0.0
            client_robot.command_joint_pos(current_joint_pos)
            time.sleep(0.03)
    elif args.mode == "visualizer":
        import mujoco

        client_robot = ClientRobot(DEFAULT_ROBOT_PORT, host=args.server_host)
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        dt: float = 0.01
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()
                joint_pos = client_robot.get_joint_pos()
                data.qpos[:] = joint_pos[: model.nq]

                # sync the model state
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main(tyro.cli(Args))
