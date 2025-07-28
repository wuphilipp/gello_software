#!/usr/bin/env python3
"""Simple YAM GELLO launcher with embedded robot class using minimal config."""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

# Add the parent directory to the path so we can import gello modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent
from gello.robots.robot import Robot


class YAMSimRobot(Robot):
    """A YAM simulation robot that reads from simple config."""

    def __init__(self, config: dict):
        """Initialize from simple config dict."""
        self.port = config["port"]
        self.joint_offsets = np.array(config["joint_offsets"])
        self.joint_signs = np.array(config["joint_signs"])
        self.gripper_config = config["gripper_config"]
        self.start_joints = np.array(config["start_joints"])

        # Initialize state
        self._joint_state = np.zeros(7)  # 6 arm joints + 1 gripper
        self._joint_velocities = np.zeros(7)
        self._gripper_state = 1.0  # open

        print(f"YAM Sim Robot initialized:")
        print(f"  Port: {self.port}")
        print(f"  Joint offsets: {self.joint_offsets}")
        print(f"  Joint signs: {self.joint_signs}")
        print(f"  Start joints: {self.start_joints}")

    def num_dofs(self) -> int:
        return 7  # 6 arm joints + 1 gripper

    def get_joint_state(self) -> np.ndarray:
        # In sim, just return the current commanded state
        return self._joint_state.copy()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        # Apply joint offsets and signs to transform from GELLO space to robot space
        robot_joints = np.zeros_like(joint_state)
        for i in range(6):  # arm joints only
            robot_joints[i] = self.joint_signs[i] * (
                joint_state[i] - self.joint_offsets[i]
            )
        robot_joints[6] = joint_state[6]  # gripper passes through

        dt = 0.01
        self._joint_velocities = (robot_joints - self._joint_state) / dt
        self._joint_state = robot_joints

        print(f"Commanded joints: {robot_joints}")

    def get_observations(self) -> Dict[str, np.ndarray]:
        ee_pos_quat = np.zeros(7)
        return {
            "joint_positions": self._joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array([self._gripper_state]),
        }


def create_gello_agent_from_simple_config(config_path: str) -> GelloAgent:
    """Create a GelloAgent from the simple config file."""

    # Load simple config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = config_data["config"]

    # Create DynamixelRobotConfig from simple config
    dynamixel_config = DynamixelRobotConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],  # 6 arm joints
        joint_offsets=config["joint_offsets"],
        joint_signs=config["joint_signs"],
        gripper_config=config["gripper_config"],
    )

    return GelloAgent(
        port=config["port"],
        dynamixel_config=dynamixel_config,
        start_joints=config["start_joints"],
    )


def run_full_gello_session(sim: bool = False):
    """Run a full GELLO session like launch_yaml.py."""
    import threading
    import time

    from gello.env import RobotEnv
    from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

    config_path = Path(__file__).parent / "simple_config.yaml"

    # Load config
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = config_data["config"]

    # Create robot based on sim flag
    if sim:
        print("Creating sim robot...")
        robot = YAMSimRobot(config=config)
    else:
        print("Creating hardware robot...")
        from gello.robots.yam import YAMRobot

        robot = YAMRobot(channel=config.get("channel", "can_left"))

    # Create GELLO agent
    gello_agent = create_gello_agent_from_simple_config(str(config_path))

    # Handle robot server setup (like launch_yaml.py)
    if hasattr(robot, "serve"):  # MujocoRobotServer
        print("Starting robot server...")
        server_thread = threading.Thread(target=robot.serve, daemon=True)
        server_thread.start()
        time.sleep(2)

        robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")
    else:  # Direct robot (hardware or sim)
        print("Creating ZMQ server for robot...")
        server = ZMQServerRobot(robot, port=6001, host="127.0.0.1")
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")

    # Create environment
    env = RobotEnv(robot_client, control_rate_hz=30)

    # Move to start position
    if config["start_joints"]:
        reset_joints = np.array(config["start_joints"])
        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)

            print(f"Moving robot to start position: {reset_joints}")
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {gello_agent.__class__.__name__}"
    )
    print("Control loop: 30 Hz")

    # Initial position check
    print("Going to start position")
    start_pos = gello_agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    max_joint_delta = 1.0
    if abs_deltas.max() > max_joint_delta:
        print("Warning: Large joint delta detected!")
        for i, delta in enumerate(abs_deltas):
            if delta > max_joint_delta:
                print(f"joint[{i}]: delta: {delta:4.3f}")
        return

    # Control loop
    print("Starting control loop... Press Ctrl+C to stop")
    try:
        max_delta = 1.0
        for _ in range(25):  # Smooth startup
            obs = env.get_obs()
            command_joints = gello_agent.act(obs)
            current_joints = obs["joint_positions"]
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            env.step(current_joints + delta)

        while True:
            obs = env.get_obs()
            action = gello_agent.act(obs)
            env.step(action)

    except KeyboardInterrupt:
        print("\nStopped by user")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YAM GELLO Simple Launcher")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    parser.add_argument(
        "--test", action="store_true", help="Run test mode instead of full session"
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / "simple_config.yaml"

    print("=== YAM GELLO Simple Launcher ===")
    print(f"Mode: {'Simulation' if args.sim else 'Hardware'}")
    print(f"Config: {config_path}")

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        if args.test:
            # Test mode - just verify components work
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            print("\nTesting components...")

            # Test sim robot
            if args.sim:
                print("Testing YAM Sim Robot:")
                sim_robot = YAMSimRobot(config=config_data["config"])
                test_joints = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, 0.5])
                sim_robot.command_joint_state(test_joints)
                obs = sim_robot.get_observations()
                print(f"Sim robot OK: {obs['joint_positions'][:3]}...")

            # Test GELLO agent
            print("Testing GELLO Agent:")
            gello_agent = create_gello_agent_from_simple_config(str(config_path))
            print(f"GELLO Agent OK: {gello_agent.port}")

            print("Test completed successfully!")
        else:
            # Full session mode
            run_full_gello_session(sim=args.sim)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
