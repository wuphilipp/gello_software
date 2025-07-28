import importlib
import threading
import time

import numpy as np
from omegaconf import OmegaConf


def instantiate(cfg):
    if isinstance(cfg, dict) and "_target_" in cfg:
        module_path, class_name = cfg["_target_"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        return cls(**{k: instantiate(v) for k, v in kwargs.items()})
    elif isinstance(cfg, dict):
        return {k: instantiate(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [instantiate(v) for v in cfg]
    else:
        return cfg


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--left-config-path", type=str, required=True)
    parser.add_argument("--right-config-path", type=str, required=False)
    args = parser.parse_args()

    bimanual = args.right_config_path is not None

    # Load configurations
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate(left_cfg["agent"]),
            agent_right=instantiate(right_cfg["agent"]),
        )
    else:
        agent = instantiate(left_cfg["agent"])

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Start server in background
        server_thread = threading.Thread(target=robot.serve, daemon=True)
        server_thread.start()
        time.sleep(2)  # Give server time to start

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(
            port=cfg["robot"].get("port", 5556),
            host=cfg["robot"].get("host", "127.0.0.1"),
        )
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=6001, host="127.0.0.1")
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30))

    # Move robot to start_joints position if specified in config
    if bimanual:
        # For bimanual, get start_joints from both arms and concatenate
        left_start = left_cfg["agent"].get("start_joints")
        right_start = right_cfg["agent"].get("start_joints")
        if left_start is not None and right_start is not None:
            reset_joints = np.concatenate([np.array(left_start), np.array(right_start)])
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                print(f"Moving robot to start position: {reset_joints}")
                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
    else:
        # Single arm handling
        if "start_joints" in cfg["agent"] and cfg["agent"]["start_joints"] is not None:
            reset_joints = np.array(cfg["agent"]["start_joints"])
            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                print(f"Moving robot to start position: {reset_joints}")
                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(
        f"Control loop: {cfg.get('hz', 30)} Hz, max_steps: {cfg.get('max_steps', 1000)}"
    )

    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 1.0
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 1.0
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    while True:
        obs = env.get_obs()
        action = agent.act(obs)
        env.step(action)


if __name__ == "__main__":
    main()
