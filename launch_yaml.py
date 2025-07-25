import time
import importlib
import threading
from omegaconf import OmegaConf
import numpy as np

def instantiate(cfg):
    if isinstance(cfg, dict) and '_target_' in cfg:
        module_path, class_name = cfg['_target_'].rsplit('.', 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        kwargs = {k: v for k, v in cfg.items() if k != '_target_'}
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
    parser.add_argument('--config-path', type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)

    robot_cfg = cfg['robot']
    if isinstance(robot_cfg.get('config'), str):
        robot_cfg['config'] = OmegaConf.to_container(OmegaConf.load(robot_cfg['config']), resolve=True)

    robot = instantiate(robot_cfg)
    
    # Handle different robot types
    if hasattr(robot, 'serve'):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.zmq_core.robot_node import ZMQClientRobot
        from gello.env import RobotEnv
        
        # Start server in background
        server_thread = threading.Thread(target=robot.serve, daemon=True)
        server_thread.start()
        time.sleep(2)  # Give server time to start
        
        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(
            port=robot_cfg.get('port', 5556),
            host=robot_cfg.get('host', '127.0.0.1'),
        )
    else:  # Direct robot (hardware)
        from gello.zmq_core.robot_node import ZMQServerRobot, ZMQClientRobot
        from gello.env import RobotEnv
        
        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=6001, host="127.0.0.1")
        server_thread = threading.Thread(target=server.serve, daemon=True)
        server_thread.start()
        time.sleep(1)
        
        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")

    env = RobotEnv(robot_client, control_rate_hz=cfg.get('hz', 30))
    agent = instantiate(cfg['agent'])
    
    # Move robot to start_joints position if specified in config
    if 'start_joints' in cfg['agent'] and cfg['agent']['start_joints'] is not None:
        reset_joints = np.array(cfg['agent']['start_joints'])
        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)
            
            print(f"Moving robot to start position: {reset_joints}")
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)
    
    print(f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}")
    print(f"Control loop: {cfg.get('hz', 30)} Hz, max_steps: {cfg.get('max_steps', 1000)}")

    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
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

    max_delta = 0.05
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
    
        
if __name__ == '__main__':
    main()
