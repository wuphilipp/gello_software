import time
import importlib
from omegaconf import OmegaConf

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

    # If robot config is a path, load it
    robot_cfg = cfg['robot']
    if isinstance(robot_cfg.get('config'), str):
        robot_cfg['config'] = OmegaConf.to_container(OmegaConf.load(robot_cfg['config']), resolve=True)

    robot = instantiate(robot_cfg)
    agent = instantiate(cfg['agent'])
    hz = cfg.get('hz', 30)
    max_steps = cfg.get('max_steps', 1000)

    print(f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}")
    print(f"Control loop: {hz} Hz, max_steps: {max_steps}")

    for step in range(max_steps):
        obs = robot.get_observations()
        action = agent.act(obs)
        robot.command_joint_state(action)
        time.sleep(1.0 / hz)
        if step % 100 == 0:
            print(f"Step {step}/{max_steps}")

if __name__ == '__main__':
    main()