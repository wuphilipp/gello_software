import time
import importlib
import sys
from omegaconf import OmegaConf

def instantiate(cfg):
    """Instantiate objects from config using _target_ field."""
    if isinstance(cfg, dict) and '_target_' in cfg:
        try:
            module_path, class_name = cfg['_target_'].rsplit('.', 1)
            cls = getattr(importlib.import_module(module_path), class_name)
            kwargs = {k: v for k, v in cfg.items() if k != '_target_'}
            # Recursively instantiate nested objects
            instantiated_kwargs = {k: instantiate(v) for k, v in kwargs.items()}
            return cls(**instantiated_kwargs)
        except Exception as e:
            print(f"Error instantiating {cfg['_target_']}: {e}")
            raise
    elif isinstance(cfg, dict):
        return {k: instantiate(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [instantiate(v) for v in cfg]
    else:
        return cfg

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple YAML-based launcher for gello_software")
    parser.add_argument('--config-path', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    try:
        # Load and resolve config
        print(f"Loading config from: {args.config_path}")
        cfg = OmegaConf.load(args.config_path)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        
        print("Config loaded successfully:")
        print(f"  Robot: {cfg.get('robot', {}).get('_target_', 'Not specified')}")
        print(f"  Agent: {cfg.get('agent', {}).get('_target_', 'Not specified')}")
        print(f"  Hz: {cfg.get('hz', 30)}")
        print(f"  Max steps: {cfg.get('max_steps', 1000)}")
        
        # Instantiate robot and agent
        print("\nInstantiating robot...")
        robot = instantiate(cfg['robot'])
        print("Robot instantiated successfully!")
        
        print("Instantiating agent...")
        agent = instantiate(cfg['agent'])
        print("Agent instantiated successfully!")
        
        # Get configuration
        hz = cfg.get('hz', 30)
        max_steps = cfg.get('max_steps', 1000)
        
        print(f"\nStarting control loop at {hz} Hz for {max_steps} steps...")
        print("Press Ctrl+C to stop")
        
        # Main control loop
        for step in range(max_steps):
            try:
                obs = robot.get_observations()
                action = agent.act(obs)
                robot.command_joint_state(action)
                time.sleep(1.0 / hz)
                
                if step % 100 == 0:
                    print(f"Step {step}/{max_steps}")
                    
            except KeyboardInterrupt:
                print("\nStopping control loop...")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                break
                
    except KeyError as e:
        print(f"Missing required key in config: {e}")
        print("Make sure your YAML has 'robot' and 'agent' sections")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()