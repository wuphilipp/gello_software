"""Shared utilities for robot control loops."""

import datetime
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from gello.env import RobotEnv
from gello.agents.agent import Agent


def move_to_start_position(env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25) -> bool:
    """Move robot to start position gradually.
    
    Args:
        env: Robot environment
        agent: Agent that provides target position
        max_delta: Maximum joint delta per step
        steps: Number of steps for gradual movement
        
    Returns:
        bool: True if successful, False if position too far
    """
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
        return False

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)
        
    return True


class SaveInterface:
    """Handles keyboard-based data saving interface."""
    
    def __init__(self, data_dir: str = "data", agent_name: str = "Agent", expand_user: bool = False):
        """Initialize save interface.
        
        Args:
            data_dir: Base directory for saving data
            agent_name: Name of agent (used for subdirectory)
            expand_user: Whether to expand ~ in data_dir path
        """
        from gello.data_utils.keyboard_interface import KBReset
        
        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None
        
        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording")
        print("  Q: Stop recording")
    
    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Update save interface and handle saving.
        
        Args:
            obs: Current observations
            action: Current action
            
        Returns:
            Optional[str]: "quit" if user wants to exit, None otherwise
        """
        from gello.data_utils.format_obs import save_frame
        
        dt = datetime.datetime.now()
        state = self.kb_interface.update()
        
        if state == "start":
            dt_time = datetime.datetime.now()
            self.save_path = self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {self.save_path}")
        elif state == "save":
            if self.save_path is not None:
                save_frame(self.save_path, dt, obs, action)
        elif state == "normal":
            self.save_path = None
        elif state == "quit":
            print("\nExiting.")
            return "quit"
        else:
            raise ValueError(f"Invalid state {state}")
            
        return None


def run_control_loop(
    env: RobotEnv, 
    agent: Agent, 
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False
) -> None:
    """Run the main control loop.
    
    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    if use_colors:
        try:
            from termcolor import colored
            print_func = lambda msg, **kwargs: print(colored(msg, **kwargs), end="", flush=True)
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            print_func = lambda msg, **kwargs: print(msg, end="", flush=True)
            start_msg = "\nStart 🚀🚀🚀"
    else:
        print_func = lambda msg, **kwargs: print(msg, end="", flush=True)
        start_msg = "\nStart 🚀🚀🚀"
    
    print(start_msg)
    
    start_time = time.time()
    obs = env.get_obs()
    
    while True:
        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            if use_colors:
                print_func(message, color="white", attrs=["bold"])
            else:
                print_func(message)
        
        action = agent.act(obs)
        
        # Handle save interface
        if save_interface is not None:
            result = save_interface.update(obs, action)
            if result == "quit":
                break
        
        obs = env.step(action) 