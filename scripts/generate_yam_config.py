#!/usr/bin/env python3
"""
Automated YAM configuration generator for GELLO.

This script guides the user through setting up their YAM arm in the known position
and automatically generates a YAML configuration file with the detected joint offsets.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import glob
import time

import numpy as np
import tyro
import yaml

from gello.dynamixel.driver import DynamixelDriver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class Args:
    output_path: Optional[str] = None
    """Output path for the generated YAML config. If not provided, will use configs/yam_auto_generated.yaml"""
    
    port: Optional[str] = None
    """The port that GELLO is connected to. If not provided, will auto-detect."""
    
    start_joints: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
    """The joint angles that the GELLO should be placed in (in radians). Default is YAM known position."""
    
    joint_signs: Tuple[float, ...] = (1, 1, -1, -1, 1, 1)
    """The joint signs for YAM arm."""
    
    gripper: bool = True
    """Whether or not the gripper is attached."""
    
    channel: str = "can_left"
    """CAN channel for YAM robot communication."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints


def find_gello_port() -> Optional[str]:
    """Auto-detect GELLO port by looking for FTDI USB-Serial converters."""
    possible_ports = glob.glob("/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_*")
    
    if not possible_ports:
        return None
    elif len(possible_ports) == 1:
        return possible_ports[0]
    else:
        print("Multiple FTDI ports found:")
        for i, port in enumerate(possible_ports):
            print(f"  {i+1}: {port}")
        
        while True:
            try:
                choice = int(input("Select port number: ")) - 1
                if 0 <= choice < len(possible_ports):
                    return possible_ports[choice]
                else:
                    print("Invalid choice, please try again.")
            except ValueError:
                print("Please enter a valid number.")


def get_joint_offsets(args: Args, port: str) -> Tuple[list, Optional[Tuple[float, float]]]:
    """Get joint offsets using the same logic as gello_get_offset.py."""
    joint_ids = list(range(1, args.num_joints + 1))
    driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = args.joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = args.start_joints[index]
        return np.abs(joint_i - start_i)

    # Warmup
    for _ in range(10):
        driver.get_joints()

    best_offsets = []
    curr_joints = driver.get_joints()
    
    for i in range(args.num_robot_joints):
        best_offset = 0
        best_error = 1e6
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 8 * 4 + 1):
            error = get_error(offset, i, curr_joints)
            if error < best_error:
                best_error = error
                best_offset = offset
        best_offsets.append(best_offset)

    gripper_config = None
    if args.gripper:
        gripper_current = np.rad2deg(driver.get_joints()[-1])
        gripper_open = gripper_current - 0.2
        gripper_close = gripper_current - 42
        gripper_config = (gripper_open, gripper_close)

    return best_offsets, gripper_config


def generate_yaml_config(args: Args, port: str, joint_offsets: list, gripper_config: Optional[Tuple[float, float]]) -> dict:
    """Generate the YAML configuration dictionary."""
    config = {
        'robot': {
            '_target_': 'gello.robots.yam.YAMRobot',
            'channel': args.channel
        },
        'agent': {
            '_target_': 'gello.agents.gello_agent.GelloAgent',
            'port': port,
            'dynamixel_config': {
                '_target_': 'gello.agents.gello_agent.DynamixelRobotConfig',
                'joint_ids': list(range(1, args.num_joints + 1)),
                'joint_offsets': [round(offset, 5) for offset in joint_offsets],
                'joint_signs': list(args.joint_signs),
                'gripper_config': [7, gripper_config[1], gripper_config[0]] if gripper_config else None
            },
            'start_joints': list(args.start_joints) + ([1.0] if args.gripper else [])
        },
        'hz': 30,
        'max_steps': 1000
    }
    
    if not args.gripper:
        del config['agent']['dynamixel_config']['gripper_config']
    
    return config


def main(args: Args) -> None:
    print("=== GELLO YAM Configuration Generator ===\n")
    
    # Step 1: Port detection
    if args.port is None:
        print("Detecting GELLO port...")
        port = find_gello_port()
        if port is None:
            print("❌ No FTDI USB-Serial converter found!")
            print("Please ensure your GELLO device is connected and try again.")
            sys.exit(1)
        print(f"✅ Found GELLO at: {port}\n")
    else:
        port = args.port
        print(f"Using specified port: {port}\n")
    
    # Step 2: Physical setup instructions
    print("📋 SETUP INSTRUCTIONS:")
    print("1. Position your YAM arm in the known configuration:")
    print("   - All joints at 0 degrees (straight up position)")
    print("   - Match the position shown in the documentation image")
    print("2. Ensure all Dynamixel motors are powered and connected")
    print("3. Make sure the gripper is attached (if using)")
    print()
    
    input("Press Enter when your YAM arm is in the correct position...")
    print()
    
    # Step 3: Detect offsets
    print("🔍 Detecting joint offsets...")
    try:
        joint_offsets, gripper_config = get_joint_offsets(args, port)
        print("✅ Joint offsets detected successfully!")
        print(f"   Offsets: {[f'{x:.3f}' for x in joint_offsets]}")
        if gripper_config:
            print(f"   Gripper: open={gripper_config[0]:.1f}°, close={gripper_config[1]:.1f}°")
        print()
    except Exception as e:
        print(f"❌ Error detecting offsets: {e}")
        print("Please check your connection and try again.")
        sys.exit(1)
    
    # Step 4: Generate config
    print("⚙️  Generating YAML configuration...")
    config = generate_yaml_config(args, port, joint_offsets, gripper_config)
    
    # Step 5: Save config
    if args.output_path is None:
        output_path = Path(__file__).parent.parent / "configs" / "yam_auto_generated.yaml"
    else:
        output_path = Path(args.output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration saved to: {output_path}")
    print()
    print("🚀 You can now run GELLO with:")
    print(f"   python launch_yaml.py --config-path {output_path}")
    print()
    print("Configuration generated successfully! 🎉")


if __name__ == "__main__":
    main(tyro.cli(Args))