"""Automated YAM configuration generator for GELLO.

This script guides the user through setting up their YAM arm in the known position
and automatically generates a YAML configuration file with the detected joint offsets.
"""

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro
import yaml

from gello.dynamixel.driver import DynamixelDriver

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class Args:
    output_path: Optional[str] = None
    """Output path for the generated YAML config. If not provided, will use configs/yam_auto_generated.yaml"""

    sim_output_path: Optional[str] = None
    """Output path for the simulation YAML config. If not provided, will use configs/yam_auto_generated_sim.yaml"""

    port: Optional[str] = None
    """The port that GELLO is connected to. If not provided, will auto-detect."""

    start_joints: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
    """The joint angles that the GELLO should be placed in (in radians). Default is YAM known position."""

    joint_signs: Tuple[float, ...] = (1, -1, -1, -1, 1, 1)
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


def get_joint_offsets(
    args: Args, port: str
) -> Tuple[list, Optional[Tuple[float, float]]]:
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
        gripper_open = gripper_current
        gripper_close = gripper_current - 59.5
        gripper_config = (gripper_open, gripper_close)

    return best_offsets, gripper_config


class FlowStyleList(list):
    """Custom list class to force flow style in YAML output."""


def flow_style_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def update_config_with_offsets(
    template_config: dict,
    port: str,
    joint_offsets: list,
    gripper_config: Optional[Tuple[float, float]],
) -> dict:
    """Update a template config with detected offsets and port."""
    import copy

    def to_flow_list(data, preserve_ints=False):
        """Convert data to FlowStyleList with proper type conversion."""
        if preserve_ints:
            return FlowStyleList(
                [
                    (
                        int(x)
                        if isinstance(x, (int, float, np.number)) and x == int(x)
                        else float(x) if isinstance(x, (int, float, np.number)) else x
                    )
                    for x in data
                ]
            )
        return FlowStyleList(
            [float(x) if isinstance(x, (int, float, np.number)) else x for x in data]
        )

    config = copy.deepcopy(template_config)
    dynamixel_config = config["agent"]["dynamixel_config"]

    # Update basic config
    config["agent"]["port"] = port

    # Update offsets and convert to flow style
    dynamixel_config["joint_offsets"] = to_flow_list(
        [round(offset, 5) for offset in joint_offsets]
    )

    # Update gripper config if present
    if gripper_config and "gripper_config" in dynamixel_config:
        gripper_vals = [7, gripper_config[1], gripper_config[0]]
        dynamixel_config["gripper_config"] = to_flow_list(
            gripper_vals, preserve_ints=True
        )

    # Convert existing lists to flow style
    for key in ["joint_ids", "joint_signs"]:
        preserve_ints = key == "joint_ids"
        dynamixel_config[key] = to_flow_list(
            dynamixel_config[key], preserve_ints=preserve_ints
        )

    # Set gripper start position to open (1.0) if gripper exists
    start_joints = list(config["agent"]["start_joints"])
    if len(start_joints) == 7:
        start_joints[6] = 1.0
    config["agent"]["start_joints"] = to_flow_list(start_joints)

    return config


def main(args: Args) -> None:
    print("=== GELLO YAM Configuration Generator ===\n")

    # Step 1: Port detection
    if args.port is None:
        print("Detecting GELLO port...")
        port = find_gello_port()
        if port is None:
            print("Please ensure your GELLO device is connected and try again.")
            sys.exit(1)
        print(f"Found GELLO at: {port}\n")
    else:
        port = args.port
        print(f"Using specified port: {port}\n")

    # Step 2: Physical setup instructions
    print("SETUP INSTRUCTIONS:")
    print(
        "1. Position your GELLO in the position shown in the build documentation image."
    )
    print("2. Ensure all Dynamixel motors are powered and connected")
    print("3. Make sure the gripper is attached (if using)")
    print()

    input("Press Enter when your GELLO is in the correct position...")
    print()

    # Step 3: Detect offsets
    print("Detecting joint offsets...")
    try:
        joint_offsets, gripper_config = get_joint_offsets(args, port)
        print("Joint offsets detected successfully!")
        print(f"   Offsets: {[f'{x:.3f}' for x in joint_offsets]}")
        if gripper_config:
            print(
                f"   Gripper: open={gripper_config[0]:.1f}°, close={gripper_config[1]:.1f}°"
            )
        print()
    except Exception as e:
        print(f"Error detecting offsets: {e}")
        print("Please check your connection and try again.")
        sys.exit(1)

    # Step 4: Load template configs and update with offsets
    print("Loading template configurations and updating with detected offsets...")

    config_dir = Path(__file__).parent.parent / "configs"
    hardware_template_path = config_dir / "templates/yam_template.yaml"
    sim_template_path = config_dir / "templates/yam_sim_template.yaml"

    try:
        # Load hardware template
        with open(hardware_template_path, "r") as f:
            hardware_template = yaml.safe_load(f)

        # Load simulation template
        with open(sim_template_path, "r") as f:
            sim_template = yaml.safe_load(f)

        # Update configs with detected offsets
        hardware_config = update_config_with_offsets(
            hardware_template, port, joint_offsets, gripper_config
        )
        sim_config = update_config_with_offsets(
            sim_template, port, joint_offsets, gripper_config
        )

    except FileNotFoundError as e:
        print(f"Error: Template config file not found: {e}")
        print(
            "Please ensure yam_template.yaml and yam_sim_template.yaml exist in the configs/templates/ directory."
        )
        sys.exit(1)

    # Step 5: Save updated configs
    if args.output_path is None:
        hardware_output_path = config_dir / "yam_auto_generated.yaml"
    else:
        hardware_output_path = Path(args.output_path)

    if args.sim_output_path is None:
        sim_output_path = config_dir / "yam_auto_generated_sim.yaml"
    else:
        sim_output_path = Path(args.sim_output_path)

    # Register custom representer for flow style lists
    yaml.add_representer(FlowStyleList, flow_style_representer)

    # Save hardware config
    hardware_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hardware_output_path, "w") as f:
        yaml.dump(
            hardware_config,
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
            width=1000,
        )

    # Save simulation config
    sim_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sim_output_path, "w") as f:
        yaml.dump(
            sim_config,
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
            width=1000,
        )

    print(f"Hardware configuration saved to: {hardware_output_path}")
    print(f"Simulation configuration saved to: {sim_output_path}")
    print()
    print("You can now run GELLO with:")
    print(
        f"   Hardware: python experiments/launch_yaml.py --left-config-path {hardware_output_path}"
    )
    print(
        f"   Simulation: python experiments/launch_yaml.py --left-config-path {sim_output_path}"
    )
    print()
    print("Configuration files generated successfully!")


if __name__ == "__main__":
    main(tyro.cli(Args))
