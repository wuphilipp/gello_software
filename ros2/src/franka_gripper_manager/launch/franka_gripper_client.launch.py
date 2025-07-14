import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def generate_nodes(context):
    config_file = LaunchConfiguration("gripper_manager_config_file").perform(context)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        nodes.append(
            Node(
                package="franka_gripper_manager",
                executable="franka_gripper_client",
                name="franka_gripper_client",
                namespace=str(config["namespace"]),
                output="screen",
                parameters=[{"namespace": str(config["namespace"])}],
            )
        )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gripper_manager_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("franka_gripper_manager"),
                        "config",
                        "franka_gripper_config.yaml",
                    ]
                ),
                description="Path to the franka_gripper_manager configuration file to load",
            ),
            OpaqueFunction(function=generate_nodes),
        ]
    )
