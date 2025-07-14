import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def generate_robot_nodes(context):
    config_file = LaunchConfiguration("gello_config_file").perform(context)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config["namespace"]
        com_port = config["com_port"]
        gello_name = item_name
        nodes.append(
            Node(
                package="franka_gello_state_publisher",
                executable="gello_publisher",
                name="gello_publisher",
                namespace=namespace,
                output="screen",
                parameters=[
                    {"com_port": "/dev/serial/by-id/" + com_port},
                    {"gello_name": gello_name},
                ],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gello_config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("franka_gello_state_publisher"),
                        "config",
                        "gello_config.yaml",
                    ]
                ),
                description="Path to the robot configuration file to load",
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
