import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def generate_robot_nodes(context):
    config_file_name = LaunchConfiguration("config_file").perform(context)
    package_config_dir = FindPackageShare("franka_gello_state_publisher").perform(context)
    config_file = os.path.join(package_config_dir, "config", config_file_name)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config["namespace"]
        nodes.append(
            Node(
                package="franka_gello_state_publisher",
                executable="gello_publisher",
                name="gello_publisher",
                namespace=namespace,
                output="screen",
                parameters=[
                    {"com_port": "/dev/serial/by-id/" + config["com_port"]},
                    {"gello_name": item_name},
                    {"num_joints": config["num_joints"]},
                    {"joint_signs": config["joint_signs"]},
                    {"gripper": config["gripper"]},
                    {"gripper_range_rad": config["gripper_range_rad"]},
                    {"best_offsets": config["best_offsets"]},
                ],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="example_fr3_config.yaml",
                description="Name of the gello configuration file to load",
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
