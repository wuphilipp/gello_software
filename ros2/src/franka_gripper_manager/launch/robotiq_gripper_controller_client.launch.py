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
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_robot_nodes(context):
    config_file = LaunchConfiguration('robotiq_config_file').perform(context)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config['namespace']
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare('franka_gripper_manager'), 'launch', 'robotiq.launch.py'
                    ])
                ),
                launch_arguments={
                    'namespace': namespace ,
                    'com_port': str(config['com_port']),
                }.items(),
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robotiq_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('franka_gripper_manager'), 'config', 'robotiq-config.yaml'
            ]),
            description='Path to the robot configuration file to load',
        ),
        OpaqueFunction(function=generate_robot_nodes),
    ])