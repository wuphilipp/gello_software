from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = []
    args.append(
        DeclareLaunchArgument(
            name="com_port", default_value=None, description="Default COM port"
        )
    )
    nodes = [
        Node(
            package="franka_gello_state_publisher",
            executable="gello_publisher",
            name="gello_publisher",
            output="screen",
            parameters=[{"com_port": LaunchConfiguration("com_port")}],
        ),
    ]

    return LaunchDescription(args + nodes)
