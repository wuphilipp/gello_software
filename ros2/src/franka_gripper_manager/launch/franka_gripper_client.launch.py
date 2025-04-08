from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="franka_gripper_manager",
                executable="franka_gripper_client",
                arguments=["franka_gripper_client"],
                output="screen",
            ),
        ]
    )
