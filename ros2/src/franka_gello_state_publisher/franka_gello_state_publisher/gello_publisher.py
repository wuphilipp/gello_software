import os
import sys
import glob
from typing import Tuple
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import yaml
from ament_index_python.packages import get_package_share_directory, get_package_prefix


class GelloPublisher(Node):
    def __init__(self):
        super().__init__("gello_publisher")

        default_com_port = self.determine_default_com_port()
        self.declare_parameter("com_port", default_com_port)
        self.com_port = self.get_parameter("com_port").get_parameter_value().string_value
        self.port = self.com_port.split("/")[-1]
        """The port that GELLO is connected to."""

        config_path = os.path.join(
            get_package_share_directory("franka_gello_state_publisher"),
            "config",
            "gello_config.yaml",
        )
        self.get_values_from_config(config_path)

        self.robot_joint_publisher = self.create_publisher(JointState, "/gello/joint_states", 10)
        self.gripper_joint_publisher = self.create_publisher(
            Float32, "/gripper_client/target_gripper_width_percent", 10
        )

        self.timer = self.create_timer(1 / 25, self.publish_joint_jog)

        self.joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]

    def determine_default_com_port(self) -> str:
        matches = glob.glob("/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter*")
        if matches:
            self.get_logger().info(f"Auto-detected com_ports: {matches}")
            return matches[0]
        else:
            self.get_logger().warn("No com_ports detected. Please specify the com_port manually.")
            return "INVALID_COM_PORT"

    def get_values_from_config(self, config_file: str):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.num_robot_joints: int = config[self.port]["num_joints"]
        """The number of joints in the robot."""

        self.joint_signs: Tuple[float, ...] = config[self.port]["joint_signs"]
        """Depending on how the motor is mounted on the Gello, its rotation direction can be reversed."""

        self.gripper: bool = config[self.port]["gripper"]
        """Whether or not the gripper is attached."""

        joint_ids = list(range(1, self.num_joints + 1))
        self.add_dynamixel_driver_path()
        from gello.dynamixel.driver import DynamixelDriver

        self.driver = DynamixelDriver(joint_ids, port=self.com_port, baudrate=57600)
        """The driver for the Dynamixel motors."""

        self.best_offsets = np.array(config[self.port]["best_offsets"])
        """The best offsets for the joints."""

        self.gripper_range_rad: Tuple[float, float] = config[self.port]["gripper_range_rad"]
        """The range of the gripper in radians."""

        self.__post_init__()

    def __post_init__(self):
        assert len(self.joint_signs) == self.num_robot_joints
        for idx, j in enumerate(self.joint_signs):
            assert j == -1 or j == 1, f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints

    def publish_joint_jog(self):
        current_joints = self.driver.get_joints()
        current_robot_joints = current_joints[: self.num_robot_joints]
        current_joints_corrected = (current_robot_joints - self.best_offsets) * self.joint_signs

        robot_joint_states = JointState()
        robot_joint_states.header.stamp = self.get_clock().now().to_msg()
        robot_joint_states.name = self.joint_names
        robot_joint_states.header.frame_id = "fr3_link0"
        robot_joint_states.position = [float(joint) for joint in current_joints_corrected]

        gripper_joint_states = Float32()
        if self.gripper:
            gripper_position = current_joints[-1]
            gripper_joint_states.data = self.gripper_readout_to_percent(gripper_position)
        else:
            gripper_joint_states.position = 0.0
        self.robot_joint_publisher.publish(robot_joint_states)
        self.gripper_joint_publisher.publish(gripper_joint_states)

    def gripper_readout_to_percent(self, gripper_position: float) -> float:
        gripper_percent = (gripper_position - self.gripper_range_rad[0]) / (
            self.gripper_range_rad[1] - self.gripper_range_rad[0]
        )
        return max(0.0, min(1.0, gripper_percent))

    def add_dynamixel_driver_path(self):
        gello_path = os.path.abspath(
            os.path.join(get_package_prefix("franka_gello_state_publisher"), "../../../")
        )
        sys.path.insert(0, gello_path)


def main(args=None):
    rclpy.init(args=args)
    gello_publisher = GelloPublisher()
    rclpy.spin(gello_publisher)
    gello_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
