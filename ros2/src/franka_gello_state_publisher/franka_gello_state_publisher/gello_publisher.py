import rclpy
from rclpy.executors import ExternalShutdownException
from glob import glob
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from rcl_interfaces.msg import ParameterEvent
from rclpy.parameter import parameter_value_to_python
from franka_gello_state_publisher.gello_hardware import GelloHardware, GelloHardwareParams
from franka_gello_state_publisher.gello_parameter_config import (
    ParameterConfig,
    GelloParameterConfig,
)


class GelloPublisher(Node):
    """ROS2 node for publishing GELLO device joint states and handling parameter updates."""

    def __init__(self) -> None:
        super().__init__("gello_publisher")
        self.PUBLISHING_RATE = 25  # Hz

        hardware_params: GelloHardwareParams = self._setup_hardware_parameters()

        self.gello_hardware = GelloHardware(hardware_params)

        self.arm_joint_publisher = self.create_publisher(JointState, "gello/joint_states", 10)
        self.gripper_joint_publisher = self.create_publisher(
            Float32, "gripper/gripper_client/target_gripper_width_percent", 10
        )

        # Subscribe to parameter events to allow dynamic updates of parameters
        self.parameter_subscription = self.create_subscription(
            ParameterEvent, "/parameter_events", self.parameter_event_callback, 10
        )

        self.timer = self.create_timer(1 / self.PUBLISHING_RATE, self.publish_joint_jog)

    def parameter_event_callback(self, event: ParameterEvent) -> None:
        """Handle parameter change events for this node."""
        if event.node != self.get_fully_qualified_name():
            return

        for param in event.changed_parameters:
            # Skip parameters that are not related to the Dynamixel control parameters
            if not param.name.startswith("dynamixel_"):
                continue
            param_value = parameter_value_to_python(param.value)
            self.gello_hardware.update_dynamixel_control_parameter(param.name, param_value)

    def publish_joint_jog(self) -> None:
        """Publish current joint states and gripper position."""
        JOINT_NAMES = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
        [gello_arm_joints, gripper_position] = self.gello_hardware.read_joint_states()

        arm_joint_states = JointState()
        arm_joint_states.header.stamp = self.get_clock().now().to_msg()
        arm_joint_states.name = JOINT_NAMES
        arm_joint_states.header.frame_id = "fr3_link0"
        arm_joint_states.position = gello_arm_joints.tolist()

        gripper_joint_states = Float32()
        gripper_joint_states.data = gripper_position
        self.arm_joint_publisher.publish(arm_joint_states)
        self.gripper_joint_publisher.publish(gripper_joint_states)

    def destroy_node(self) -> None:
        """Override the destroy_node method to disable torque mode before shutting down."""
        self.gello_hardware.disable_torque()
        super().destroy_node()

    def _determine_default_com_port(self) -> str:
        """Auto-detect GELLO device COM port or return invalid placeholder."""
        matches = glob("/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter*")
        if matches:
            self.get_logger().info(f"Auto-detected com_ports: {matches}")
            return matches[0]
        else:
            self.get_logger().warn("No com_ports detected. Please specify the com_port manually.")
            return "INVALID_COM_PORT"

    def _declare_ros2_param(self, param: ParameterConfig):
        """Declare ROS2 parameters."""
        parameter_value = self.declare_parameter(
            param.descriptor.name, param.default, param.descriptor
        ).get_parameter_value()

        return parameter_value_to_python(parameter_value)

    def _setup_hardware_parameters(self):
        """Declare and setup all hardware configuration parameters."""
        default_com_port = self._determine_default_com_port()
        config = GelloParameterConfig(default_com_port)

        hardware_params: GelloHardwareParams = {}
        for param in config:
            hardware_params[param.descriptor.name] = self._declare_ros2_param(param)

        return hardware_params


def main(args=None):
    rclpy.init(args=args)
    gello_publisher = GelloPublisher()

    try:
        rclpy.spin(gello_publisher)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        gello_publisher.gello_hardware.disable_torque()
        gello_publisher.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
