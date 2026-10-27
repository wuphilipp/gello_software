import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter, parameter_value_to_python
from franka_gello_state_publisher.gello_hardware import GelloHardware, GelloHardwareParams
from franka_gello_state_publisher.gello_parameter_config import (
    ParameterConfig,
    GelloParameterConfig,
)


class GelloPublisher(Node):
    """ROS2 node for publishing GELLO device joint states and handling parameter updates."""

    def __init__(self) -> None:
        super().__init__("gello_publisher")

        hardware_params, publishing_rate = self._setup_hardware_parameters()
        baudrate = int(hardware_params["baudrate"])
        self._warn_if_rate_exceeds_baud(publishing_rate, baudrate)

        try:
            self.gello_hardware = GelloHardware(hardware_params, self.get_logger())
        except (ConnectionError, ValueError) as e:
            self.get_logger().error(f"Failed to initialize GELLO hardware: {e}")
            raise

        self.arm_joint_publisher = self.create_publisher(JointState, "gello/joint_states", 10)
        self.gripper_joint_publisher = self.create_publisher(
            Float32, "gripper/gripper_client/target_gripper_width_percent", 10
        )

        # Hardware writes happen here because Humble's rclpy has no post-set parameter callback.
        self.add_on_set_parameters_callback(self.set_parameters_callback)

        self.get_logger().info(
            f"Publishing GELLO joint states at {publishing_rate} Hz (baudrate {baudrate})."
        )
        self.timer = self.create_timer(1.0 / publishing_rate, self.publish_joint_jog)

    def set_parameters_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Validate parameter updates and write accepted Dynamixel control parameters."""
        result = self.validate_parameter_updates(params)
        if not result.successful:
            return result
        for param in params:
            if param.name not in GelloHardware.DYNAMIXEL_CONTROL_PARAMS:
                continue
            try:
                self.gello_hardware.update_dynamixel_control_parameter(param.name, param.value)
            except (RuntimeError, ValueError) as e:
                reason = f"Failed to write {param.name}: {e}"
                self.get_logger().error(
                    f"{reason}. Motors may be partially updated and differ from the parameters."
                )
                return SetParametersResult(successful=False, reason=reason)
        return result

    def validate_parameter_updates(self, params: list[Parameter]) -> SetParametersResult:
        """Reject updates of Dynamixel control parameters with a wrong number of entries."""
        num_arm_joints = self.get_parameter("num_arm_joints").value
        gripper = self.get_parameter("gripper").value
        for param in params:
            if not param.name.startswith("dynamixel_"):
                continue
            if param.name not in GelloHardware.DYNAMIXEL_CONTROL_PARAMS:
                return SetParametersResult(
                    successful=False,
                    reason=f"{param.name} is not a writable dynamixel control parameter",
                )
            try:
                GelloHardware.validate_dynamixel_control_parameter(
                    param.name, param.value, num_arm_joints, gripper
                )
            except ValueError as e:
                return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

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
        [gello_arm_joints, gripper_position] = self.gello_hardware.get_joint_and_gripper_positions()

        gripper_joint_states = Float32()
        gripper_joint_states.data = gripper_position
        if len(gello_arm_joints) > 0:
            arm_joint_states = JointState()
            arm_joint_states.header.stamp = self.get_clock().now().to_msg()
            arm_joint_states.name = JOINT_NAMES[: len(gello_arm_joints)]
            arm_joint_states.header.frame_id = "fr3_link0"
            arm_joint_states.position = gello_arm_joints.tolist()
            self.arm_joint_publisher.publish(arm_joint_states)
        self.gripper_joint_publisher.publish(gripper_joint_states)

    def destroy_node(self) -> None:
        """Override the destroy_node method to disable torque mode before shutting down."""
        self.gello_hardware.disable_torque()
        super().destroy_node()

    def _declare_ros2_param(self, param: ParameterConfig):
        """Declare ROS2 parameters."""
        parameter_value = self.declare_parameter(
            param.descriptor.name, param.default, param.descriptor
        ).get_parameter_value()

        value = parameter_value_to_python(parameter_value)
        return [] if value is None else value

    def _setup_hardware_parameters(self):
        """Declare and setup all hardware configuration parameters."""
        config = GelloParameterConfig()

        hardware_params: GelloHardwareParams = {}
        for param in config:
            hardware_params[param.descriptor.name] = self._declare_ros2_param(param)

        publishing_rate = int(hardware_params.pop("publishing_rate"))
        if publishing_rate < 1:
            raise ValueError(f"publishing_rate must be >= 1, got {publishing_rate}")
        return hardware_params, publishing_rate

    def _warn_if_rate_exceeds_baud(self, publishing_rate: int, baudrate: int) -> None:
        ceilings = {57600: 25, 115200: 50, 1000000: 100}
        ceiling = ceilings.get(baudrate)
        if ceiling is not None and publishing_rate > ceiling:
            self.get_logger().warning(
                f"publishing_rate {publishing_rate} Hz exceeds unique-sample ceiling "
                f"~{ceiling} Hz at baudrate {baudrate}; publishes may repeat samples."
            )


def main(args=None):
    rclpy.init(args=args)

    try:
        gello_publisher = GelloPublisher()
    except (ConnectionError, ValueError):
        rclpy.try_shutdown()
        return

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
