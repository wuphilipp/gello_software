import os
import sys
import glob
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from ament_index_python.packages import get_package_prefix
from rcl_interfaces.msg import ParameterEvent, ParameterDescriptor, ParameterValue


class GelloPublisher(Node):
    def __init__(self):
        super().__init__("gello_publisher")

        read_only_desc = ParameterDescriptor(read_only=True)
        default_com_port = self.determine_default_com_port()
        self.com_port = (
            self.declare_parameter("com_port", default_com_port, read_only_desc)
            .get_parameter_value()
            .string_value
        )
        self.gello_name = (
            self.declare_parameter("gello_name", default_com_port, read_only_desc)
            .get_parameter_value()
            .string_value
        )
        self.num_arm_joints = (
            self.declare_parameter("num_arm_joints", 7, read_only_desc)
            .get_parameter_value()
            .integer_value
        )
        self.joint_signs = np.array(
            self.declare_parameter("joint_signs", [1] * 7, read_only_desc)
            .get_parameter_value()
            .integer_array_value
        )
        self.gripper = (
            self.declare_parameter("gripper", True, read_only_desc)
            .get_parameter_value()
            .bool_value
        )
        self.num_total_joints = self.num_arm_joints + (1 if self.gripper else 0)
        self.gripper_range_rad = (
            self.declare_parameter("gripper_range_rad", (0.0, 0.0), read_only_desc)
            .get_parameter_value()
            .double_array_value
        )
        self.best_offsets = np.array(
            self.declare_parameter("best_offsets", [0.0] * 7, read_only_desc)
            .get_parameter_value()
            .double_array_value
        )

        self.initialize_dynamixels()

        self.arm_joint_publisher = self.create_publisher(JointState, "gello/joint_states", 10)
        self.gripper_joint_publisher = self.create_publisher(
            Float32, "gripper/gripper_client/target_gripper_width_percent", 10
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

    def parameter_event_callback(self, event: ParameterEvent):
        for param in event.changed_parameters:
            # Skip parameters that are not related to the Dynamixel control parameters
            if not param.name.startswith("dynamixel_"):
                continue
            # Extract the data name and value
            data_name = param.name.replace("dynamixel_", "")
            def convert(x): return x
            for entry in self.ordered_dynamixel_params:
                if entry["name"] == data_name:
                    if "convert" in entry:
                        convert = entry["convert"]
                    break
            value = list(convert(param.value))
            # Write the parameter value to the Dynamixel driver
            self.driver.write_value_by_name(data_name, value)
            self.get_logger().info(f"Parameter {data_name} changed to {value}. Updated Dynamixels.")

    def determine_default_com_port(self) -> str:
        matches = glob.glob("/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter*")
        if matches:
            self.get_logger().info(f"Auto-detected com_ports: {matches}")
            return matches[0]
        else:
            self.get_logger().warn("No com_ports detected. Please specify the com_port manually.")
            return "INVALID_COM_PORT"

    def initialize_dynamixels(self):
        joint_ids = list(range(1, self.num_total_joints + 1))
        self.add_dynamixel_driver_path()
        from gello.dynamixel.driver import DynamixelDriver
        self.driver = DynamixelDriver(joint_ids, port=self.com_port, baudrate=57600)

        def to_integer_array_value(param_value: ParameterValue) -> list[int]:
            return list(param_value.integer_array_value)
        
        def goal_position_to_pulses(param_value: ParameterValue) -> list[int]:
            goals = np.array(param_value.double_array_value)
            arm_goals_raw = (goals[:self.num_arm_joints] * self.joint_signs) + self.best_offsets
            goals_raw = np.append(arm_goals_raw, goals[-1]) if self.gripper else arm_goals_raw
            return [self.driver._rad_to_pulses(rad) for rad in goals_raw]

        CURRENT_BASED_POSITION_MODE = 5
        CURRENT_LIMIT = 600  # mA
        self.ordered_dynamixel_params = [
            {
                "name": "operating_mode",  # resets kp_p, kp_i, kp_d, goal_current, goal_position
                "default": [CURRENT_BASED_POSITION_MODE] * self.num_total_joints,
            },
            {
                "name": "goal_current",
                "default": [CURRENT_LIMIT] * self.num_total_joints,
            },
            {
                "name": "kp_p",
                "default": [0] * self.num_total_joints,
                "constraints": "Sensible values: 0 to ~1000",
                "is_ros2_param": True,
                "convert": to_integer_array_value
            },
            {
                "name": "kp_i",
                "default": [0] * self.num_total_joints,
                "constraints": "Sensible values: 0 to ~1000",
                "is_ros2_param": True,
                "convert": to_integer_array_value
            },
            {
                "name": "kp_d",
                "default": [0] * self.num_total_joints,
                "constraints": "Sensible values: 0 to ~1000",
                "is_ros2_param": True,
                "convert": to_integer_array_value
            },
            {
                "name": "torque_enable",  # resets goal_position
                "default": [0] * self.num_total_joints,
                "constraints": "0 (disabled), 1 (enabled)",
                "is_ros2_param": True,
                "convert": to_integer_array_value
            },
            {
                "name": "goal_position",
                "default": [0.0] * self.num_total_joints,
                "constraints": "4095 corresponds to 360 degrees",
                "is_ros2_param": True,
                "convert": goal_position_to_pulses,
            },
        ]

        for param in self.ordered_dynamixel_params:
            param_value = param["default"]
            if param.get("is_ros2_param", False):
                desc = ParameterDescriptor(additional_constraints=param.get("constraints", ""))
                param_value = (
                    self.declare_parameter(
                        f"dynamixel_{param['name']}", param["default"], desc
                    )
                    .get_parameter_value()
                )
            convert = param.get("convert", lambda x: x)
            self.driver.write_value_by_name(param["name"], list(convert(param_value)))
        self.get_logger().info("Sent initial control parameters to all Dynamixel motors.")

        # Subscribe to parameter events to allow dynamic updates of parameters
        self.parameter_subscription = self.create_subscription(
            ParameterEvent,
            '/parameter_events',
            self.parameter_event_callback,
            10
        )

    def __post_init__(self):
        assert len(self.joint_signs) == self.num_arm_joints
        for idx, j in enumerate(self.joint_signs):
            assert j == -1 or j == 1, f"Joint idx: {idx} should be -1 or 1, but got {j}."

    def publish_joint_jog(self):
        gello_joints_raw = self.driver.get_joints()

        gello_arm_joints_raw = gello_joints_raw[:self.num_arm_joints]
        gello_arm_joints = (gello_arm_joints_raw - self.best_offsets) * self.joint_signs

        arm_joint_states = JointState()
        arm_joint_states.header.stamp = self.get_clock().now().to_msg()
        arm_joint_states.name = self.joint_names
        arm_joint_states.header.frame_id = "fr3_link0"
        arm_joint_states.position = [float(joint) for joint in gello_arm_joints]

        gripper_joint_states = Float32()
        if self.gripper:
            gripper_position = gello_joints_raw[-1]
            gripper_joint_states.data = self.gripper_readout_to_percent(gripper_position)
        else:
            gripper_joint_states.data = 0.0
        self.arm_joint_publisher.publish(arm_joint_states)
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

    def destroy_node(self) -> None:
        """Override the destroy_node method to disable torque mode before shutting down."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.set_torque_mode(False)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    gello_publisher = GelloPublisher()
    try:
        rclpy.spin(gello_publisher)
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    finally:
        # Always destroy the node to ensure torque is disabled
        try:
            gello_publisher.destroy_node()
        except Exception:
            pass  # Node might already be destroyed
        # Only shutdown if ROS2 is still running
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
