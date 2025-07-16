// Copyright (c) 2025 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/test_joint_impedance_controller.hpp"

TEST_F(JointImpedanceControllerTest, TestInitialization) {
  EXPECT_EQ(controller_->on_init(), CallbackReturn::SUCCESS);
}

TEST_F(JointImpedanceControllerTest, TestValidConfiguration) {
  controller_->on_init();
  rclcpp_lifecycle::State state;

  EXPECT_EQ(controller_->on_configure(state), CallbackReturn::SUCCESS);
}

TEST_F(JointImpedanceControllerTest, TestActivate) {
  EXPECT_EQ(startController(), CallbackReturn::SUCCESS);
}

TEST_F(JointImpedanceControllerTest, TestUpdateMotionGeneratorOnly) {
  static const std::vector<double> kInitialRobotPosition = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  static constexpr double kExpectedValue = -0.05;
  static constexpr double kTolerance = 1e-6;

  setRobotPosition(kInitialRobotPosition);
  startController();

  rclcpp::Time time;
  rclcpp::Duration period = rclcpp::Duration::from_seconds(1.0);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);

  for (size_t i = 0; i < joint_commands_.size(); ++i) {
    EXPECT_NEAR(joint_commands_[i], kExpectedValue, kTolerance);
  }
}

TEST_F(JointImpedanceControllerTest, TestUpdateMotionGeneratorAndGelloPositionValues) {
  static const std::vector<double> kInitialRobotPosition = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  static constexpr double kExpectedValue = -0.05;
  static constexpr double kTolerance = 1e-6;

  setRobotPosition(kInitialRobotPosition);
  startController();

  rclcpp::Time time;
  rclcpp::Duration period = rclcpp::Duration::from_seconds(1.0);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);

  for (size_t i = 0; i < joint_commands_.size(); ++i) {
    EXPECT_NEAR(joint_commands_[i], kExpectedValue, kTolerance);
  }
}

TEST_F(JointImpedanceControllerTest, TestUpdateGelloPositionValuesOnly) {
  static const std::vector<double> kInitialRobotPosition = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  static constexpr double kExpectedValue = -0.075;
  static constexpr double kTolerance = 1e-6;

  setRobotPosition(kInitialRobotPosition);
  startController();

  rclcpp::Time time;
  rclcpp::Duration period = rclcpp::Duration::from_seconds(1.0);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);
  period = rclcpp::Duration::from_seconds(1.1);
  EXPECT_EQ(controller_->update(time, period), controller_interface::return_type::OK);

  for (size_t i = 0; i < joint_commands_.size(); ++i) {
    EXPECT_NEAR(joint_commands_[i], kExpectedValue, kTolerance);
  }
}

TEST_F(JointImpedanceControllerTest, TestUpdateInvalidGelloPositionValues) {
  GTEST_SKIP() << "Skipping this test because yet not implemented";

  /**
   * Description: Modify mock_joint_state_publisher.hpp (Fake Gello), to be reconfigured to publish
   * invalid joint states (invalid / old timestamp).
   */
}

TEST_F(JointImpedanceControllerTest, TestCommandInterfaceConfiguration) {
  controller_->on_init();
  rclcpp_lifecycle::State state;
  controller_->on_configure(state);

  auto config = controller_->command_interface_configuration();
  EXPECT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  ASSERT_EQ(config.names.size(), command_interfaces_.size());
  for (size_t i = 0; i < command_interfaces_.size(); i++) {
    EXPECT_EQ(config.names[i], joint_names_[i] + "/" + HW_IF_EFFORT);
  }
}

TEST_F(JointImpedanceControllerTest, TestStateInterfaceConfiguration) {
  controller_->on_init();
  rclcpp_lifecycle::State state;
  controller_->on_configure(state);

  auto config = controller_->state_interface_configuration();
  EXPECT_EQ(config.type, controller_interface::interface_configuration_type::INDIVIDUAL);
  ASSERT_EQ(config.names.size(), state_interfaces_.size());
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    EXPECT_EQ(config.names[2 * i], joint_names_[i] + "/" + HW_IF_POSITION);
    EXPECT_EQ(config.names[2 * i + 1], joint_names_[i] + "/" + HW_IF_VELOCITY);
  }
}