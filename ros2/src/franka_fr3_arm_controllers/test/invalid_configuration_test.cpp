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

/**
 * @brief Test fixture for the JointImpedanceController
 */
class InvalidConfigurationTest : public JointImpedanceControllerTest,
                                 public ::testing::WithParamInterface<rclcpp::Parameter> {};

TEST_P(InvalidConfigurationTest, TestInvalidConfiguration) {
  auto parameter_invalid_configuration = GetParam();
  controller_->get_node()->set_parameters({parameter_invalid_configuration});
  controller_->on_init();
  rclcpp_lifecycle::State state;

  EXPECT_EQ(controller_->on_configure(state), CallbackReturn::FAILURE);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidConfigurations,
    InvalidConfigurationTest,
    ::testing::Values(rclcpp::Parameter("k_gains",
                                        std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                            1.0}),  // exceeding number of joints
                      rclcpp::Parameter("d_gains",
                                        std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                            1.0}),  // exceeding number of joints
                      rclcpp::Parameter("k_gains", std::vector<double>{}),  // empty
                      rclcpp::Parameter("d_gains", std::vector<double>{}),  // empty
                      rclcpp::Parameter("k_alpha", double(-1.0)),           // out of range (<0)
                      rclcpp::Parameter("k_alpha", double(2.0))));          // out of range (>1)
