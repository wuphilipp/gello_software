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

#pragma once

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include "franka_fr3_arm_controllers/joint_impedance_controller.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "mocks/mock_joint_state_publisher.hpp"
#include "mocks/mock_parameter_server.hpp"

using hardware_interface::CommandInterface;
using hardware_interface::HW_IF_EFFORT;
using hardware_interface::HW_IF_POSITION;
using hardware_interface::HW_IF_VELOCITY;
using hardware_interface::LoanedCommandInterface;
using hardware_interface::LoanedStateInterface;
using hardware_interface::StateInterface;

/**
 * @brief Test fixture for the JointImpedanceController
 *
 */
class JointImpedanceControllerTest : public ::testing::Test {
 public:
  JointImpedanceControllerTest();

 protected:
  void SetUp() override;
  void TearDown() override;
  void startExecutorThread();
  void stopExecutorThread();
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn startController();
  void setRobotPosition(const std::vector<double>& positions);
  void setValidControllerParameters();

  std::shared_ptr<franka_fr3_arm_controllers::JointImpedanceController> controller_;
  std::shared_ptr<MockRobotParameterServer> mock_parameter_server_;
  std::shared_ptr<MockGelloJointStatePublisher> mock_joint_state_publisher_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::thread executor_thread_;
  std::condition_variable cv_executor_thread_;
  std::mutex cv_mut_executor_thread_;
  bool done_;

  /**
   * @brief Current state of the "robot"
   */
  static constexpr size_t num_joints = 7;
  static constexpr std::array<double, num_joints> kInitialJointCommands = {1.0, 1.0, 1.0, 1.0,
                                                                           1.0, 1.0, 1.0};
  static constexpr std::array<double, num_joints> kInitialPositionStates = {1.0, 1.0, 1.0, 1.0,
                                                                            1.0, 1.0, 1.0};
  static constexpr std::array<double, num_joints> kInitialVelocityStates = {1.0, 1.0, 1.0, 1.0,
                                                                            1.0, 1.0, 1.0};

  const std::array<std::string, num_joints> joint_names_ = {
      "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
      "fr3_joint5", "fr3_joint6", "fr3_joint7"};

  std::vector<double> joint_commands_{kInitialJointCommands.begin(), kInitialJointCommands.end()};
  std::vector<double> position_states_{kInitialPositionStates.begin(),
                                       kInitialPositionStates.end()};
  std::vector<double> velocity_states_{kInitialVelocityStates.begin(),
                                       kInitialVelocityStates.end()};
  std::vector<CommandInterface> command_interfaces_;
  std::vector<StateInterface> state_interfaces_;
};