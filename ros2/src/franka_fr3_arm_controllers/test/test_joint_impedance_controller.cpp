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

JointImpedanceControllerTest::JointImpedanceControllerTest() : done_(false) {
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces_.emplace_back(joint_names_[i], HW_IF_EFFORT, &joint_commands_[i]);
    state_interfaces_.emplace_back(joint_names_[i], HW_IF_POSITION, &position_states_[i]);
    state_interfaces_.emplace_back(joint_names_[i], HW_IF_VELOCITY, &velocity_states_[i]);
  }
}

void JointImpedanceControllerTest::SetUp() {
  rclcpp::init(0, nullptr);
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();

  controller_ = std::make_shared<franka_fr3_arm_controllers::JointImpedanceController>();
  controller_->init("single_arm_controller_test");

  mock_parameter_server_ = std::make_shared<MockRobotParameterServer>();
  mock_joint_state_publisher_ = std::make_shared<MockGelloJointStatePublisher>();

  executor_->add_node(controller_->get_node()->get_node_base_interface());
  executor_->add_node(mock_parameter_server_);
  executor_->add_node(mock_joint_state_publisher_);

  std::vector<LoanedCommandInterface> loaned_command_interfaces;
  loaned_command_interfaces.reserve(command_interfaces_.size());
  for (auto& command_interface : command_interfaces_) {
    loaned_command_interfaces.emplace_back(command_interface);
  }

  std::vector<LoanedStateInterface> loaned_state_interfaces;
  loaned_state_interfaces.reserve(state_interfaces_.size());
  for (auto& state_interface : state_interfaces_) {
    loaned_state_interfaces.emplace_back(state_interface);
  }

  controller_->assign_interfaces(std::move(loaned_command_interfaces),
                                 std::move(loaned_state_interfaces));

  setValidControllerParameters();

  startExecutorThread();
}

void JointImpedanceControllerTest::TearDown() {
  stopExecutorThread();

  executor_->remove_node(controller_->get_node()->get_node_base_interface());
  executor_->remove_node(mock_joint_state_publisher_);
  executor_->remove_node(mock_parameter_server_);

  rclcpp::shutdown();
}

void JointImpedanceControllerTest::startExecutorThread() {
  done_ = false;
  executor_thread_ = std::thread([this]() {
    std::unique_lock<std::mutex> lock(cv_mut_executor_thread_);
    while (!done_) {
      executor_->spin_some();
      cv_executor_thread_.wait_for(lock, std::chrono::milliseconds(10));
    }
  });
}

void JointImpedanceControllerTest::stopExecutorThread() {
  {
    std::lock_guard<std::mutex> lock(cv_mut_executor_thread_);
    done_ = true;
  }
  cv_executor_thread_.notify_all();
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
JointImpedanceControllerTest::startController() {
  controller_->on_init();
  rclcpp_lifecycle::State state;
  controller_->on_configure(state);
  return controller_->on_activate(state);
}

void JointImpedanceControllerTest::setRobotPosition(const std::vector<double>& positions) {
  for (size_t i = 0; i < positions.size(); ++i) {
    position_states_[i] = positions[i];
  }
}

void JointImpedanceControllerTest::setValidControllerParameters() {
  controller_->get_node()->set_parameters(
      {{"arm_id", kArmId_}, {"k_gains", kKGains_}, {"d_gains", kDGains_}, {"k_alpha", kKAlpha_}});
}
