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

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

/**
 * @class MockGelloJointStatePublisher
 * @brief A mock joint state publisher node replacing the gello publisher for testing purposes.
 *
 */
class MockGelloJointStatePublisher : public rclcpp::Node {
 public:
  MockGelloJointStatePublisher() : Node("joint_state_publisher_node") {
    publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("/gello/joint_states", 3);
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(100),
                                std::bind(&MockGelloJointStatePublisher::publishJointState_, this));
  }

 private:
  void publishJointState_() {
    auto message = sensor_msgs::msg::JointState();
    message.header.stamp = this->now();
    message.name = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"};
    message.position = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    message.velocity = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    message.effort = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};

    publisher_->publish(message);
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};