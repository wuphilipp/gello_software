cmake_minimum_required(VERSION 3.5)
project(franka_fr3_arm_controllers)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default is Debug)" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
  endif()
endif()

option(CHECK_TIDY "Adds clang-tidy tests" OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(controller_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)

add_library(
        ${PROJECT_NAME}
        SHARED
        src/joint_impedance_controller.cpp
        src/motion_generator.cpp)
target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        include
        ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(
        ${PROJECT_NAME}
        controller_interface
        hardware_interface
        pluginlib
        rclcpp
        rclcpp_lifecycle
        franka_semantic_components
        moveit_core
        moveit_msgs
)

pluginlib_export_plugin_description_file(
        controller_interface franka_fr3_arm_controllers.xml)

install(
        TARGETS
        ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
)

install(
        DIRECTORY include/
        DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(
        include
)
ament_export_libraries(
        ${PROJECT_NAME}
)
ament_export_dependencies(
        controller_interface
        pluginlib
        rclcpp
        rclcpp_lifecycle
        hardware_interface
        moveit_core
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_copyright REQUIRED)
  find_package(ament_cmake_cppcheck REQUIRED)
  find_package(ament_cmake_lint_cmake REQUIRED)
  find_package(ament_cmake_xmllint REQUIRED)

  set(CPP_DIRECTORIES src include test)

  ament_cppcheck(${CPP_DIRECTORIES})
  ament_lint_cmake(CMakeLists.txt)
  ament_xmllint()
  ament_clang_format(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-format ${CPP_DIRECTORIES})
  ament_copyright(${CPP_DIRECTORIES} package.xml)

ament_add_gtest(
        test_joint_impedance_controller
        test/test_joint_impedance_controller.cpp
        test/invalid_configuration_test.cpp
        test/setup_test.cpp
)
  if(TARGET test_joint_impedance_controller)
    target_link_libraries(test_joint_impedance_controller ${PROJECT_NAME})
  endif()
endif()

if(CHECK_TIDY)
  find_package(ament_cmake_clang_tidy REQUIRED)
  set(ament_cmake_clang_tidy_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy)
  ament_clang_tidy(${CMAKE_BINARY_DIR})
endif()


ament_package()