# Changelog

> This changelog covers only the changes relevant to the ROS 2 implementation of GELLO.

## `ros2-v2.1.0` - 2026-04-08
 - Updated the Dockerfile and Docker Compose files to support Cyclone DDS as the ROS 2 RMW.
 - **Behavioral**: Cyclone DDS is now the default RMW in the devcontainer.
 - Added getting started guide for Franka GELLO Duo.

## `ros2-v2.0.0` - 2026-02-16
 - Published joint positions are now independent of the GELLO's joint positions on power-on.
 - Added assembly offset calculation script which differs from the one in the non-ROS part of this repository.
 - **Breaking**: In the publisher's config files, the `best_offsets` parameter has been renamed to 
  `assembly_offsets` and their values need to be updated:
    - For pre-assembled Franka GELLO Duo's, use the new values provided.
    - For self-assembled GELLO variants, use the newly added `get_offsets.py` script to determine
      them (see README).

## `ros2-v1.2.1` - 2026-01-22
 - Bump dependencies for compatibility with Franka FR3 robot system version 5.9.x.

## `ros2-v1.2.0` - 2025-12-15
 - Adaptations for OpenRB-150 support as an alternative to the U2D2 converter.
 - In the publisher's default config files, virtual springs and dampers are now disabled by default, in order to prevent accidental high power draw when the OpenRB-150 board is used without external power supply.
 - Added a configuration file for the pre-assembled Franka GELLO Duo.
 - Renamed example config files and updated the default config file used by `franka_gello_state_publisher` accordingly.
 - Improved Dynamixel initialization reliability by starting joint-position polling only after initialization is complete.

## `ros2-v1.1.0` - 2025-11-20
 - Improved error handling for Dynamixel driver connection failures.
 - In dual arm systems, always terminate both publishers when one of them shuts down
 - Signals get handled by ROS 2 launch system instead of by the publisher node itself.

## `ros2-v1.0.0` - 2025-11-17
 - A more generalized Dynamixel motor driver is introduced. To avoid breaking changes with non-ROS2 parts of this repository, the driver is now part of the `franka_gello_state_publisher` package.
 - Refactored `franka_gello_state_publisher` into modular components.
 - Added functionality for configurable virtual springs and dampers using the Dynamixel motors internal current-based position mode
 - **Breaking**: In the publisher's config files, the `num_joints` parameter has been renamed to `num_arm_joints`. Please update your config files accordingly.

## `ros2-v0.2.1` - 2025-11-04
 - Fixed Dev Container build failure by updating Franka-related dependencies

## `ros2-v0.2.0` - 2025-07-22
 - Added support for controlling multiple Franka FR3 arms in ROS 2 using namespaces.
 - Replaced all links and references to github.com/frankaemika with github.com/frankarobotics

## `ros2-v0.1.1` - 2025-06-13
 - Improve setup process of the Franka ROS 2 integration for local environments

## `ros2-v0.1.0` - 2025-06-03
 - Added ROS 2 implementation of GELLO for Franka FR3 robots:
     - `franka_fr3_arm_controllers`: joint impedance controller for Franka FR3 robots based on GELLO input
     - `franka_gello_state_publisher`: publishes GELLO joint states as `sensor_msgs/JointState` ROS 2 messages
     - `franka_gripper_manager`: control of Franka Hand and Robotiq 2F-85 grippers based on GELLO input
