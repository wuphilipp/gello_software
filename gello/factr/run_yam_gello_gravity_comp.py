#!/usr/bin/env python3
"""
Direct YAM_GELLO Gravity Compensation Script

This script directly launches the YAM_GELLO in FACTR gravity compensation mode.
"""

import sys
import signal
import numpy as np
from pathlib import Path
import pinocchio as pin
import time

# Add the gello package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gello.dynamixel.driver import DynamixelDriver


def calibrate_joint_offsets(
    driver: DynamixelDriver, joint_signs: list, num_joints: int = 6
) -> np.ndarray:
    """
    Automatically calibrate joint offsets using the proven approach from gello_get_offset.py
    """
    print("Calibrating joint offsets using proven gello_get_offset.py method...")

    for _ in range(10):
        driver.get_positions_and_velocities()  # FIXED: Use same method as main loop

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = 0.0  # Target is [0,0,0,0,0,0] for YAM_GELLO URDF
        return np.abs(joint_i - start_i)

    # Get current joint positions
    curr_joints, _ = (
        driver.get_positions_and_velocities()
    )  # FIXED: Use same method as main loop
    print(f"Current raw joint positions: {[f'{x:.3f}' for x in curr_joints]}")

    # Target calibration position: YAM_GELLO URDF home [0,0,0,0,0,0]
    print("Target calibration position: [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]")
    print("   This targets the YAM_GELLO URDF's home position")

    # Search for best offsets using the proven method
    best_offsets = []
    for i in range(num_joints):
        best_offset = 0
        best_error = 1e6

        # Search in range ±8π with intervals of π/2 (same as gello_get_offset.py)
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 8 * 4 + 1):
            error = get_error(offset, i, curr_joints)
            if error < best_error:
                best_error = error
                best_offset = offset

        best_offsets.append(best_offset)
        print(
            f"  Joint {i+1}: best offset = {best_offset:.3f} (error = {best_error:.3f})"
        )

    joint_offsets = np.array(best_offsets)
    print(f"✅ Calibrated offsets: {[f'{x:.3f}' for x in joint_offsets]}")
    print(
        f"   Offsets as multiples of π/2: [{', '.join([f'{int(np.round(x/(np.pi/2)))}*π/2' for x in joint_offsets])}]"
    )

    return joint_offsets


def main():
    """Main function to run YAM_GELLO gravity compensation"""

    print("=" * 60)
    print("YAM_GELLO FACTR Gravity Compensation (Direct Physics)")
    print("=" * 60)

    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4GA-if00-port0"
    servo_types = [
        "XC330_T288_T",
        "XM430_W210_T",
        "XM430_W210_T",
        "XC330_T288_T",
        "XC330_T288_T",
        "XC330_T288_T",
    ]
    joint_signs = [1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    urdf_path = str(
        Path(__file__).parent / "urdf" / "yam_active_gello" / "robot.urdf"
    )  # FIXED: Absolute path

    print(f"Configuration:")
    print(f"  Port: {port}")
    print(f"  Servo types: {servo_types}")
    print(f"  Joint signs: {joint_signs}")
    print(f"  URDF path: {urdf_path}")
    print(f"  DOF: 6")

    # Check if URDF exists
    if not Path(urdf_path).exists():
        print(f"Error: URDF file not found: {urdf_path}")
        print("Make sure the yam_active_gello folder is in the correct location")
        sys.exit(1)

    # First, create a temporary driver to calibrate offsets
    try:
        print("\n🔧 Creating temporary driver for offset calibration...")
        temp_driver = DynamixelDriver(
            ids=list(range(1, 7)),  # [1, 2, 3, 4, 5, 6]
            port=port,
            servo_types=servo_types,
        )
        print("✓ Temporary driver created successfully")

        # Check if robot is in a reasonable position
        print("\n🔍 Checking robot position...")
        test_positions, _ = (
            temp_driver.get_positions_and_velocities()
        )  # FIXED: Use same method as main loop
        print(f"Current robot position: {[f'{x:.3f}' for x in test_positions]}")

        extreme_threshold = 0.8  # radians
        if any(abs(pos) > extreme_threshold for pos in test_positions):
            print("⚠️  WARNING: Robot appears to be in extreme positions")
            print("   Consider moving to a more comfortable, extended pose")
            print("   This will improve calibration accuracy")

        # Calibrate joint offsets automatically
        joint_offsets = calibrate_joint_offsets(temp_driver, joint_signs)

        # Close temporary driver
        temp_driver.close()
        print("✓ Temporary driver closed")

    except Exception as e:
        print(f"✗ Failed to calibrate offsets: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        sys.exit(1)

    # Now create the main driver and physics system
    try:
        print("\n🔧 Creating main driver and physics system...")
        driver = DynamixelDriver(
            ids=list(range(1, 7)), port=port, servo_types=servo_types
        )
        print("✓ Main driver created successfully")

        # Configure servos
        driver.set_torque_mode(False)
        driver.set_operating_mode(0)  # Current control mode
        driver.set_torque_mode(True)
        print("✓ Servos configured for current control")

        # Load Pinocchio model
        print(f"Loading URDF: {urdf_path}")
        urdf_model_dir = str(Path(urdf_path).parent)
        pin_model, _, _ = pin.buildModelsFromUrdf(
            filename=str(urdf_path), package_dirs=urdf_model_dir
        )
        pin_data = pin_model.createData()
        print("✓ Pinocchio model loaded successfully")

        # Print configuration summary for debugging
        print(f"\n🔧 CONFIGURATION SUMMARY:")
        print(f"  Joint signs: {joint_signs}")
        print(f"  Joint offsets: {[f'{x:.3f}' for x in joint_offsets]}")
        print(f"  Servo types: {servo_types}")
        print(f"  Gravity gain: 0.4")
        print(f"  URDF: {urdf_path}")
        print(f"  Control frequency: 100 Hz")

    except Exception as e:
        print(f"✗ Failed to create main system: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        sys.exit(1)

    # Setup signal handler for clean shutdown
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal")
        driver.set_torque_mode(False)
        driver.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("\n" + "=" * 60)
        print("FACTR Gravity Compensation Active (Direct Physics)")
        print("=" * 60)
        print("The YAM_GELLO should now feel lighter and more responsive.")
        print(
            "You can manually move the device and it will provide gravity compensation."
        )
        print("   If any joint feels wrong, press Ctrl+C immediately")
        print("Press Ctrl+C to stop.")
        print("=" * 60 + "\n")

        # Run the gravity compensation loop
        print("Starting gravity compensation control loop at 100 Hz")
        print("Press Ctrl+C to stop")

        running = True
        dt = 1.0 / 500.0

        # Control parameters
        joint_limit_kp = 1.0
        joint_limit_kd = 0.05
        null_space_kp = 0.1
        null_space_kd = 0.01
        stiction_comp_enable_speed = 0.1
        stiction_comp_gain = 0.2
        stiction_dither_flag = np.ones(6, dtype=bool)

        # Joint limits from URDF (all joints have ±π limits)
        joint_limits_min = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
        joint_limits_max = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

        # Null space target (comfortable middle position)
        null_space_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        while running:
            start_time = time.time()

            try:
                # Get current joint states
                joint_pos_raw, joint_vel_raw = driver.get_positions_and_velocities()

                # Check if we got valid data
                if joint_pos_raw is None or joint_vel_raw is None:
                    print(f"\n⚠️  Warning: No joint data received, skipping iteration")
                    time.sleep(0.01)
                    continue

                # Apply offsets and signs for arm joints
                joint_pos_arm = (joint_pos_raw - joint_offsets) * joint_signs
                joint_vel_arm = joint_vel_raw * joint_signs

                # Initialize torque commands
                torque_arm = np.zeros(6)

                # 1. JOINT LIMIT BARRIERS
                exceed_max_mask = joint_pos_arm > joint_limits_max
                tau_l = (
                    -joint_limit_kp * (joint_pos_arm - joint_limits_max)
                    - joint_limit_kd * joint_vel_arm
                ) * exceed_max_mask

                exceed_min_mask = joint_pos_arm < joint_limits_min
                tau_l += (
                    -joint_limit_kp * (joint_pos_arm - joint_limits_min)
                    - joint_limit_kd * joint_vel_arm
                ) * exceed_min_mask

                torque_arm += tau_l

                # 2. NULL SPACE REGULATION
                J = pin.computeJointJacobian(pin_model, pin_data, joint_pos_arm, 6)
                J_dagger = np.linalg.pinv(J)
                null_space_projector = np.eye(6) - J_dagger @ J
                q_error = joint_pos_arm - null_space_target
                tau_n = null_space_projector @ (
                    -null_space_kp * q_error - null_space_kd * joint_vel_arm
                )
                torque_arm += tau_n

                # 3. GRAVITY COMPENSATION
                tau_g = pin.rnea(
                    pin_model,
                    pin_data,
                    joint_pos_arm,
                    joint_vel_arm,
                    np.zeros_like(joint_vel_arm),
                )
                tau_g *= 0.4  # Gravity compensation gain
                torque_arm += tau_g

                # 4. FRICTION COMPENSATION
                tau_ss = np.zeros(6)
                for i in range(6):
                    if abs(joint_vel_arm[i]) < stiction_comp_enable_speed:
                        if stiction_dither_flag[i]:
                            tau_ss[i] += stiction_comp_gain * abs(tau_g[i])
                        else:
                            tau_ss[i] -= stiction_comp_gain * abs(tau_g[i])
                        stiction_dither_flag[i] = ~stiction_dither_flag[i]
                torque_arm += tau_ss

                # LIVE DEBUG PRINTING - see what's happening
                print(
                    f"\r🔍 LIVE: Raw={[f'{x:.2f}' for x in joint_pos_raw[:3]]}... | "
                    f"Calibrated={[f'{x:.2f}' for x in joint_pos_arm[:3]]}... | "
                    f"Total_torques={[f'{x:.2f}' for x in torque_arm[:3]]}... | "
                    f"Applied_torques={[f'{x:.2f}' for x in (torque_arm * joint_signs)[:3]]}...",
                    end="",
                    flush=True,
                )

                # Special debugging for joint 3 (the problematic one)
                if abs(joint_pos_arm[2]) > 0.4:  # If joint 3 is getting extreme
                    print(
                        f"\n🚨 JOINT 3 DEBUG: pos={joint_pos_arm[2]:.3f}, raw={joint_pos_raw[2]:.3f}, "
                        f"offset={joint_offsets[2]:.3f}, sign={joint_signs[2]}, "
                        f"gravity_torque={tau_g[2]:.3f}, total_torque={torque_arm[2]:.3f}"
                    )

                try:
                    driver.set_torque((torque_arm * joint_signs).tolist())
                except Exception as e:
                    print(f"\n⚠️  Warning: Failed to set torque: {e}, continuing...")

                # Maintain loop timing
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"\n⚠️  Loop overrun: {elapsed - dt:.4f}s")

            except KeyboardInterrupt:
                running = False
            except Exception as e:
                print(f"\n❌ Error in control loop: {e}")
                print(f"   Continuing after 0.1s delay...")
                time.sleep(0.1)
                continue

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error during operation: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        print("⚠️  Emergency shutdown due to error")
    finally:
        try:
            driver.set_torque_mode(False)
            driver.close()
            print("YAM_GELLO FACTR agent shutdown complete")
        except Exception as e:
            print(f"Warning: Error during shutdown: {e}")
            print("Forcing cleanup...")


if __name__ == "__main__":
    main()
