import atexit
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict

# Global variables for cleanup
active_threads = []
active_servers = []
active_cameras = {}
active_save_interface = None
cleanup_in_progress = False


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    # First, before any other output: this finalises an in-progress episode and
    # restores the terminal. signal_handler calls os._exit(0), which runs no
    # atexit handlers, so this is all that stands between Ctrl-C and a shell
    # left with no echo. It also clears the status line, so printing before it
    # would interleave with it.
    if active_save_interface is not None:
        try:
            active_save_interface.close()
        except Exception as e:
            print(f"Error closing save interface: {e}")

    print("Cleaning up resources...")

    for name, camera in active_cameras.items():
        try:
            if hasattr(camera, "close"):
                camera.close()
        except Exception as e:
            print(f"Error closing camera {name}: {e}")

    for server in active_servers:
        try:
            # ZMQServerRobot exposes stop(), not close() -- checking only for
            # close() meant serve() was never asked to break out of its loop.
            if hasattr(server, "stop"):
                server.stop()
            elif hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error stopping server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")

    # Both exit paths end in os._exit(), which does not flush buffered stdio.
    # Interactive runs are line-buffered so nothing is lost, but piping to a log
    # file is block-buffered and would silently drop the session summary.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    use_save_interface: bool = False
    """Enable saving data with keyboard interface."""

    task: Optional[str] = None
    """Natural-language task description. Used as the policy prompt and to group
    episodes on disk. Prompted for interactively if omitted."""

    data_dir: str = "data"
    """Root directory for recorded episodes."""

    jpeg_quality: int = 95
    """JPEG quality for recorded RGB frames."""

    monitor_port: Optional[int] = None
    """Serve a live web monitor (camera view + recording status) on this port.
    Try 8081, then open http://<host>:8081/ on a second screen."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Motor chains are brought up BEFORE the GELLO leaders.
    #
    # DynamixelDriver.__init__ starts a daemon thread that polls joint states
    # over FTDI serial in a tight loop. Creating the agent first leaves one such
    # thread per leader holding the GIL while DMChainCanInterface._motor_on()
    # tries to land each motor's CAN reply inside a 10 ms timeout -- and the
    # FTDI adapters share a USB 2.0 hub with the CAN adapters. Missing that
    # window aborts bring-up on whichever motor id happened to be next, which is
    # why the failing id moved around between runs.
    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    # Cameras open LAST, after both motor chains are up.
    #
    # Ordering here is load-bearing and both directions hurt, so this is a
    # trade, not a clean win. Cameras left streaming during motor bring-up add
    # enough contention to blow the 10 ms CAN receive timeout in _motor_on(),
    # which fails the second arm outright. Opening them afterwards instead
    # stalls the already-running chains for ~0.65 s inside pipeline.start(),
    # which holds the GIL -- survivable, but only because the configs enable
    # i2rt's enable_auto_recovery so that stall is recovered rather than fatal.
    #
    # Cameras are shared across both arms, so they come from the primary config
    # only, same convention as hz.
    if bimanual and right_cfg.get("cameras"):
        print("Warning: ignoring 'cameras' in the right config; cameras are read "
              "from the left/primary config only.")
    camera_cfg = cfg.get("cameras") or {}
    if camera_cfg:
        print(f"Opening {len(camera_cfg)} camera(s): {', '.join(camera_cfg)}")
    else:
        print("Warning: no 'cameras:' block in config; recording state only.")
    camera_dict = instantiate_from_dict(camera_cfg)
    active_cameras.update(camera_dict)

    env = RobotEnv(
        robot_client, control_rate_hz=cfg.get("hz", 30), camera_dict=camera_dict
    )

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface if requested
    global active_save_interface
    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=args.data_dir,
            agent_name=agent.__class__.__name__,
            expand_user=True,
            task=args.task,
            jpeg_quality=args.jpeg_quality,
            monitor_port=args.monitor_port,
            meta_extra={
                "robot": robot.__class__.__name__,
                "bimanual": bimanual,
                "hz_target": cfg.get("hz", 30),
                "cameras": {
                    name: camera_cfg[name].get("device_id") for name in camera_dict
                },
                "config_paths": {
                    "left": args.left_config_path,
                    "right": args.right_config_path,
                },
                "configs": {
                    "left": left_cfg,
                    "right": right_cfg if bimanual else None,
                },
            },
        )
        active_save_interface = save_interface

    # Run main control loop
    run_control_loop(env, agent, save_interface)

    # Quitting with 'x' returns here normally, and a normal interpreter exit
    # blocks on non-daemon threads: the ZMQ server thread plus i2rt's per-chain
    # _set_torques_and_update_state and robot_server threads. None of them
    # return on their own, so this used to print "Exiting." and hang. Take the
    # same path SIGINT already takes.
    #
    # Deliberately NOT calling i2rt's robot.close(): its docstring is "safely
    # close the robot by setting all torques to zero", which drops an arm that
    # gravity compensation is currently holding up. Exiting this way leaves the
    # motors on their last grav-comp command, exactly as Ctrl-C always has.
    cleanup()
    os._exit(0)


if __name__ == "__main__":
    main()
