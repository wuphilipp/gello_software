from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
import signal
import sys

import tyro

from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQServerRobot


@dataclass
class Args:
    # Leader robot configuration
    leader_robot: str = "sim_ur"  # "sim_ur", "sim_panda", "sim_xarm", "ur", "panda", "xarm"
    leader_robot_port: int = 6001
    leader_hostname: str = "127.0.0.1"
    leader_robot_ip: str = "192.168.1.10"
    
    # Follower YAM robot configuration
    follower_robot: str = "sim_yam"  # "sim_yam", "yam"
    follower_robot_port: int = 6002
    follower_hostname: str = "127.0.0.1"
    follower_robot_ip: str = "192.168.1.11"
    
    # General configuration
    print_joints: bool = False


def launch_robot_server(robot_type: str, port: int, hostname: str, robot_ip: str, print_joints: bool = False):
    """Launch a robot server based on the robot type."""
    
    if robot_type == "sim_ur":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"
        gripper_xml = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, 
            gripper_xml_path=gripper_xml, 
            port=port, 
            host=hostname,
            print_joints=print_joints
        )
        return server
        
    elif robot_type == "sim_yam":
        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "i2rt_yam" / "yam.xml"
        from gello.robots.sim_robot import MujocoRobotServer

        server = MujocoRobotServer(
            xml_path=xml, 
            gripper_xml_path=None, 
            port=port, 
            host=hostname,
            print_joints=print_joints
        )
        return server
        
    elif robot_type == "sim_panda":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, 
            gripper_xml_path=gripper_xml, 
            port=port, 
            host=hostname,
            print_joints=print_joints
        )
        return server
        
    elif robot_type == "sim_xarm":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "third_party" / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm7" / "xarm7.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, 
            gripper_xml_path=gripper_xml, 
            port=port, 
            host=hostname,
            print_joints=print_joints
        )
        return server

    else:
        # Real robot implementations
        if robot_type == "xarm":
            from gello.robots.xarm_robot import XArmRobot
            robot = XArmRobot(ip=robot_ip)
        elif robot_type == "ur":
            from gello.robots.ur import URRobot
            robot = URRobot(robot_ip=robot_ip)
        elif robot_type == "panda":
            from gello.robots.panda import PandaRobot
            robot = PandaRobot(robot_ip=robot_ip)
        elif robot_type == "yam":
            from gello.robots.yam import YAMRobot
            robot = YAMRobot()
        elif robot_type == "none" or robot_type == "print":
            robot = PrintRobot(8)
        else:
            raise NotImplementedError(
                f"Robot {robot_type} not implemented, choose one of: sim_ur, sim_yam, sim_panda, sim_xarm, xarm, ur, panda, yam, none"
            )
        
        server = ZMQServerRobot(robot, port=port, host=hostname)
        print(f"Starting {robot_type} robot server on port {port}")
        return server


def main(args: Args):
    print(f"Launching Leader-Follower Robot Servers")
    print(f"Leader: {args.leader_robot} on port {args.leader_robot_port}")
    print(f"Follower: {args.follower_robot} on port {args.follower_robot_port}")
    
    # Launch leader robot server
    print(f"\nStarting leader robot server ({args.leader_robot})...")
    leader_server = launch_robot_server(
        args.leader_robot, 
        args.leader_robot_port, 
        args.leader_hostname, 
        args.leader_robot_ip,
        args.print_joints
    )
    
    # Launch follower robot server in a separate process
    print(f"\nStarting follower robot server ({args.follower_robot})...")
    follower_cmd = [
        sys.executable, 
        __file__, 
        "--follower-only",
        "--robot", args.follower_robot,
        "--port", str(args.follower_robot_port),
        "--hostname", args.follower_hostname,
        "--robot-ip", args.follower_robot_ip
    ]
    
    if args.print_joints:
        follower_cmd.append("--print-joints")
    
    follower_process = subprocess.Popen(follower_cmd)
    
    try:
        print(f"\nBoth servers started. Leader server starting...")
        leader_server.serve()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        follower_process.terminate()
        follower_process.wait()
        print("Servers shut down.")


if __name__ == "__main__":
    # Check if this is a follower-only launch
    if "--follower-only" in sys.argv:
        # Remove the follower-only flag and parse remaining args
        sys.argv.remove("--follower-only")
        args = tyro.cli(Args)
        
        # Launch only the follower server
        follower_server = launch_robot_server(
            args.follower_robot, 
            args.follower_robot_port, 
            args.follower_hostname, 
            args.follower_robot_ip,
            args.print_joints
        )
        print(f"Follower server starting on port {args.follower_robot_port}...")
        follower_server.serve()
    else:
        # Normal leader-follower launch
        main(tyro.cli(Args)) 