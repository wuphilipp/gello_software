import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    # Leader arm configuration
    leader_agent: str = "gello"  # "gello", "quest", "spacemouse", "dummy"
    leader_robot_port: int = 6001
    leader_hostname: str = "127.0.0.1"
    leader_robot_type: str = None  # only needed for quest agent or spacemouse agent
    gello_port: Optional[str] = None
    start_joints: Optional[Tuple[float, ...]] = None
    
    # Follower YAM arm configuration
    follower_robot_port: int = 6002
    follower_hostname: str = "127.0.0.1"
    
    # General configuration
    hz: int = 100
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    verbose: bool = False
    
    # Leader-follower specific
    enable_follower: bool = True
    follower_gain: float = 1.0  # Scaling factor for follower motion
    max_follower_delta: float = 0.05  # Maximum joint delta per step for follower

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


class LeaderFollowerAgent:
    """Agent that reads from leader arm and commands follower arm."""
    
    def __init__(self, leader_agent, follower_robot, follower_gain=1.0, max_delta=0.05):
        self.leader_agent = leader_agent
        self.follower_robot = follower_robot
        self.follower_gain = follower_gain
        self.max_delta = max_delta
        
    def act(self, obs):
        # Get leader action (joint positions from Gello/other agent)
        leader_action = self.leader_agent.act(obs)
        
        # Apply gain and send to follower
        follower_action = leader_action * self.follower_gain
        
        # Command the follower robot
        self.follower_robot.command_joint_state(follower_action)
        
        # Return the leader action for the environment
        return leader_action


def main(args):
    print_color("Setting up Leader-Follower System", color="green", attrs=("bold",))
    
    # Setup leader robot client
    if args.mock:
        leader_robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {}
        leader_robot_client = ZMQClientRobot(port=args.leader_robot_port, host=args.leader_hostname)
    
    # Setup follower robot client
    if args.enable_follower:
        follower_robot_client = ZMQClientRobot(port=args.follower_robot_port, host=args.follower_hostname)
        print(f"Follower robot connected on port {args.follower_robot_port}")
    else:
        follower_robot_client = None
        print("Follower robot disabled")
    
    # Create environment (only for leader)
    env = RobotEnv(leader_robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    # Setup leader agent
    if args.leader_agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"Using leader Gello port: {gello_port}")
            else:
                raise ValueError(
                    "No gello port found, please specify one or plug in gello"
                )
        
        if args.start_joints is None:
            reset_joints = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        else:
            reset_joints = np.array(args.start_joints)
            
        leader_agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
        
        # Reset leader to starting position
        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)

            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)
                
    elif args.leader_agent == "quest":
        from gello.agents.quest_agent import SingleArmQuestAgent
        leader_agent = SingleArmQuestAgent(robot_type=args.leader_robot_type, which_hand="l")
        
    elif args.leader_agent == "spacemouse":
        from gello.agents.spacemouse_agent import SpacemouseAgent
        leader_agent = SpacemouseAgent(robot_type=args.leader_robot_type, verbose=args.verbose)
        
    elif args.leader_agent == "dummy" or args.leader_agent == "none":
        leader_agent = DummyAgent(num_dofs=leader_robot_client.num_dofs())
        
    else:
        raise ValueError("Invalid leader agent name")

    # Create leader-follower agent if follower is enabled
    if args.enable_follower and follower_robot_client is not None:
        agent = LeaderFollowerAgent(
            leader_agent=leader_agent,
            follower_robot=follower_robot_client,
            follower_gain=args.follower_gain,
            max_delta=args.max_follower_delta
        )
        print(f"Leader-follower agent created with gain: {args.follower_gain}")
    else:
        agent = leader_agent
        print("Using leader agent only")

    # Going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    # Smooth transition to start position
    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    if args.use_save_interface:
        from gello.data_utils.keyboard_interface import KBReset
        kb_interface = KBReset()

    print_color("\nStart Leader-Follower Control 🚀🚀🚀", color="green", attrs=("bold",))

    save_path = None
    start_time = time.time()
    while True:
        num = time.time() - start_time
        message = f"\rTime passed: {round(num, 2)}          "
        print_color(
            message,
            color="white",
            attrs=("bold",),
            end="",
            flush=True,
        )
        
        action = agent.act(obs)
        dt = datetime.datetime.now()
        
        if args.use_save_interface:
            state = kb_interface.update()
            if state == "start":
                dt_time = datetime.datetime.now()
                save_path = (
                    Path(args.data_dir).expanduser()
                    / f"{args.leader_agent}_leader_follower"
                    / dt_time.strftime("%m%d_%H%M%S")
                )
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {save_path}")
            elif state == "save":
                assert save_path is not None, "something went wrong"
                save_frame(save_path, dt, obs, action)
            elif state == "normal":
                save_path = None
            else:
                raise ValueError(f"Invalid state {state}")
                
        obs = env.step(action)


if __name__ == "__main__":
    main(tyro.cli(Args)) 