from dataclasses import dataclass
import numpy as np
import tyro
import time
from dm_control import composer, viewer
import glob

from gello.agents.gello_agent import DynamixelRobotConfig
from gello.dm_control_tasks.arms.yam import YAM
from gello.dm_control_tasks.manipulation.arenas.floors import Floor
from gello.dm_control_tasks.manipulation.tasks.block_play import BlockPlay
from gello.agents.gello_agent import GelloAgent


@dataclass
class Args:
    use_gello: bool = False
    port: str = ""


yam_config = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=[2*np.pi/2, 2*np.pi/2, 3*np.pi/2, 2*np.pi/2, 1*np.pi/2, 4*np.pi/2 ],
    joint_signs=(1, 1, -1, -1, 1, 1),
    gripper_config=(7, 20, -22),
)

reset_joints_sim = np.array([0,0,0,0,0,0,0])


def main(args: Args) -> None:
    print(f"[DEBUG] use_gello: {args.use_gello}")
    print(f"[DEBUG] port: {args.port}")
    
    if args.use_gello:
        print("[DEBUG] Entering hardware mode")
        port = args.port or "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMLV6-if00-port0"
        print(f"[INFO] Using GELLO hardware on port {port}")
        
        gello = yam_config.make_robot(
            port=port,
            start_joints=reset_joints_sim,
        )
        
        print("[INFO] Hardware mode - reading joint states (Ctrl+C to stop)")
        try:
            while True:
                joint_state = gello.get_joint_state()
                print(f"Joint state: {joint_state}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping hardware mode")
            
    else:
        print("[DEBUG] Entering simulation mode")
        arm_model = YAM()
        reset_joints_sim_8 = np.append(reset_joints_sim, 0.0)  # Add 8th joint if needed by your MJCF

        task = BlockPlay(arm_model, Floor(), reset_joints=reset_joints_sim_8)
        env = composer.Environment(task=task)
        action_space = env.action_spec()

        policy = None
        if args.use_gello:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"Using GELLO port: {gello_port}")
            else:
                raise ValueError("No GELLO port found, please plug in your GELLO device.")

            gello = yam_config.make_robot(port=gello_port, start_joints=reset_joints_sim)

            def policy(timestep):
                joint_command = gello.get_joint_state()
                joint_command = np.array(joint_command).copy()
                # If your sim expects 8 joints, pad or slice as needed
                if len(joint_command) < action_space.shape[0]:
                    joint_command = np.pad(joint_command, (0, action_space.shape[0] - len(joint_command)), 'constant')
                elif len(joint_command) > action_space.shape[0]:
                    joint_command = joint_command[:action_space.shape[0]]
                return joint_command
        else:
            def policy(timestep):
                return np.random.uniform(action_space.minimum, action_space.maximum)

        print("[INFO] Launching viewer loop")
        viewer.launch(env, policy=policy)


if __name__ == "__main__":
    main(tyro.cli(Args)) 