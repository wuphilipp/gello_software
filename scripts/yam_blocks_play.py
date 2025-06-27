from dataclasses import dataclass
import numpy as np
import tyro
from dm_control import composer, viewer

from gello.agents.gello_agent import DynamixelRobotConfig
from gello.dm_control_tasks.arms.yam import YAM
from gello.dm_control_tasks.manipulation.arenas.floors import Floor
from gello.dm_control_tasks.manipulation.tasks.block_play import BlockPlay


@dataclass
class Args:
    use_gello: bool = False
    port: str = ""


yam_config = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=[2*np.pi/2, 2*np.pi/2, 3*np.pi/2, 2*np.pi/2, 1*np.pi/2, 4*np.pi/2 ],
    joint_signs=(1, 1, -1, -1, 1, 1),
    gripper_config=(7, 24, -30),
)

reset_joints_sim = np.array([0,0,0,0,0,0,0])


def main(args: Args) -> None:
    arm_model = YAM()
    
    # The simulation environment expects 8 joints (6 arm + 2 gripper joints from MJCF)
    # Add a dummy value for the 8th joint for simulation
    reset_joints_sim_8 = np.append(reset_joints_sim, 0.0)  # Add 8th joint
    # reset_joints_sim_8 = reset_joints_sim
    # Use 8 joints for the simulation environment
    task = BlockPlay(arm_model, Floor(), reset_joints=reset_joints_sim_8)
    env = composer.Environment(task=task)

    gello = None
    if args.use_gello:
        
        port = args.port or "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMLV6-if00-port0"
        print(f"[INFO] Using GELLO hardware on port {port}")
        gello = yam_config.make_robot(
            port=port,
            start_joints=reset_joints_sim,  # Use all 7 joints for hardware
        )
 

    def policy(timestep) -> np.ndarray:
        if args.use_gello:
            joint_command = np.array(gello.get_joint_state(), dtype=np.float32)
            # Hardware has 7 joints, but simulation expects 8 joints
            # Add a dummy value for the 8th joint
            joint_command = np.append(joint_command, 0.0)
            return joint_command
        else:
            return np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)

    print("[INFO] Launching viewer loop")
    viewer.launch(env, policy=policy)

if __name__ == "__main__":
    main(tyro.cli(Args))