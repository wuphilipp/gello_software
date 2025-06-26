from dataclasses import dataclass
import numpy as np
import tyro
import time
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
        # Simulation mode - use dm_control viewer
        arm_model = YAM()
        
        # The simulation environment expects 8 joints (6 arm + 2 gripper joints from MJCF)
        # Add a dummy value for the 8th joint for simulation
        reset_joints_sim_8 = np.append(reset_joints_sim, 0.0)  # Add 8th joint
        
        # Use 8 joints for the simulation environment
        task = BlockPlay(arm_model, Floor(), reset_joints=reset_joints_sim_8)
        env = composer.Environment(task=task)

        def policy(timestep) -> np.ndarray:
            return np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)

        print("[INFO] Launching viewer loop")
        viewer.launch(env, policy=policy)


if __name__ == "__main__":
    main(tyro.cli(Args)) 