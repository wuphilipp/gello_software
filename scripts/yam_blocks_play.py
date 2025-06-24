from dataclasses import dataclass
import numpy as np
import tyro
from dm_control import composer, viewer

from gello.agents.gello_agent import DynamixelRobotConfig
from gello.dm_control_tasks.arms.yam import YAM
from gello.dm_control_tasks.manipulation.arenas.floors import Floor
from gello.dm_control_tasks.manipulation.tasks.block_play import BlockPlay
from gello.robots.yam import YAMRobot


@dataclass
class Args:
    use_gello: bool = False
    port: str = ""


yam_config = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=(3.142,6.283,4.712,6.283,3.142,6.283),
    joint_signs=(1, 1, -1, 1, 1, 1),
    gripper_config=None,
)

reset_joints_hw = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
reset_joints_sim = np.array([0, -1.57, 1.57, -1.57, -1.57, 0, 0, 0])


def main(args: Args) -> None:
    arm_model = YAM()  
    task = BlockPlay(arm_model, Floor(), reset_joints=reset_joints_sim)
    env = composer.Environment(task=task)

    gello = None
    if args.use_gello:
        port = args.port or "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMLV6-if00-port0"
        gello = yam_config.make_robot(
            port=port,
            start_joints=reset_joints_hw,
        )
        print(f"[INFO] Using GELLO hardware on port {port}")

    def policy(timestep):
        if args.use_gello:
            joint_state = np.array(gello.get_joint_state(), dtype=np.float32)
            print(joint_state)
            return joint_state[:6]
        else:
            return np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)

    print("[INFO] Launching manual viewer loop")
    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    main(tyro.cli(Args))
