from typing import Dict, Optional, List
import numpy as np
from gello.robots.robot import Robot


class YAMRobot(Robot):
    """Implementation of a simulated YAM robot interface."""
    
    def __init__(self, no_gripper: bool = False):
        
        # Initialize state
        self._joint_state = np.zeros(self.num_dofs())
        self._joint_velocities = np.zeros(self.num_dofs())
        self._gripper_state = 0.0
        self._connected = True  
    
    def num_dofs(self) -> int:
        if self._use_gripper:
            return 7
        return 6
    
    def get_joint_state(self) -> np.ndarray:
        return self._joint_state
    
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self.num_dofs(), \
            f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"
            
        # Apply joint signs and offsets for simulation
        command = np.array([
            (j * s) + o for j, s, o in 
            zip(joint_state, self._joint_signs, self._joint_offsets)
        ])
        
        # Simple velocity calculation
        dt = 0.01  # Assume 100Hz control rate
        self._joint_velocities = (command - self._joint_state) / dt
        self._joint_state = command
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get robot state observations."""
        joint_state = self.get_joint_state()
        
        # Simple end effector pose (just zeros for now, could add FK later if needed)
        ee_pos_quat = np.zeros(7)  # [x,y,z, qx,qy,qz,qw]
        
        return {
            "joint_positions": joint_state,
            "joint_velocities": self._joint_velocities,
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": np.array(self._gripper_state),
        }

def main():
    robot = YAMRobot()
    import time

    time.sleep(1)
    print(robot.get_state())

    time.sleep(1)
    print(robot.get_state())
    print("end")
    robot.stop()


if __name__ == "__main__":
    main()