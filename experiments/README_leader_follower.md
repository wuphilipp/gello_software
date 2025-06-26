# Leader-Follower Robot Control System

This system allows you to control a follower YAM robot arm by moving a leader robot arm (controlled by Gello, Quest, SpaceMouse, or other agents).

## Overview

The leader-follower system consists of:
1. **Leader Robot**: Controlled by a teleoperation device (Gello, Quest, SpaceMouse, etc.)
2. **Follower Robot**: YAM robot that follows the leader's joint positions
3. **Leader-Follower Agent**: Coordinates between leader and follower

## Setup Instructions

### 1. Launch Robot Servers

First, launch both the leader and follower robot servers:

```bash
# For simulation (leader: UR5e, follower: YAM)
python experiments/launch_leader_follower.py \
    --leader-robot=sim_ur \
    --follower-robot=sim_yam \
    --leader-robot-port=6001 \
    --follower-robot-port=6002

# For real robots
python experiments/launch_leader_follower.py \
    --leader-robot=ur \
    --follower-robot=yam \
    --leader-robot-ip=192.168.1.10 \
    --follower-robot-ip=192.168.1.11 \
    --leader-robot-port=6001 \
    --follower-robot-port=6002
```

### 2. Run Leader-Follower Control

In a separate terminal, run the leader-follower control script:

```bash
# Basic usage with Gello as leader
python experiments/run_leader_follower.py \
    --leader-agent=gello \
    --leader-robot-port=6001 \
    --follower-robot-port=6002

# With custom gain and motion limits
python experiments/run_leader_follower.py \
    --leader-agent=gello \
    --leader-robot-port=6001 \
    --follower-robot-port=6002 \
    --follower-gain=0.5 \
    --max-follower-delta=0.02

# Using Quest as leader
python experiments/run_leader_follower.py \
    --leader-agent=quest \
    --leader-robot-type=sim_ur \
    --leader-robot-port=6001 \
    --follower-robot-port=6002

# Using SpaceMouse as leader
python experiments/run_leader_follower.py \
    --leader-agent=spacemouse \
    --leader-robot-type=sim_ur \
    --leader-robot-port=6001 \
    --follower-robot-port=6002
```

## Configuration Options

### Leader Agent Types
- `gello`: Use Gello device for teleoperation
- `quest`: Use Quest VR controller
- `spacemouse`: Use SpaceMouse 3D controller
- `dummy`: Use dummy agent (no input)

### Robot Types
- `sim_ur`: Simulated UR5e robot
- `sim_yam`: Simulated YAM robot
- `sim_panda`: Simulated Franka Panda
- `sim_xarm`: Simulated xArm
- `ur`: Real UR robot
- `yam`: Real YAM robot
- `panda`: Real Franka Panda
- `xarm`: Real xArm

### Leader-Follower Parameters
- `--follower-gain`: Scaling factor for follower motion (default: 1.0)
- `--max-follower-delta`: Maximum joint delta per step for follower (default: 0.05)
- `--enable-follower`: Enable/disable follower control (default: True)

## Example Use Cases

### 1. Teaching and Learning
Use a Gello device to demonstrate movements on a leader robot, while a follower YAM robot learns the same motions.

### 2. Scale Control
Use `--follower-gain=0.5` to make the follower move at half the scale of the leader.

### 3. Safety Control
Use `--max-follower-delta=0.02` to limit the maximum joint movement per step for safety.

### 4. Data Collection
Use `--use-save-interface` to collect demonstration data from the leader-follower system.

## Troubleshooting

### Port Conflicts
If you get port binding errors, make sure:
1. No other processes are using the specified ports
2. The ports are different for leader and follower
3. Firewall settings allow the connections

### Robot Connection Issues
For real robots:
1. Check network connectivity to robot IPs
2. Ensure robot controllers are powered on
3. Verify robot-specific dependencies are installed

### Gello Connection Issues
1. Check USB connection
2. Verify Gello port configuration in `gello/agents/gello_agent.py`
3. Ensure proper joint offsets and signs are configured

## Advanced Usage

### Custom Joint Mappings
You can modify the `LeaderFollowerAgent` class to implement custom joint mappings between leader and follower robots if they have different kinematics.

### Multiple Followers
Extend the system to support multiple follower robots by modifying the agent to command multiple robots simultaneously.

### Trajectory Recording
Use the `--use-save-interface` flag to record demonstrations for imitation learning applications.

## File Structure

- `experiments/run_leader_follower.py`: Main leader-follower control script
- `experiments/launch_leader_follower.py`: Robot server launcher
- `gello/agents/gello_agent.py`: Gello agent implementation
- `gello/robots/yam.py`: YAM robot implementation
- `gello/zmq_core/robot_node.py`: ZMQ communication utilities 