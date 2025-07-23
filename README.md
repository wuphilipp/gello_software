# GELLO: General, Low-Cost, and Intuitive Teleoperation Framework

<p align="center">
  <img src="imgs/title.png" />
</p>

GELLO is a teleoperation framework for robot manipulators that provides intuitive, low-cost control for research and development. This repository contains all the software components needed to build and operate your own GELLO system.

## Resources

- **Paper and Documentation**: https://wuphilipp.github.io/gello_site/
- **Hardware Instructions**: https://github.com/wuphilipp/gello_mechanical
- **ROS 2 Support**: See [ROS 2 README](ros2/README.md) for Franka FR3 integration

## Quick Start

```bash
git clone https://github.com/wuphilipp/gello_software.git
cd gello_software
```

## Installation

### Option 1: Virtual Environment (Recommended)

First, install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create environment and install dependencies:
```bash
uv venv --python 3.11
source .venv/bin/activate  # Run this every time you open a new shell
git submodule init
git submodule update
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -e third_party/DynamixelSDK/python
```

### Option 2: Docker

Install [Docker](https://docs.docker.com/engine/install/ubuntu/) on your host machine, then:

```bash
docker build . -t gello:latest
python scripts/launch.py
```

## Hardware Configuration

### 1. Update Motor IDs

Each Dynamixel motor needs a unique ID. Use the [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/) to configure them:

1. Connect motors one at a time to the U2D2 controller
2. Open Dynamixel Wizard and scan for the motor
3. Change the motor ID (starting from base motor in increasing order)
4. Repeat for each motor in your GELLO system

### 2. Configuration Systems

GELLO supports two configuration approaches:

#### Python Configuration (Core System)
- Located in `gello/agents/gello_agent.py`
- Uses `PORT_CONFIG_MAP` dictionary
- Maps USB serial ports to robot configurations

#### YAML Configuration (ROS 2 Integration)
- Used for ROS 2 packages
- Runtime configuration loading
- Located in `ros2/src/franka_gello_state_publisher/config/gello_config.yaml`

Example YAML configuration:
```yaml
usb-FTDI_USB__-__Serial_Converter_FT94EVW4-if00-port0:
  num_joints: 7
  joint_signs: [1, 1, 1, 1, 1, -1, 1]
  gripper: true
  best_offsets: [0, 1.57, 0, 0, 0, 0, 0]
  gripper_range_rad: [0, 1.2]
```

### 3. Joint Offset Calibration

After setting motor IDs, calibrate joint offsets by positioning GELLO in a known configuration:

<p align="center">
  <img src="imgs/gello_matching_joints.jpg" width="29%"/>
  <img src="imgs/robot_known_configuration.jpg" width="29%"/>
  <img src="imgs/fr3_gello_calib_pose.jpeg" width="31%"/>
  <img src="imgs/YAM_known_position.jpg" width="31%">
</p>

#### Supported Robot Configurations

**UR Robot:**
```bash
python scripts/gello_get_offset.py \
    --start-joints 0 -1.57 1.57 -1.57 -1.57 0 \
    --joint-signs 1 1 -1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6
```

**Franka FR3:**
```bash
python scripts/gello_get_offset.py \
    --start-joints 0 0 0 -1.57 0 1.57 0 \
    --joint-signs 1 1 1 1 1 -1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6
```

**I2RT YAM:**
```bash
python scripts/gello_get_offset.py \
    --start-joints 0 0 0 0 0 0 \
    --joint-signs 1 1 -1 -1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMLV6-if00-port0
```

#### Joint Signs Reference
- **UR**: `1 1 -1 1 1 1`
- **Panda**: `1 -1 1 1 1 -1 1`
- **FR3**: `1 1 1 1 1 -1 1`
- **xArm**: `1 1 1 1 1 1 1`
- **YAM**: `1 1 -1 -1 1 1`

**Finding Your Serial Port:**
- Ubuntu: `ls /dev/serial/by-id` (look for `usb-FTDI_USB__-__Serial_Converter`)
- Mac: Check `/dev/` for devices starting with `cu.usbserial`

After running the calibration script, add the generated `DynamixelRobotConfig` to the `PORT_CONFIG_MAP` in `gello/agents/gello_agent.py`.

## Usage

GELLO uses a multiprocess architecture with [ZMQ](https://zeromq.org/) for communication. No ROS required for basic operation.

### Testing with Simulation

First, test your GELLO configuration with a simulated robot:

```bash
# Terminal 1: Launch robot simulation
python experiments/launch_nodes.py --robot <sim_ur|sim_panda|sim_xarm|sim_yam>

# Terminal 2: Launch GELLO controller
python experiments/run_env.py --agent=gello
```

A Mujoco viewer will open showing the simulated robot responding to your GELLO movements.

### Real Robot Operation

Install the required robot-specific package:
- **UR**: [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html)
- **Panda**: [polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html)
- **xArm**: [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- **YAM**: [i2rt](https://github.com/i2rt-robotics/i2rt)

Launch the system:
```bash
# Terminal 1: Launch robot interface
python experiments/launch_nodes.py --robot=<your_robot>

# Terminal 2: Launch GELLO controller
python experiments/run_env.py --agent=gello
```

**YAM Robot Specific:**
```bash
python experiments/launch_nodes.py --robot=yam
python experiments/run_env.py --agent=gello --start-joints 0 0 0 0 0 0 1
```

### Optional: Known Starting Configuration

Use `--start-joints` to specify GELLO's starting configuration for automatic robot reset:
```bash
python experiments/run_env.py --agent=gello --start-joints <joint_angles>
```

## Data Collection

Collect teleoperation demonstrations with keyboard controls:
```bash
python experiments/run_env.py --agent=gello --use-save-interface
```

Process collected data:
```bash
python gello/data_utils/demo_to_gdict.py --source-dir=<source_dir>
```

## Advanced Features

### Bimanual Operation

GELLO supports bimanual robot control:
```bash
python experiments/launch_nodes.py --robot=bimanual_ur
python experiments/run_env.py --agent=gello --bimanual
```

### Process Management

Kill all Python processes if needed:
```bash
./kill_nodes.sh
```

## Adding New Robots

To integrate a new robot:

1. **Check Compatibility**: Ensure your GELLO kinematics match the target robot
2. **Implement Robot Interface**: Create a new class implementing the `Robot` protocol from `gello/robots/robot.py`
3. **Add Configuration**: Update the configuration system with your robot's parameters

See existing implementations in `gello/robots/` for reference:
- `panda.py` - Franka Panda robot
- `ur.py` - Universal Robots
- `xarm_robot.py` - xArm robots
- `yam.py` - YAM robot

## Development

### Code Organization

```
├── scripts/              # Utility scripts
├── experiments/          # Entry points and launch scripts
├── gello/               # Core GELLO package
│   ├── agents/          # Teleoperation agents
│   ├── cameras/         # Camera interfaces
│   ├── data_utils/      # Data processing utilities
│   ├── dm_control_tasks/# MuJoCo environment utilities
│   ├── dynamixel/       # Dynamixel hardware interface
│   ├── robots/          # Robot-specific interfaces
│   └── zmq_core/        # ZMQ multiprocessing utilities
```

### Development Environment

Install development dependencies:
```bash
uv pip install -r requirements_dev.txt
```

Set up pre-commit hooks for code formatting:
```bash
uv pip install pre-commit
pre-commit install
```

The codebase uses `isort` and `black` for code formatting.

### Contributing

We welcome contributions! Please submit pull requests to help make teleoperation more accessible and higher quality.

## Citation

```bibtex
@misc{wu2023gello,
    title={GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework for Robot Manipulators},
    author={Philipp Wu and Yide Shentu and Zhongke Yi and Xingyu Lin and Pieter Abbeel},
    year={2023},
}
```

## License & Acknowledgements

This project is licensed under the MIT License (see LICENSE file).

### Third-Party Dependencies
- [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie): Robot models for MuJoCo
- [brentyi/tyro](https://github.com/brentyi/tyro): Argument parsing and configuration
- [ZMQ](https://zeromq.org/): Multiprocessing communication framework