# GELLO: General, Low-Cost, and Intuitive Teleoperation Framework

<p align="center">
  <img src="imgs/title.png" />
</p>


## Use your own virtual enviroment
## Create a vitual environment
First, install uv if you do not already have it installed
`curl -LsSf https://astral.sh/uv/install.sh | sh`

Create a uv virtual environment and use uv pip to install the necessary packages
```
uv venv --python 3.11
source .venv/bin/activate # run this every time you open a new shell
git submodule init
git submodule update
uv pip install -r requirements.txt
uv pip install -e .
uv pip install -e third_party/DynamixelSDK/python
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

## Use with ROS 2

> **Note:** GELLO also supports ROS 2 humble for the Franka FR3 robot. For more details, see the [ROS 2-specific README](ros2/README.md) located in the `ros2` directory.

# GELLO configuration setup (PLEASE READ)
Now that you have downloaded the code, there is some additional preparation work to properly configure the Dynamixels and GELLO.
These instructions will guide you on how to update the motor ids of the Dynamixels and then how to extract the joint offsets to configure your GELLO.

## Update motor IDs
Install the [dynamixel_wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/).
By default, each motor has the ID 1. In order for multiple dynamixels to be controlled by the same U2D2 controller board, each dynamixel must have a unique ID.
This process must be done one motor at a time. Connect each motor, starting from the base motor, and assign them in increasing order until you reach the gripper.

Steps:
 * Connect a single motor to the controller and connect the controller to the computer.
 * Open the dynamixel wizard
 * Click scan (found at the top left corner), this should detect the dynamixel. Connect to the motor
 * Look for the ID address and change the ID to the appropriate number.
 * Repeat for each motor
 * If some ID are missing, perform an ID scan

## Create the GELLO configuration and determining joint ID's
After the motor ID's are set, we can now connect to the GELLO controller device. However each motor has its own joint offset, which will result in a joint offset between GELLO and your actual robot arm.
Dynamixels have a symmetric 4 hole pattern which means there the joint offset is a multiple of pi/2.
The `GelloAgent` class  accepts a `DynamixelRobotConfig` (found in `gello/agents/gello_agent.py`). The Dynamixel config specifies the parameters you need to find to operate your GELLO. Look at the documentation for more details.

We have created a simple script to automatically detect the joint offset:
* set GELLO into a known configuration, where you know what the corresponding joint angles should be. For example, we set our GELLO for the UR and Franka FR3 in this configuration, where we know the desired ground truth joints (0, -90, 90, -90, -90, 0), or (0, 0, 0, -90, 0, 90 , 0) respectively. For the YAM the ground truth is in position (0, 0, 0, 0, 0, 0 , 0)
<p align="center">
  <img src="imgs/gello_matching_joints.jpg" width="29%"/>
  <img src="imgs/robot_known_configuration.jpg" width="29%"/>
  <img src="imgs/fr3_gello_calib_pose.jpeg" width="31%"/>
  <img src="imgs/YAM_known_position.jpg" width="31%">

</p>

* For the UR run 
```
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
# replace values with your own
```
* For the YAM run
```
python scripts/gello_get_offset.py \
    --start-joints 0 0 0 0 0 0 \ #in radians
    --joint-signs 1 1 -1 -1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMLV6-if00-port0
# replace values with your own
```
* Use the known starting joints for `start-joints`.
* Depending on the mechanical setup of your GELLO, the joint signs can flip, so you need to specify them for each axis.
* Use your serial port for `port`. You can find the port id of your U2D2 Dynamixel device by running `ls /dev/serial/by-id` and looking for the path that starts with `usb-FTDI_USB__-__Serial_Converter` (on Ubuntu). On Mac, look in /dev/ and the device that starts with `cu.usbserial`

`joint-signs` for each robot type:
* UR: `1 1 -1 1 1 1`
* Panda: `1 -1 1 1 1 -1 1`
* FR3: `1 1 1 1 1 -1 1`
* xArm: `1 1 1 1 1 1 1`
* YAM: `1 1 -1 -1 1 1`

The script prints out a list of joint offsets. Go to `gello/agents/gello_agent.py` and add a DynamixelRobotConfig to the PORT_CONFIG_MAP. You are now ready to run your GELLO!


# Using GELLO to control a robot!

The code provided here is simple and only relies on python packages. The code does NOT use ROS, but a ROS wrapper can easily be adapted from this code.
For multiprocessing, we leverage [ZMQ](https://zeromq.org/)

## Testing in sim
First test your GELLO with a simulated robot to make sure that the joint angles match as expected.
In one terminal run
```
python experiments/launch_nodes.py --robot <sim_ur, sim_panda, or sim_xarm, or sim_yam>
```
This launched the robot node. A simulated robot using the mujoco viewer should appear.

Then, launch your GELLO (the controller node).
```
python experiments/run_env.py --agent=gello
```
You should be able to use GELLO to control the simulated robot!

## Running on a real robot.
Once you have verified that your GELLO is properly configured, you can test it on a real robot!

Before you run with the real robot, you will have to install a robot specific python package.
The supported robots are in `gello/robots`.
 * UR: [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html)
 * panda: [polymetis](https://facebookresearch.github.io/fairo/polymetis/installation.html). If you use a different framework to control the panda, the code is easy to adopt. See/Modify `gello/robots/panda.py`
 * xArm: [xArm python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
 * YAM: [i2rt] (https://github.com/i2rt-robotics/i2rt) 

```
# Launch all of the node
python experiments/launch_nodes.py --robot=<your robot>
# run the environment loop
python experiments/run_env.py --agent=gello 

```

For the YAM append start joint position to run:
```
# Launch all of the node
python experiments/launch_nodes.py --robot=yam
# run the environment loop
# run the enviroment loop, the last start joint is 1 to set the gripper open
python experiments/run_env.py --agent=gello --start-joints 0 0 0 0 0 0 1

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

### Using a new robot!
If you want to use a new robot you need a GELLO that is compatible. If the kinemtics are close enough, you may directly use an existing GELLO. Otherwise you will have to design your own.
To add a new robot, simply implement the `Robot` protocol found in `gello/robots/robot`. See `gello/robots/panda.py`, `gello/robots/ur.py`, `gello/robots/xarm_robot.py`, `gello/robots/yam.py` for examples.

### Contributing
Please make a PR if you would like to contribute! The goal of this project is to enable more accessible and higher quality teleoperation devices and we would love your input!

You can optionally install some dev packages.
```
uv pip install -r requirements_dev.txt
```

The code is organized as follows:
 * `scripts`: contains some helpful python `scripts`
 * `experiments`: contains entrypoints into the gello code
 * `gello`: contains all of the `gello` python package code
    * `agents`: teleoperation agents
    * `cameras`: code to interface with camera hardware
    * `data_utils`: data processing utils. used for imitation learning
    * `dm_control_tasks`: dm_control utils to build a simple dm_control environment. used for demos
    * `dynamixel`: code to interface with the dynamixel hardware
    * `robots`: robot specific interfaces
    * `zmq_core`: zmq utilities for enabling a multi node system


This code base uses `isort` and `black` for code formatting.
pre-commits hooks are great. This will automatically do some checking/formatting. To use the pre-commit hooks, run the following:
```
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