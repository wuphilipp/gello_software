# GELLO ROS 2 humble integration

This folder contains all required ROS 2 packages for using GELLO. 

## Packages

### 1. `franka_fr3_arm_controllers`
This package provides a Joint Impedance controller for the Franka FR3. It subscribes to the GELLO joint states and sends torque commands to the robot.

#### Key Features:
- Implements a `JointImpedanceController` for controlling the robot's torques.
- Subscribes to `/gello/joint_states` topic for the GELLO joint states.

#### Launch Files:
- **`franka.launch.py`**: Launches the Franka robot ros interfaces.
- **`franka_fr3_arm_controllers.launch.py`**: Launches the Joint Impedance controller.

### 2. `franka_gello_state_publisher`
This package provides a ROS 2 node that reads input from the GELLO and publishes it as `sensor_msgs/msg/JointState` messages.

#### Key Features:
- Publishes GELLO state to the `/gello/joint_states` topic.

#### Launch Files:
- **`main.launch.py`**: Launches the GELLO publisher node.

### 3. `franka_gripper_manager`
This package provides a ROS 2 node for managing the gripper connected to the Franka robot. Supported grippers are either the `Franka Hand` or the `Robotiq 2F-85`. It allows sending commands to control the gripper's width and perform homing actions. 

#### Key Features:
- Subscribes to `/gripper_client/target_gripper_width_percent` for gripper width commands.
- Supports homing and move actions for the gripper.

#### Launch Files:
- **`franka_gripper_client.launch.py`**: Launches the gripper manager node for the `Franka Hand`.
- **`robotiq_gripper_controller_client.launch.py`**: Launches the gripper manager node for the `Robotiq 2F-85`.

## VS-Code Dev-Container

We recommend working inside the provided VS-Code Dev-Container for a seamless development experience. Dev-Containers allow you to use a consistent environment with all necessary dependencies pre-installed. 

To start the Dev-Container, open the `ros2` sub-folder of this repository (not the entire `gello_software` folder) in VS Code. If prompted, select **"Reopen in Container"** to launch the workspace inside the Dev-Container. If you are not prompted, open the Command Palette (`Ctrl+Shift+P`) and select **"Dev Containers: Reopen in Container"**. Building the container for the first time will take a few minutes. For more information, refer to the [VS-Code Dev-Containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).

If you choose not to use the Dev-Container, please refer to the [Local Setup](#local-setup) section below for manual installation instructions.

## Local Setup

### Prerequisites

- **ROS 2 Humble Desktop** must be installed.  
  See the [official installation guide](https://docs.ros.org/en/humble/Installation.html) for instructions.
- **libfranka** and **franka_ros2** must be installed.  
  Refer to the [Franka Robotics documentation](https://frankaemika.github.io/docs/index.html) for installation steps and compatibility information.
- **ros2_robotiq_gripper** (if required) must be installed.  
  See the [ros2_robotiq_gripper GitHub repository](https://github.com/PickNikRobotics/ros2_robotiq_gripper) for installation and usage instructions.

> 💡 **Hint:**  
> You can also find example installation commands for `libfranka`, `franka_ros2`, and `ros2_robotiq_gripper` in the [Dockerfile](./.devcontainer/Dockerfile) located in the `ros2/.devcontainer` directory. These commands can be copy-pasted for your local setup.

### Further Dependency Installations

After installing the prerequisites, you may need to install additional dependencies required by this workspace. For this you can run the `install_workspace_dependencies.bash` script.

If you add new dependencies to your packages, remember to update the relevant `requirements.txt`, `requirements_dev.txt` or `package.xml` files and re-run the script.

## Build and Test

### Building the Project

To build the project, use the following `colcon` command with CMake arguments, required for clang-tidy:

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCHECK_TIDY=ON
```

### Testing the Project

The packages come with a set of tests, which can be executed using the following command:

```bash
colcon test 
```

> ⚠️ **Important:**  
> All commands for building and testing must be executed from the `ros2` directory of this repository.

## Getting Started

### 1. **Run the GELLO Publisher**  
#### Step 1: Determine your GELLO USB ID
      
To proceed, you need to know the USB ID of your GELLO device. This can be determined by running:

```bash
ls /dev/serial/by-id
```

Example output:

```bash
usb-FTDI_USB__-__Serial_Converter_FT7WBG6
```

In this case, the `GELLO_USB_ID` would be `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6`.

#### Step 2: Configure your GELLO 
      
If not done already, follow the instructions of the `Create the GELLO configuration and determining joint ID's` section in the main README.md. Use the provided script to configure the GELLO for Franka FR3:

```bash
python3 scripts/gello_get_offset.py \
--start-joints 0 0 0 -1.57 0 1.57 0 \
--joint-signs 1 1 1 1 1 -1 1 \
--port /dev/serial/by-id/<GELLO_USB_ID>
```
      
To apply your configuration:
- Update the `/workspace/src/franka_gello_state_publisher/config/gello_config.yaml` file.
- Rebuild the project to ensure the updated configuration is applied:

```bash
colcon build
```

#### Step 3: Launch the GELLO publisher:  
Launch the GELLO publisher node with the appropriate `com_port` parameter:

```bash
ros2 launch franka_gello_state_publisher main.launch.py com_port:=/dev/serial/by-id/<GELLO_USB_ID>
```

### 2. **Launch the Joint Impedance Controller**  
   
   Launch the controller to send torque commands to the Franka robot:  
   ```bash
   ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_ip:=<robot-ip> load_gripper:=<true_or_false>
   ```

   - `robot_ip:` Replace `<robot-ip>` with the IP address of your Franka robot.
   - `load_gripper`: A boolean parameter (true or false) that specifies whether to load the Franka Hand:
      - Set load_gripper:=true if you are using the Franka Hand.
      - Set load_gripper:=false if you are not using the Franka Hand.
  
### 3. **Launch the Gripper Manager**
   
   To control the gripper, use the appropriate launch file based on the gripper type:

   - **For the Franka Hand**:  
     ```bash
     ros2 launch franka_gripper_manager franka_gripper_client.launch.py
     ```

   - **For the Robotiq 2F-85**:  
     ```bash
     ros2 launch franka_gripper_manager robotiq_gripper_controller_client.launch.py com_port:=/dev/serial/by-id/<ROBOTIQ_USB_ID>
     ```

     The `ROBOTIQ_USB_ID` can be determined by `ls /dev/serial/by-id`.


## Troubleshooting

### SerialException(msg.errno, "could not open port {}: {}".format(self._port, msg))

The open com port could not be opened. Possible reasons are:
- Wrongly specified, the full path is required, such as: `/dev/serial/by-id/usb-FTDI_***`
- The device was plugged in after the docker container started, re-open the container
  
### libfranka: Incompatible library version (server version: X, library version: Y).

The libfranka version and robot system version are not compatible. More information can be found [here](https://frankaemika.github.io/docs/compatibility.html).
Fix this by correcting the `LIBFRANKA_VERSION=0.15.0` in the [Dockerfile](./.devcontainer/Dockerfile) and update the `FRANKA_ROS2_VERSION` and `FRANKA_DESCRIPTION_VERSION` accordingly.


## Acknowledgements
The source code for the Robotiq gripper control is based on
[ros2_robotiq_gripper](https://github.com/PickNikRobotics/ros2_robotiq_gripper.git), licensed under the BSD 3-Clause license.

