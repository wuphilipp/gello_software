# GELLO Setup for Franka Panda

## Environment

```bash
conda create -n gello python=3.11 -y
conda activate gello
cd ~/robot/gello_software
git submodule init && git submodule update
pip install -r requirements.txt
pip install -e .
pip install -e third_party/DynamixelSDK/python
```
## Prerequisites

Add yourself to the `dialout` group to access USB serial ports without `sudo`. **Do this once and reboot.**

```bash
sudo usermod -aG dialout $USER
```

## Set Unique Motor IDs

Each Dynamixel servo needs a unique ID (1–7 for arm joints, 8 for gripper). Do this **one motor at a time** with Dynamixel Wizard.

1. Install [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
2. Connect **only one motor** to the U2D2 controller at a time
3. Open Dynamixel Wizard, click **Scan** to detect the motor
4. In the control table, find **ID** and set it to the joint number (1 = base, 7 = wrist, 8 = gripper)
5. Click **Save**, the motor will reboot with the new ID
6. Disconnect that motor, connect the next one, repeat

## Calibration

**1. Move Panda to calibration pose** (from `robotio` env):
```bash
conda activate robotio
cd ~/robot/robot_io
python robot_io/examples/move_to_gello_calib_pose.py
```
> Script: [`robot_io/examples/move_to_gello_calib_pose.py`](../robot_io/examples/move_to_gello_calib_pose.py)
> Target pose: `[0, 0, 0, -1.5708, 0, 1.5708, 0]` rad

**2. Match GELLO arm to the same pose by hand.**

**3. Run offset detection** (from `gello` env):

Check gello port with: `ls /dev/serial/by-id/` and replace if its different than above
```bash
conda activate gello
cd ~/robot/gello_software
python scripts/gello_get_offset.py \
    --start-joints 0 0 0 -1.5708 0 1.5708 0 \
    --joint-signs 1 1 1 -1 1 -1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0
```

**4. Update offsets** in `gello/agents/gello_agent.py` → `PORT_CONFIG_MAP` (Franka Panda entry).

> If a joint moves in the wrong direction in simulation, flip its sign (1 → -1) and re-run step 3.

## Calibrated Config

| Parameter | Value |
|-----------|-------|
| Port | `usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0` |
| Joint IDs | `(1, 2, 3, 4, 5, 6, 7)` |
| Joint signs | `(1, 1, 1, -1, 1, -1, 1)` |
| Joint offsets | `(4π/2, 1π/2, 4π/2, 1π/2, 2π/2, 3π/2, 1π/2)` |
| Gripper | motor 8, open=112°, close=71° |

## Simulation Test

Terminal 1 — MuJoCo sim:
```bash
conda activate gello
cd ~/robot/gello_software
python experiments/launch_nodes.py --robot sim_panda
```

Terminal 2 — GELLO controller:
```bash
conda activate gello
cd ~/robot/gello_software
python experiments/run_env.py \
    --agent gello \
    --gello-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0 \
    --start-joints 0 0 0 -1.5708 0 1.5708 0 0
```
