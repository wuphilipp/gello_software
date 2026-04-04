# GELLO Setup for Franka Panda

## Clone The Repo
```bash
git clone https://github.com/berkecyln/gello-franka-freiburg.git
cd gello-franka-freiburg
```

## Environment

```bash
conda create -n gello python=3.11 -y
conda activate gello
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

Each Dynamixel servo needs a unique ID (1–7 for arm joints, 8 for gripper). Do this **one motor at a time** with Dynamixel Wizard 2.0.

1. Install [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
2. Connect **only one motor** to the U2D2 controller at a time
3. Open Dynamixel Wizard, click **Scan** to detect the motor
4. In the control table, find **ID** and set it to the joint number (1 = base, 7 = wrist, 8 = gripper)
5. Click **Save**, the motor will reboot with the new ID
6. Disconnect that motor, connect the next one, repeat

## Calibration
**1. Match GELLO arm to the same pose by hand.**
<p align="center">
  <img src="imgs/fr3_gello_calib_pose.jpeg" />
</p>

**2. Run offset detection** (from `gello` env):

Check gello port with: `ls /dev/serial/by-id/` and replace if its different than below
```bash
python scripts/gello_get_offset.py \
    --start-joints 0 0 0 -1.5708 0 1.5708 0 \
    --joint-signs 1 1 1 -1 1 -1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0
```
> Run this everytime gello joint configuration is changed drastically. 

**3. Update offsets** in `gello/agents/gello_agent.py` → `PORT_CONFIG_MAP` (Franka Panda entry).

> If a joint moves in the wrong direction in simulation, flip its sign (1 → -1) and re-run step 3.

**4. Move Real Panda to the same calibration pose** (from `robotio` env):
```bash
conda activate robotio
cd ~/robot/robot_io
python robot_io/examples/move_to_gello_calib_pose.py
```
> Script: [`robot_io/examples/move_to_gello_calib_pose.py`](../robot_io/examples/move_to_gello_calib_pose.py)
> Target pose: `[0, 0, 0, -1.5708, 0, 1.5708, 0]` rad


## Calibrated Config

| Parameter | Value |
|-----------|-------|
| Port | `usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0` |
| Joint IDs | `(1, 2, 3, 4, 5, 6, 7)` |
| Joint signs | `(1, 1, 1, -1, 1, -1, 1)` |
| Joint offsets | `(4π/2, 1π/2, 4π/2, 1π/2, 2π/2, 3π/2, 1π/2)` |
| Gripper | motor 8, open=112°, close=71° |

> The above config can differ with different gello joint confirgurations.

## Simulation Test

Terminal 1 — MuJoCo sim:
```bash
python experiments/launch_nodes.py --robot sim_panda
```

Terminal 2 — GELLO controller:
```bash
python experiments/run_env.py \
    --agent gello \
    --gello-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0 \
    --start-joints 0 0 0 -1.5708 0 1.5708 0 0
```
