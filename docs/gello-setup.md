# GELLO Setup for Franka Panda

---

## 1. One-Time System Setup

### Add yourself to the dialout group
Required to access USB serial ports without `sudo`. Do this once and reboot.

```bash
sudo usermod -aG dialout $USER
# reboot after this
```

### USB latency fix
Linux defaults to 16ms USB latency which causes Dynamixel communication failures.

**Permanent fix (run once):**
```bash
echo 'SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"' | \
  sudo tee /etc/udev/rules.d/99-dynamixel.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Verify:
```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
# Must print: 1
```

### Set unique motor IDs
Each Dynamixel servo needs a unique ID. Do this **one motor at a time** using Dynamixel Wizard.

1. Install [Dynamixel Wizard 2.0](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/)
2. Connect **one motor at a time** to the U2D2 controller
3. Open Dynamixel Wizard → click **Scan**
4. In the control table set **ID** to the joint number:
   - 1 = base, 2–6 = intermediate joints, 7 = wrist, 8 = gripper
5. Click **Save** (motor reboots), disconnect, repeat for next motor

---

## 2. Environment

```bash
conda create -n gello python=3.11 -y
conda activate gello
cd ~/robot/gello_software
git submodule init && git submodule update
pip install -r requirements.txt
pip install -e .
pip install -e third_party/DynamixelSDK/python
```

---

## 3. Find Your Port

```bash
ls /dev/serial/by-id/ | grep -i ftdi
```

Current port: `usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0`

---

## 4. Calibration

Maps each GELLO servo to the matching robot joint angle. Redo if you reassemble the arm
or change any servo horn position.

**Step 1 — Move robot to calibration pose** (from `robotio` env, see robot_io docs):
```bash
conda activate robotio
cd ~/robot/robot_io
python robot_io/examples/move_to_gello_calib_pose.py
```
Target: `[0, 0, 0, -1.5708, 0, 1.5708, 0.7854]` rad — gripper in natural forward orientation.

**Step 2 — Physically match GELLO to robot pose.** Joint 7 (wrist) is critical — the
gripper head must point in the same direction as the robot gripper.

**Step 3 — Compute offsets:**
```bash
conda activate robotio
cd ~/robot/gello_software
python scripts/gello_get_offset.py \
  --port /dev/ttyUSB0 \
  --start-joints 0 0 0 -1.5708 0 1.5708 0.7854 \
  --joint-signs 1 1 1 -1 1 -1 1 \
  --gripper
```

**Step 4 — Update `PORT_CONFIG_MAP`** in `gello/agents/gello_agent.py`, Franka entry:
```python
joint_offsets=(X * np.pi / 2, ...),   # from script output
gripper_config=(8, open_deg, close_deg),
```

---

## 5. Calibrated Config (current values)

| Parameter | Value |
|-----------|-------|
| Port | `usb-FTDI_USB__-__Serial_Converter_FTAO4UAS-if00-port0` |
| Joint IDs | `(1, 2, 3, 4, 5, 6, 7)` |
| Joint signs | `(1, 1, 1, -1, 1, -1, 1)` |
| Joint offsets | `(0, 1, 0, 1, 2, 3, 3) × π/2` |
| Gripper | motor 8, open=113°, close=71° |

---

## 6. Simulation Test

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
  --start-joints 0 0 0 -1.5708 0 1.5708 0.7854 0
```
