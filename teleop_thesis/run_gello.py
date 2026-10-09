"""Registers this leader arm's calibration, then runs upstream run_env.py unchanged."""
import runpy
import numpy as np
from gello.agents.gello_agent import PORT_CONFIG_MAP, DynamixelRobotConfig

PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAK8B0I-if00-port0"

PORT_CONFIG_MAP[PORT] = DynamixelRobotConfig(
    joint_ids=(1, 2, 3, 4, 5, 6),
    joint_offsets=(0*np.pi/2, 0*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 0*np.pi/2),
    joint_signs=(1, 1, -1, 1, 1, 1),
    gripper_config=(7, 193, 147),
)

runpy.run_path("experiments/run_env.py", run_name="__main__")
