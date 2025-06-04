#!/usr/bin/env python3

from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.robotis_def import *

# Control table address
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
BROADCAST_ID = 254  # Broadcast ID
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
# Set the protocol version
# Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
packetHandler = PacketHandler(1.0)  # Using Protocol 1.0

def main():
    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        return

    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        return

    print("Scanning for Dynamixel servos...")
    for dxl_id in range(0, 253):  # Valid ID range
        # Try to ping the Dynamixel
        dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, dxl_id)
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            print(f"Found Dynamixel ID {dxl_id} with model number: {dxl_model_number}")

    # Close port
    portHandler.closePort()

if __name__ == '__main__':
    main() 