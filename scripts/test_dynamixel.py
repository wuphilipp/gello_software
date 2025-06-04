#!/usr/bin/env python3

from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.robotis_def import *

# Control table address
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
BROADCAST_ID = 254  # Broadcast ID
DEVICENAME = '/dev/ttyUSB0'
BAUDRATES = [57600, 1000000, 9600, 115200]  # Common baudrates
PROTOCOLS = [1.0, 2.0]  # Try both protocols

def scan_servos(port_handler, packet_handler):
    found = False
    print(f"\nScanning with Protocol {packet_handler.getProtocolVersion()}")
    for dxl_id in range(0, 253):
        dxl_model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            found = True
            print(f"Found Dynamixel ID {dxl_id} with model number: {dxl_model_number}")
    return found

def main():
    port_handler = PortHandler(DEVICENAME)
    if not port_handler.openPort():
        print("Failed to open port")
        return

    for baudrate in BAUDRATES:
        print(f"\nTrying baudrate: {baudrate}")
        if not port_handler.setBaudRate(baudrate):
            print("Failed to change baudrate")
            continue
        
        for protocol in PROTOCOLS:
            packet_handler = PacketHandler(protocol)
            scan_servos(port_handler, packet_handler)

    port_handler.closePort()
    print("\nIf no servos were found, please check:")
    print("1. Is the power supply connected and turned on?")
    print("2. Are all cables securely connected?")
    print("3. Is the USB adapter (U2D2 or similar) properly connected?")

if __name__ == '__main__':
    main() 