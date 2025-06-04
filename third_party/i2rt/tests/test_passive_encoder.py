import time

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    MotorType,
    PassiveEncoderReader,
    ReceiveMode,
)

motor_list = [
    [0x01, MotorType.DM4310],
    [0x02, MotorType.DM4310],
    [0x03, MotorType.DM4310],
    [0x04, MotorType.DM4310],
    [0x05, MotorType.DM3507],
    [0x06, MotorType.DM3507],
    [0x07, MotorType.DM3507],
]
motor_offset = [0, 0, 0, 0, 0, 0, 0]
motor_direction = [1, -1, 1, -1, 1, 1, 1]

channel = "can0"


def get_encoder_chain(can_interface: CanInterface) -> None:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


motor_chain = DMChainCanInterface(
    motor_list,
    motor_offset,
    motor_direction,
    channel,
    motor_chain_name="yam_real",
    receive_mode=ReceiveMode.p16,
    start_thread=True,
    get_same_bus_device_driver=get_encoder_chain,
    use_buffered_reader=True,
)


for _ in range(10000):
    motor_states = motor_chain.read_states()
    encoder_states = motor_chain.get_same_bus_device_states()
    print(motor_states)
    print(encoder_states)
    time.sleep(0.1)
