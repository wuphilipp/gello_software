import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import tyro
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS

# Constants
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4


@dataclass
class Args:
    port: str = (
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAK8B0I-if00-port0"
    )
    """The port that GELLO is connected to."""

    baud: int = 57600
    """The baudrate of the dynamixel bus."""

    ids: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    """The dynamixel ids to read, gripper included."""

    duration: float = 10.0
    """How long to read for (in seconds)."""

    label: str = ""
    """Optional tag added to the output file name."""

    outdir: str = "teleop_thesis/data"
    """Where the raw cycle times are saved."""


def get_latency_timer(port: str) -> str:
    tty = os.path.basename(os.path.realpath(port))
    path = f"/sys/bus/usb-serial/devices/{tty}/latency_timer"
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def run_benchmark(args: Args) -> None:
    latency = get_latency_timer(args.port)
    print("port          : ", args.port)
    print("baud          : ", args.baud)
    print("ids           : ", list(args.ids))
    print("latency timer : ", latency, "ms")
    if latency != "1":
        print("WARNING: latency timer is not 1 ms, this will limit the rate")

    port_handler = PortHandler(args.port)
    packet_handler = PacketHandler(2.0)
    if not port_handler.openPort():
        raise RuntimeError("Failed to open the port")
    if not port_handler.setBaudRate(args.baud):
        raise RuntimeError(f"Failed to change the baudrate, {args.baud}")

    # read only, torque is never enabled and nothing is written to the motors
    group_sync_read = GroupSyncRead(
        port_handler, packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
    )
    for dxl_id in args.ids:
        if not group_sync_read.addParam(dxl_id):
            raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")

    periods = []
    num_failed = 0
    end_time = time.perf_counter() + args.duration
    try:
        while time.perf_counter() < end_time:
            start = time.perf_counter()
            dxl_comm_result = group_sync_read.txRxPacket()
            period = time.perf_counter() - start
            # only count cycles where every motor answered
            if dxl_comm_result != COMM_SUCCESS or not all(
                group_sync_read.isAvailable(
                    dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                )
                for dxl_id in args.ids
            ):
                num_failed += 1
                continue
            periods.append(period)
    finally:
        port_handler.closePort()

    if len(periods) == 0:
        raise RuntimeError(f"No successful reads, {num_failed} failed")

    periods_ms = np.array(periods) * 1000
    print()
    print("successful cycles : ", len(periods_ms))
    print("failed cycles     : ", num_failed)
    print("mean rate (Hz)    : ", f"{1000 / periods_ms.mean():.1f}")
    print("period mean (ms)  : ", f"{periods_ms.mean():.3f}")
    print("period median (ms): ", f"{np.median(periods_ms):.3f}")
    print("period std (ms)   : ", f"{periods_ms.std():.3f}")
    print("period p99 (ms)   : ", f"{np.percentile(periods_ms, 99):.3f}")
    print("period max (ms)   : ", f"{periods_ms.max():.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"_{args.label}" if args.label else ""
    path = os.path.join(
        args.outdir, f"busrate_{args.baud}_lat{latency}{label}_{stamp}.csv"
    )
    with open(path, "w", newline="") as f:
        f.write(
            f"# baud={args.baud} latency_ms={latency} ids={list(args.ids)} "
            f"failed={num_failed} duration_s={args.duration}\n"
        )
        writer = csv.writer(f)
        writer.writerow(["cycle", "period_ms"])
        for i, p in enumerate(periods_ms):
            writer.writerow([i, f"{p:.4f}"])
    print()
    print("saved to          : ", path)


def main(args: Args) -> None:
    run_benchmark(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
