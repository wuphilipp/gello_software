"""Preflight check: are the cameras sharing a USB bus with the CAN adapters?

RealSense streams are isochronous, and isochronous transfers reserve up to 80%
of every USB frame by spec. The gs_usb CAN adapters are 12 Mbit/s full-speed
devices, which reach a high-speed bus through a hub's transaction translator --
and those split transactions are what an isochronous video reservation starves.
The symptom is "loss communication" on random motor IDs the moment the cameras
open, with all CAN error counters still reading zero because nothing actually
went wrong on the CAN wire.

Reducing video bandwidth does not reliably fix this. Moving the cameras onto a
different host controller does.

    python scripts/check_usb_topology.py
"""

import re
import sys
from pathlib import Path

USB_DEVICES = Path("/sys/bus/usb/devices")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def bus_of(devpath: Path) -> str:
    """Bus number is the part of a sysfs USB device name before the first '-'."""
    return devpath.name.split("-")[0]


def find_cameras():
    """RealSense cameras with their bus and negotiated USB generation.

    Uses librealsense rather than sysfs: a D4xx's USB descriptor iSerial is a
    different number from the serial librealsense reports (and the D435's is
    empty), so a sysfs serial cannot be matched against the camera serials in
    the config -- which is exactly what you need in order to know which camera
    to go and replug.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    out = []
    for dev in rs.context().query_devices():
        def info(field, default=""):
            try:
                return dev.get_info(field)
            except Exception:
                return default

        port = info(rs.camera_info.physical_port)
        # .../usb3/3-6/3-6.2/... -> bus 3
        m = re.search(r"/usb(\d+)/", port)
        bus = m.group(1) if m else "?"
        usb_gen = info(rs.camera_info.usb_type_descriptor, "?")
        out.append({
            "serial": info(rs.camera_info.serial_number),
            "product": info(rs.camera_info.name, "RealSense"),
            "bus": bus,
            "usb_gen": usb_gen,
            "superspeed": usb_gen.startswith("3"),
        })
    return sorted(out, key=lambda c: (c["bus"], c["serial"]))


def find_can():
    """CAN netdevs backed by USB, with the bus their adapter sits on."""
    out = []
    for net in sorted(Path("/sys/class/net").glob("*")):
        if not (net / "device").exists():
            continue
        try:
            dev = (net / "device").resolve()
        except OSError:
            continue
        # walk up to the USB device directory
        node = dev
        while node != node.parent and not (node / "idVendor").exists():
            node = node.parent
        if not (node / "idVendor").exists():
            continue
        driver = ""
        for iface in dev.parent.glob("*:*"):
            link = iface / "driver"
            if link.exists():
                driver = link.resolve().name
                break
        if not (net / "type").exists() or _read(net / "type") != "280":  # ARPHRD_CAN
            continue
        out.append({
            "name": net.name,
            "bus": bus_of(node),
            "speed": _read(node / "speed"),
            "driver": driver or _read(dev / "uevent").count("gs_usb") and "gs_usb",
            "path": node.name,
        })
    return out


def find_serial():
    """FTDI adapters (the GELLO leaders) -- same starvation applies to them."""
    out = []
    for dev in sorted(USB_DEVICES.glob("*-*")):
        if ":" in dev.name or _read(dev / "idVendor") != "0403":
            continue
        out.append({"product": _read(dev / "product") or "FTDI",
                    "bus": bus_of(dev), "speed": _read(dev / "speed"),
                    "path": dev.name})
    return out


def main() -> int:
    cams, cans, serials = find_cameras(), find_can(), find_serial()

    print("Cameras")
    for c in cams:
        flag = "" if c["superspeed"] else "   <-- USB 2 link, check the cable"
        print(f"  bus {c['bus']}  USB {c['usb_gen']:<4}  "
              f"{c['product']:<22} sn={c['serial']}{flag}")
    if not cams:
        print("  none found (is pyrealsense2 importable?)")

    print("\nCAN adapters")
    for c in cans:
        print(f"  bus {c['bus']}  {c['speed']:>5} Mb/s        {c['name']}")
    if not cans:
        print("  none found")

    print("\nSerial (GELLO leaders)")
    for s in serials:
        print(f"  bus {s['bus']}  {s['speed']:>5} Mb/s        {s['product']}")

    cam_buses = {c["bus"] for c in cams}
    can_buses = {c["bus"] for c in cans}
    ser_buses = {s["bus"] for s in serials}

    print()
    problems = []
    clash = cam_buses & can_buses
    if clash:
        names = [c["name"] for c in cans if c["bus"] in clash]
        problems.append(
            f"Cameras share bus {'/'.join(sorted(clash))} with CAN adapter(s) "
            f"{', '.join(names)}.\n"
            "  Isochronous video on that controller will starve the CAN\n"
            "  adapters' split transactions -> 'loss communication' on random\n"
            "  motor IDs the moment the cameras open. Move the camera(s) to a\n"
            "  USB3 port on a different controller."
        )
    if cam_buses & ser_buses:
        shared = "/".join(sorted(cam_buses & ser_buses))
        problems.append(
            f"Cameras share bus {shared} with the GELLO FTDI adapter(s); "
            "expect\n  intermittent 'comm failed' warnings from Dynamixel."
        )
    for c in cams:
        if not c["superspeed"]:
            problems.append(
                f"{c['product']} sn={c['serial']} negotiated USB {c['usb_gen']}.\n"
                "  Many USB-C cables carry only USB 2 pairs, so a camera in a\n"
                "  USB3 hub still lands on that hub's USB2 half -- and that half\n"
                "  is a different bus, shared with CAN. Use a known USB 3 cable."
            )

    if problems:
        print("PROBLEMS")
        for p in problems:
            print(f"- {p}")
        return 1

    print("OK - cameras are on their own controller(s), clear of CAN and serial.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
