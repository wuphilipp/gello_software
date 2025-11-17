#!/usr/bin/env python3
import argparse, os, time
from typing import List, Tuple
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11

def canon(p: str) -> str:
    try: return os.path.realpath(os.path.normpath(p))
    except: return p

def parse_ids(spec: str) -> List[int]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok: continue
        if "-" in tok:
            a,b = map(int, tok.split("-",1))
            if a>b: a,b = b,a
            out.extend(range(a,b+1))
        else:
            out.append(int(tok))
    return sorted(set(out))

def write1(pkt, port, dxl_id: int, addr: int, val: int) -> Tuple[bool,int,int]:
    cr, err = pkt.write1ByteTxRx(port, dxl_id, addr, val)
    ok = (cr == COMM_SUCCESS and err == 0)
    print(f"[DXL] write1 id={dxl_id} addr={addr} val={val} -> comm={cr} err={err} ok={ok}")
    return ok, cr, err

def retry(times: int, fn, *a, sleep: float = 0.05, **kw) -> bool:
    for _ in range(times):
        ok, *_ = fn(*a, **kw)
        if ok: return True
        time.sleep(sleep)
    return False

def preflight_ping(pkt, ph, proto: float, dxl_id: int) -> bool:
    try:
        if proto >= 2.0:
            model, cr, err = pkt.ping(ph, dxl_id)
            print(f"[DXL] ping id={dxl_id} -> model={model} comm={cr} err={err}")
            return (cr == COMM_SUCCESS and err == 0)
        else:
            model, cr = pkt.ping(ph, dxl_id)
            print(f"[DXL] ping id={dxl_id} -> model={model} comm={cr}")
            return (cr == COMM_SUCCESS)
    except Exception as e:
        print(f"[DXL] ping exception: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Dynamixel ping + safe torque toggle with port flush and fallbacks.")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--proto", type=float, default=2.0, choices=[1.0, 2.0])
    ap.add_argument("--ids", default="1")
    ap.add_argument("--set-mode", type=int, default=None)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--fallback", action="store_true",
                    help="If initial ping fails, try quick fallback sweep over common bauds/protos.")
    args = ap.parse_args()

    port_path = canon(args.port)
    ids = parse_ids(args.ids)
    print(f"Port: {args.port} -> {port_path}\nBaud: {args.baud}  Proto: {args.proto}  IDs: {ids}  set-mode: {args.set_mode}")

    if not os.path.exists(port_path):
        raise SystemExit(f"[ERR] Port does not exist: {port_path}")

    ph = PortHandler(port_path)
    if not ph.openPort():
        raise SystemExit(f"[ERR] Failed to open {port_path}")

    # Clear stale bytes that can cause COMM_RX_CORRUPT
    try:
        ph.clearPort()
    except Exception:
        pass
    time.sleep(0.05)

    if not ph.setBaudRate(args.baud):
        ph.closePort()
        raise SystemExit(f"[ERR] Failed to set baud {args.baud} on {port_path}")

    pkt = PacketHandler(args.proto)

    print("\n-- Preflight ping --")
    ok_all = True
    for dxl_id in ids:
        ok = preflight_ping(pkt, ph, args.proto, dxl_id)
        ok_all &= ok

    # Optional fallback sweep if first ping failed (helps when another app changed baud/proto)
    if not ok_all and args.fallback:
        print("\n-- Fallback sweep (quick) --")
        bauds = [57600, 115200, 1000000, 2000000, 3000000, 4500000]
        protos = [args.proto] + ([1.0] if args.proto == 2.0 else [2.0])
        for b in bauds:
            if not ph.setBaudRate(b):
                continue
            for p in protos:
                pkt2 = PacketHandler(p)
                ok = preflight_ping(pkt2, ph, p, ids[0])
                if ok:
                    print(f"[OK] Found id {ids[0]} at baud {b}, proto {p}. Using that.")
                    args.baud, args.proto, pkt = b, p, pkt2
                    ok_all = True
                    break
            if ok_all: break
        if not ok_all:
            ph.closePort()
            raise SystemExit("[ERR] No response during fallback sweep. Check power/port conflicts.")

    if not ok_all:
        ph.closePort()
        raise SystemExit("[ERR] Preflight ping failed—likely port conflict or wrong baud/proto. See lines above.")

    # Optional: safely set operating mode
    if args.set_mode is not None:
        print(f"\n-- Set Operating Mode = {args.set_mode} --")
        for dxl_id in ids:
            if not retry(args.tries, write1, pkt, ph, dxl_id, ADDR_TORQUE_ENABLE, 0): 
                ph.closePort(); raise SystemExit(f"[ERR] disable torque id={dxl_id}")
        time.sleep(args.sleep)
        for dxl_id in ids:
            if not retry(args.tries, write1, pkt, ph, dxl_id, ADDR_OPERATING_MODE, int(args.set_mode)):
                ph.closePort(); raise SystemExit(f"[ERR] set mode id={dxl_id}")
        time.sleep(args.sleep)
        for dxl_id in ids:
            if not retry(args.tries, write1, pkt, ph, dxl_id, ADDR_TORQUE_ENABLE, 1):
                ph.closePort(); raise SystemExit(f"[ERR] re-enable torque id={dxl_id}")
        time.sleep(args.sleep)

    print("\n-- Torque toggle --")
    for state in (0, 1, 0):
        for dxl_id in ids:
            if not retry(args.tries, write1, pkt, ph, dxl_id, ADDR_TORQUE_ENABLE, state):
                ph.closePort(); raise SystemExit(f"[ERR] torque={state} id={dxl_id}")
        time.sleep(args.sleep)

    print("\nDone.")
    ph.closePort()

if __name__ == "__main__":
    main()
