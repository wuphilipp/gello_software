import time
import numpy as np
import rtde_receive

r = rtde_receive.RTDEReceiveInterface("192.168.1.102")
print("connected:", r.isConnected())

t0 = r.getTimestamp()
try:
    while True:
        q = np.degrees(r.getActualQ())
        f = np.array(r.getActualTCPForce())
        print(f"t={r.getTimestamp() - t0:7.2f}s  "
              f"q[deg]={np.round(q, 1)}  "
              f"F[N]={np.round(f[:3], 1)}  T[Nm]={np.round(f[3:], 2)}")
        time.sleep(0.5)
except KeyboardInterrupt:
    r.disconnect()
