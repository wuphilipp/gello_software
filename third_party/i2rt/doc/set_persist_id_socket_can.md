# Setting Persistent IDs for SocketCAN Devices

SocketCAN is easy to use, but the default device name is `can{idx}`, which can vary depending on the order in which the device is connected to the computer. Below is the procedure for setting up persistent IDs for these SocketCAN devices.

## For Canable Devices
Visit [Canable Updater](https://canable.io/updater/) to flash the firmware to candlelight to use with SocketCAN. YAM comes with pre-flashed candlelight firmware.

## Step 1: Find sysfd Paths for CAN Devices

```shell
$ ls -l /sys/class/net/can*
```

This should give you an output similar to:
```shell
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can0 -> ../../devices/platform/soc/your_can_device/can0
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can1 -> ../../devices/platform/soc/your_can_device/can1
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can2 -> ../../devices/platform/soc/your_can_device/can2
```

## Step 2: Use `udevadm` to Gather Attributes
```shell
udevadm info -a -p /sys/class/net/can0 | grep -i serial
```

## Step 3: Create udev Rules
edit `/etc/udev/rules.d/90-can.rules`
```shell
sudo vim /etc/udev/rules.d/90-can.rules
```
add
```
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="004E00275548501220373234", NAME="can_follow_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="0031005F5548501220373234", NAME="can_follow_r"
```

**Important:** The name should start with `can` (for USB-CAN adapters) or `en`/`eth` (for EtherCAT-CAN adapters). The maximum length for a CAN interface name is 13 characters.

## Step 4: Reload udev Rules
```shell
sudo udevadm control --reload-rules && sudo systemctl restart systemd-udevd && sudo udevadm trigger
```

Unplug and replug the CAN device to ensure the changes take effect.

Run the following command to set up the CAN device, and you need to run this command after every reboot.
```
sudo ip link set up can_right type can bitrate 1000000
```

## Step 5: Verify the CAN device
```shell
$ ip link show
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: enp5s0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT group default qlen 1000
    link/ether d8:43:ae:b7:43:0b brd ff:ff:ff:ff:ff:ff
5: tailscale0: <POINTOPOINT,MULTICAST,NOARP,UP,LOWER_UP> mtu 1280 qdisc fq_codel state UNKNOWN mode DEFAULT group default qlen 500
    link/none 
6: can_right: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can 
7: can_left: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can 
```

You should see that the CAN device is named `can_right`/`can_left`.
