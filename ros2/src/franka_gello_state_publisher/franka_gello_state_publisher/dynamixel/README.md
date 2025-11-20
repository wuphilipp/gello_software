# Dynamixel Motor Configuration

This directory contains the configuration-driven Dynamixel driver that supports multiple motor types through YAML configuration files.

## Configuration File Structure

Motor configurations are stored in the `motor_configs/` subdirectory, with one YAML file per motor type. The filename (without extension) is used as the motor type identifier.

### File Structure:
```
franka_gello_state_publisher/franka_gello_state_publisher/dynamixel/
├── driver.py
├── motor_configs/
│   ├── xl330.yaml
│   ├── <add_yours>.yaml 
│   └── ...
└── README.md
```

### Configuration File Format

Each motor configuration file has the following structure:

```yaml
# Motor configuration for <motor model>
# Reference: https://example.com/motor/documentation

pulses_per_revolution: 4095  # Motor-specific resolution
control_table:
  parameter_name:
    addr: 64         # Control table address
    len: 1           # Data length in bytes
    min: 0           # Optional: minimum value
    max: 1           # Optional: maximum value
    unit: "1 [mA]"   # Optional: factor and unit (for reference)
    read_only: true  # Optional: if parameter is read-only
  # ... more parameters
eeprom_area_end: 63  # Last address of the EEPROM area. Adresses above are in RAM area.
operating_modes:
  0: "Mode Name"
  1: "Another Mode"
  # ... more modes
```

## Usage

### Regular Driver

```python
from gello.dynamixel.driver import DynamixelDriver
try:
  driver = DynamixelDriver(
      ids=[1, 2, 3], 
      port="/dev/ttyUSB0", 
      baudrate=57600, 
      motor_type="xl330"
  )
except ConnectionError as e:
  print(f"Error initializing DynamixelDriver: {e}")
  return
```

### Fake Driver for Testing

```python
from gello.dynamixel.driver import FakeDynamixelDriver

fake_driver = FakeDynamixelDriver(ids=[1, 2, 3], motor_type="xl330")
```

## Adding New Motor Types

To add support for a new motor type:

1. Create a new YAML file in `motor_configs/` directory
2. Name the file after your motor type (e.g., `my_motor.yaml`)
3. Follow the format shown above
4. Use the filename (without extension) as the motor type in your driver
