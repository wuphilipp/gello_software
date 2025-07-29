# Control Utilities

This module provides shared utilities for robot control loops, eliminating code duplication between `launch_yaml.py` and `run_env.py`.

## Components

### `move_to_start_position(env, agent, max_delta=1.0, steps=25)`

Gradually moves the robot to its starting position to avoid sudden movements.

**Returns:** `bool` - `True` if successful, `False` if position is too far away

### `SaveInterface(data_dir, agent_name, expand_user=False)`

Handles keyboard-based data collection interface.

**Keyboard Controls:**
- **S**: Start recording
- **Q**: Stop recording

**Parameters:**
- `data_dir`: Base directory for saving data
- `agent_name`: Agent name (used for subdirectory)  
- `expand_user`: Whether to expand `~` in paths

### `run_control_loop(env, agent, save_interface=None, print_timing=True, use_colors=False)`

Main control loop that handles:
- Observation collection
- Action execution  
- Data saving (if save_interface provided)
- Timing display
- Colored output (if use_colors=True)

## Usage Examples

### Basic Control Loop
```python
from gello.utils.control_utils import move_to_start_position, run_control_loop

# Move to start position
if not move_to_start_position(env, agent):
    print("Robot too far from start position!")
    return

# Run control loop
run_control_loop(env, agent)
```

### With Data Collection
```python
from gello.utils.control_utils import SaveInterface, run_control_loop

# Initialize save interface
save_interface = SaveInterface(
    data_dir="~/my_data",
    agent_name="GelloAgent", 
    expand_user=True
)

# Run with data collection
run_control_loop(env, agent, save_interface, use_colors=True)
```

## Migration

### Before (Duplicated Code)
```python
# In launch_yaml.py and run_env.py:
# 50+ lines of identical control loop code
# 30+ lines of identical save interface code
# 40+ lines of identical start position code
```

### After (Shared Utilities)
```python
# In both files:
from gello.utils.control_utils import move_to_start_position, SaveInterface, run_control_loop

if not move_to_start_position(env, agent):
    return

save_interface = SaveInterface(...) if args.use_save_interface else None
run_control_loop(env, agent, save_interface)
```

## Benefits

- **DRY Principle**: Eliminates ~120 lines of duplicated code
- **Consistency**: Same behavior across all control scripts
- **Maintainability**: Bug fixes and improvements in one place
- **Testability**: Shared utilities can be unit tested
- **Flexibility**: Easy to add new control scripts without duplication 