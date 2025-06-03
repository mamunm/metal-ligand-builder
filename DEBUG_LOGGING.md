# Debug Logging Control

The ML-XTB Prescreening package uses a custom logger that provides detailed debug information during execution. You can control the verbosity of the output using several methods.

## Methods to Control Debug Logging

### 1. Environment Variable (Recommended)

Set the `ML_XTB_DEBUG` environment variable before running your script:

```bash
# Disable debug logging
export ML_XTB_DEBUG=0
python your_script.py

# Enable debug logging (default)
export ML_XTB_DEBUG=1
python your_script.py

# Or in a single command:
ML_XTB_DEBUG=0 python your_script.py
```

### 2. Programmatically in Your Script

Import the logger and control debug output directly:

```python
from ml_xtb_prescreening.core.logger import logger

# Disable debug logging
logger.disable_debug()

# Enable debug logging
logger.enable_debug()

# Or use set_debug
logger.set_debug(False)  # Disable
logger.set_debug(True)   # Enable
```

### 3. Example Usage

Here's how to use it in your analysis scripts:

```python
from ml_xtb_prescreening import MetalLigandComplex, ComplexConfig
from ml_xtb_prescreening.core.logger import logger

# Disable debug messages for cleaner output
logger.disable_debug()

# Your analysis code here
complex = MetalLigandComplex(...)
```

## What Gets Logged at Each Level

### Always Shown:
- INFO (green): Important status updates, workflow progress
- WARNING (orange): Non-critical issues, missing optional components
- ERROR (red): Critical failures, exceptions

### Debug Only (yellow):
- Detailed command execution (e.g., xTB commands with all parameters)
- Intermediate calculation details
- File paths and working directories
- Parsing details and warnings that succeeded

## Example Output Comparison

### With Debug Enabled (default):
```
[2025-06-02 22:31:11] (0.41s) (ml_xtb_prescreening.optimizers.xtb_optimizer:95) INFO  Running xTB command: /path/to/xtb input.xyz --opt normal --chrg 0 --gfn2
[2025-06-02 22:31:11] (0.41s) (ml_xtb_prescreening.optimizers.xtb_optimizer:96) DEBUG Working directory: /path/to/work_dir
[2025-06-02 22:31:11] (0.52s) (ml_xtb_prescreening.optimizers.xtb_optimizer:130) DEBUG xTB succeeded with warnings: Note: floating-point exceptions
```

### With Debug Disabled:
```
[2025-06-02 22:31:11] (0.41s) (ml_xtb_prescreening.optimizers.xtb_optimizer:95) INFO  Running xTB command: /path/to/xtb input.xyz --opt normal --chrg 0 --gfn2
```

## When to Disable Debug

Consider disabling debug logging when:
- Running production calculations
- Processing many structures in batch
- You only need to see the overall progress
- The detailed output is too verbose

Debug logging is useful when:
- Troubleshooting failures
- Understanding what commands are being executed
- Developing new features
- Learning how the package works