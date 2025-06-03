# ML-xTB-Prescreening

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular Python package for metal-ligand binding energy analysis using xTB and preparation for ORCA DFT calculations.

## Overview

ML-xTB-Prescreening provides a streamlined workflow for:
1. Generating metal-ligand complex structures
2. Optimizing geometries with xTB
3. Calculating binding energies
4. Preparing ORCA input files for high-level DFT calculations

The package focuses on single metal-ligand pair analysis with a clean, modular architecture.

## Features

- **Smart Binding Site Detection**: Automatically identifies carboxylates, amines, hydroxyls, and other coordinating groups
- **Enhanced Pose Generation**: Creates metal-ligand complexes with force field optimization and conformational sampling
  - Uses RDKit or OpenBabel for initial pose refinement
  - Generates multiple conformations for each pose
  - Maintains metal coordination during optimization
- **Conformer Generation**: Explores ligand conformational space
- **XTB Optimization**: Fast semi-empirical quantum mechanical optimization
- **Binding Energy Calculation**: Computes metal-ligand binding energies
- **ORCA Input Preparation**: Generates ready-to-run DFT input files

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Open Babel** (for SMILES to 3D conversion):
   ```bash
   # macOS
   brew install open-babel
   
   # Linux
   sudo apt-get install openbabel
   
   # Conda
   conda install -c conda-forge openbabel
   ```

3. **xTB** (for quantum mechanical calculations):
   ```bash
   conda install -c conda-forge xtb
   ```

### Package Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-xtb-prescreening.git
cd ml-xtb-prescreening

# Install with pip
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

```python
from ml_xtb_prescreening import MetalLigandComplex

# Create a metal-ligand complex analyzer
complex = MetalLigandComplex(
    ligand_name="edta",
    ligand_smiles="C(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
    metal_symbol="Co",
    metal_charge=2,
    ligand_charge=-4,
    ligand_protonation_state="deprotonated",
    experiment_name="co_edta_binding"
)

# Run the complete workflow
results = complex.run_workflow()

# Results include:
# - Optimized structures
# - Binding energies
# - ORCA input files for best structures
```

## Workflow Steps

1. **Generate Ligand Conformers**: Creates multiple 3D conformations of the ligand
2. **Generate Metal Geometries**: Prepares metal ion structures
3. **Detect Binding Sites**: Identifies potential coordination sites on the ligand
4. **Generate Complex Poses**: Creates metal-ligand arrangements based on coordination chemistry
5. **XTB Optimization**: Optimizes all structures using semi-empirical QM
6. **Calculate Binding Energies**: Computes E_binding = E_complex - E_metal - E_ligand
7. **Rank Structures**: Orders complexes by binding energy
8. **Prepare ORCA Inputs**: Creates DFT input files for further refinement

## Configuration

```python
from ml_xtb_prescreening import ComplexConfig, XTBConfig, ORCAConfig

# Customize the analysis
config = ComplexConfig(
    max_poses_per_conformer=25,  # Refined poses per ligand conformer
    n_conformers=10,             # Number of ligand conformers
    optimize_poses_with_ff=True, # Enable force field optimization
    ff_method="auto",            # Auto-select RDKit or OpenBabel
    energy_window=10.0,          # Energy window in kcal/mol
    xtb_config=XTBConfig(
        method="gfn2",
        solvent="water"
    ),
    orca_config=ORCAConfig(
        method="B3LYP",
        basis_set="def2-SVP",
        dispersion="D3BJ"
    )
)

# Use custom configuration
complex = MetalLigandComplex(..., config=config)
```

## Architecture

The package follows a modular design:

- **`core/`**: Main classes and data models
- **`generators/`**: Structure generation (binding sites, poses, conformers)
- **`optimizers/`**: XTB and conformer optimization
- **`analysis/`**: Energy analysis and structure ranking
- **`io/`**: File handling and ORCA input generation

## Output Structure

```
metal_ligand_prescreening/
├── experiment_name/
│   ├── ligand_conformers/      # Generated conformers
│   ├── metal_structures/       # Metal ion structures
│   ├── complex_poses/          # Initial metal-ligand poses
│   ├── optimized_structures/   # XTB-optimized geometries
│   ├── binding_energies.csv    # Calculated binding energies
│   ├── rankings.csv            # Structures ranked by energy
│   └── orca_inputs/           # ORCA input files
```

## Example: Co(II)-EDTA Complex Analysis

A complete example analyzing Co(II)-EDTA binding is provided:

```bash
cd examples
python co_edta_analysis.py
```

This example demonstrates:
- Generating EDTA conformers and Co-EDTA complex poses
- Running XTB geometry optimizations
- Organizing results in a clear directory structure
- Preparing for binding energy calculations

The script analyzes the hexadentate chelation of EDTA with Co(II), creating an octahedral complex.

## Testing

Run tests using pytest with uv:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ml_xtb_prescreening --cov-report=html

# Run specific test file
uv run pytest tests/test_metal_ligand_complex.py

# Run fast tests only (skip slow tests)
uv run pytest -m "not slow"

# Skip tests requiring external dependencies
uv run pytest -m "not requires_obabel and not requires_xtb"

# Run with verbose output
uv run pytest -v
```

## Debug Logging Control

The ML-XTB Prescreening package uses a custom logger that provides detailed debug information during execution. You can control the verbosity of the output using several methods.

### Methods to Control Debug Logging

#### 1. Environment Variable (Recommended)

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

#### 2. Programmatically in Your Script

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

#### 3. Example Usage

Here's how to use it in your analysis scripts:

```python
from ml_xtb_prescreening import MetalLigandComplex, ComplexConfig
from ml_xtb_prescreening.core.logger import logger

# Disable debug messages for cleaner output
logger.disable_debug()

# Your analysis code here
complex = MetalLigandComplex(...)
```

### What Gets Logged at Each Level

#### Always Shown:
- INFO (green): Important status updates, workflow progress
- WARNING (orange): Non-critical issues, missing optional components
- ERROR (red): Critical failures, exceptions

#### Debug Only (yellow):
- Detailed command execution (e.g., xTB commands with all parameters)
- Intermediate calculation details
- File paths and working directories
- Parsing details and warnings that succeeded

### Example Output Comparison

#### With Debug Enabled (default):
```
[2025-06-02 22:31:11] (0.41s) (ml_xtb_prescreening.optimizers.xtb_optimizer:95) INFO  Running xTB command: /path/to/xtb input.xyz --opt normal --chrg 0 --gfn2
[2025-06-02 22:31:11] (0.41s) (ml_xtb_prescreening.optimizers.xtb_optimizer:96) DEBUG Working directory: /path/to/work_dir
[2025-06-02 22:31:11] (0.52s) (ml_xtb_prescreening.optimizers.xtb_optimizer:130) DEBUG xTB succeeded with warnings: Note: floating-point exceptions
```

#### With Debug Disabled:
```
[2025-06-02 22:31:11] (0.41s) (ml_xtb_prescreening.optimizers.xtb_optimizer:95) INFO  Running xTB command: /path/to/xtb input.xyz --opt normal --chrg 0 --gfn2
```

### When to Disable Debug

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Osman Mamun**  
Email: mamun.che06@gmail.com