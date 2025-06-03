# ML-xTB-Prescreening

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for automated metal-ligand binding analysis using semi-empirical quantum mechanics (xTB) with seamless preparation for high-level DFT calculations in ORCA.

## Overview

ML-xTB-Prescreening provides a comprehensive computational pipeline for metal-ligand complex analysis:

1. **Structure Generation**: Intelligent pose generation with conformational sampling
2. **Force Field Optimization**: Pre-optimization using RDKit or OpenBabel
3. **Semi-Empirical QM**: Fast geometry optimization and energy calculations with xTB
4. **Binding Energy Analysis**: Automated calculation and ranking of binding energies
5. **DFT Preparation**: Generation of production-ready ORCA input files with flexible settings

Designed for both rapid screening and detailed analysis of metal-ligand interactions with support for various coordination environments and computational settings.

## Features

- **Smart Binding Site Detection**: Automatically identifies carboxylates, amines, hydroxyls, and other coordinating groups
- **Enhanced Pose Generation**: Creates metal-ligand complexes with force field optimization and conformational sampling
  - Uses RDKit or OpenBabel for initial pose refinement
  - Generates multiple conformations for each pose
  - Maintains metal coordination during optimization
- **Conformer Generation**: Explores ligand conformational space
- **XTB Optimization**: Fast semi-empirical quantum mechanical optimization with solvent support
- **Binding Energy Calculation**: Computes metal-ligand binding energies with statistical analysis
- **ORCA Input Preparation**: Generates ready-to-run DFT input files with advanced features:
  - Vacuum and solvated calculations
  - Frequency calculations for thermodynamic analysis
  - Multiple spin multiplicities
  - Automatic UHF/RHF selection

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

1. **Ligand Conformer Generation**: Creates multiple 3D conformations exploring conformational space
2. **Metal Ion Preparation**: Prepares metal ion structures with appropriate charge and multiplicity
3. **Binding Site Detection**: Automatically identifies coordination sites (carboxylates, amines, etc.)
4. **Complex Pose Generation**: Creates metal-ligand arrangements with:
   - Force field pre-optimization (RDKit/OpenBabel)
   - Coordination geometry validation
   - RMSD-based duplicate removal
5. **Semi-Empirical Optimization**: Fast xTB optimization with:
   - Solvent effects (ALPB model)
   - Multiple conformers per pose
   - Energy and gradient convergence
6. **Binding Energy Analysis**: Computes E_binding = E_complex - E_metal - E_ligand
7. **Structure Ranking**: Orders by binding energy with statistical analysis
8. **ORCA Input Generation**: Creates production-ready DFT inputs with:
   - Automatic multiplicity determination
   - Vacuum/solvent configuration
   - Optional frequency calculations
   - Optimized computational settings

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
        solvent="water"  # or None for gas phase
    ),
    orca_config=ORCAConfig(
        method="B3LYP",
        basis_set="def2-SVP",
        dispersion="D3BJ",
        solvent="vacuum",              # Use "vacuum" for gas phase calculations
        calculate_frequencies=True     # Enable frequency calculations
    )
)

# Use custom configuration
complex = MetalLigandComplex(..., config=config)
```

### Advanced ORCA Configuration Examples

```python
# Gas phase calculations with frequencies
gas_phase_config = ORCAConfig(
    method="B3LYP",
    basis_set="def2-TZVP",
    dispersion="D3BJ", 
    solvent="vacuum",
    calculate_frequencies=True,
    n_cores=8,
    max_core=4000
)

# Solvated calculations
water_config = ORCAConfig(
    method="wB97X-D3",
    basis_set="def2-SVP",
    auxiliary_basis="def2/J",
    solvent_model="CPCM",
    solvent="water",
    calculate_frequencies=False
)

# High-level calculations
high_level_config = ORCAConfig(
    method="DLPNO-CCSD(T)",
    basis_set="def2-TZVPP",
    auxiliary_basis="def2-TZVPP/C",
    solvent="vacuum",
    additional_keywords=["TightSCF", "GridX6"]
)
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
experiment_name_analysis/
├── 01_initial_structures/
│   ├── complexes/              # Initial metal-ligand poses
│   ├── ligands/               # Ligand conformers
│   └── metals/                # Metal ion structures
├── 02_optimized_structures/    # xTB-optimized geometries
│   ├── complexes/             # Optimized complexes with energies
│   ├── ligands/               # Optimized ligand conformers
│   └── metals/                # Optimized metal structures
├── 03_orca_inputs/            # ORCA DFT input files
│   ├── complexes/             # Multiple multiplicities per complex
│   ├── ligands/               # Ligand inputs
│   └── metals/                # Metal ion inputs
├── 04_reports/                # Analysis results
│   ├── analysis_report.html   # Comprehensive HTML report
│   ├── complexes_results.csv  # Detailed binding energies
│   └── *_summary.csv          # Summary statistics
├── 05_best_structures/        # Top-ranked structures
└── 06_metadata_files/         # Workflow metadata and logs
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