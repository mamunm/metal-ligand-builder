# New Architecture for ML-xTB-Prescreening

## Overview

The codebase has been reorganized into a modular, professional structure focused on analyzing single metal-ligand pairs with binding energy calculations. The new architecture provides a clear, easy-to-understand workflow.

## Structure

```
ml_xtb_prescreening/
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── complex.py          # Main MetalLigandComplex class
│   ├── config.py           # Configuration classes
│   └── data_models.py      # Data structures
│
├── generators/             # Structure generation
│   ├── __init__.py
│   ├── binding_site_detector.py
│   ├── pose_generator.py
│   └── metal_generator.py
│
├── optimizers/             # Optimization interfaces
│   ├── __init__.py
│   ├── xtb_optimizer.py
│   └── conformer_generator.py
│
├── analysis/               # Analysis tools
│   ├── __init__.py
│   ├── energy_analyzer.py
│   ├── structure_validator.py
│   └── report_generator.py
│
├── io/                     # Input/Output
│   ├── __init__.py
│   ├── file_handler.py
│   └── orca_writer.py
│
└── utils/                  # Utilities
    ├── __init__.py
    └── parallel.py
```

## Main Components

### 1. MetalLigandComplex Class

The central class that orchestrates the entire workflow:

```python
from ml_xtb_prescreening import MetalLigandComplex

complex = MetalLigandComplex(
    ligand_name="edta",
    ligand_smiles="...",
    metal_symbol="Co",
    metal_charge=2,
    ligand_charge=-4,
    ligand_protonation_state="deprotonated",
    experiment_name="co_edta_binding"
)

# Run complete workflow
results = complex.run_workflow()
```

### 2. Workflow Steps

The workflow follows a logical sequence:

1. **Generate Ligand Conformers**: Create multiple 3D conformations
2. **Generate Metal Geometries**: Create metal ion structures
3. **Detect Binding Sites**: Find potential coordination sites
4. **Generate Complex Poses**: Create metal-ligand arrangements
5. **XTB Optimization**: Optimize all structures
6. **Calculate Binding Energies**: E_binding = E_complex - E_metal - E_ligand
7. **Rank Structures**: Sort by binding energy
8. **Prepare ORCA Inputs**: Create DFT input files

### 3. Configuration

Flexible configuration system:

```python
from ml_xtb_prescreening import ComplexConfig, XTBConfig, ORCAConfig

config = ComplexConfig(
    max_poses=100,
    n_conformers=50,
    energy_window=10.0,
    xtb_config=XTBConfig(
        method="gfn2",
        solvent="water"
    ),
    orca_config=ORCAConfig(
        method="B3LYP",
        basis_set="def2-SVP"
    )
)
```

### 4. Data Models

Clear data structures using dataclasses:

- `Metal`: Metal ion properties
- `Ligand`: Ligand information
- `Geometry`: 3D structure representation
- `BindingSite`: Coordination site information
- `OptimizationResult`: Results from optimization
- `BindingEnergyResult`: Binding energy calculations

## Key Features

1. **Single Focus**: Designed for one metal + one ligand analysis
2. **Modular Design**: Each component has a single responsibility
3. **Type Hints**: Full type annotations for clarity
4. **Professional Structure**: Follows Python best practices
5. **Easy Extension**: Simple to add new features

## Implementation Status

Currently, this is a **skeleton implementation**. The structure is in place, but the actual functionality needs to be ported from the existing code. The skeleton provides:

- ✅ Complete architecture design
- ✅ All class interfaces defined
- ✅ Type hints and documentation
- ✅ Example usage patterns
- ⏳ Implementation of methods (to be done)

## Next Steps

To complete the implementation:

1. Port binding site detection from existing `binding_sites.py`
2. Port pose generation from existing `pose_generation.py`
3. Port XTB interface from existing `xtb_interface.py`
4. Implement binding energy calculations
5. Add validation and error handling
6. Create comprehensive tests

## Benefits of New Architecture

1. **Clarity**: Clear separation of concerns
2. **Maintainability**: Easy to understand and modify
3. **Extensibility**: Simple to add new features
4. **Reusability**: Components can be used independently
5. **Professional**: Follows software engineering best practices