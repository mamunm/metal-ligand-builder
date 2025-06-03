"""
ML-xTB-Prescreening: Metal-Ligand Binding Analysis

A modular package for analyzing metal-ligand complexes with:
- Binding site detection
- Pose generation based on coordination chemistry
- XTB optimization
- Binding energy calculations
- ORCA input preparation
"""

from .core import (
    MetalLigandComplex,
    ComplexConfig,
    XTBConfig,
    ORCAConfig,
    Metal,
    Ligand,
    BindingSite,
    Geometry,
    OptimizationResult,
    BindingEnergyResult
)

from .optimizers import (
    XTBOptimizer,
    ConformerGenerator
)

from .generators import (
    BindingSiteDetector,
    PoseGenerator,
    MetalGenerator
)

from .analysis import (
    EnergyAnalyzer,
    ReportGenerator
)

from .io import (
    FileHandler
)

__version__ = "2.0.0"

__all__ = [
    # Core classes
    'MetalLigandComplex',
    'ComplexConfig',
    'XTBConfig',
    'ORCAConfig',
    'Metal',
    'Ligand',
    'BindingSite',
    'Geometry',
    'OptimizationResult',
    'BindingEnergyResult',
    
    # Optimizers
    'XTBOptimizer',
    'ConformerGenerator',
    
    # Generators
    'BindingSiteDetector',
    'PoseGenerator',
    'MetalGenerator',
    
    # Analysis
    'EnergyAnalyzer',
    'ReportGenerator',
    
    # IO
    'FileHandler'
]