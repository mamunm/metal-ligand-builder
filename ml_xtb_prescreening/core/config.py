"""Configuration classes for metal-ligand complex analysis."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class XTBConfig:
    """Configuration for xTB calculations."""
    method: str = "gfn2"  # gfn0, gfn1, gfn2, gfnff
    solvent: Optional[str] = None  # None for gas phase, or specify solvent name
    accuracy: float = 1.0
    electronic_temperature: float = 300.0
    max_iterations: int = 250
    convergence: str = "normal"  # loose, normal, tight
    parallel: int = 1
    constraints: Optional[Path] = None
    
    def to_cmd_args(self) -> List[str]:
        """Convert to xTB command line arguments."""
        args = [f"--{self.method}"]
        
        if self.solvent:
            args.extend(["--alpb", self.solvent])
        
        args.extend([
            "--acc", str(self.accuracy),
            "--etemp", str(self.electronic_temperature),
            "--iterations", str(self.max_iterations),
            f"--{self.convergence}"
        ])
        
        if self.parallel > 1:
            args.extend(["--parallel", str(self.parallel)])
            
        if self.constraints:
            args.extend(["--input", str(self.constraints)])
            
        return args


@dataclass
class ORCAConfig:
    """Configuration for ORCA calculations."""
    method: str = "B3LYP"
    basis_set: str = "def2-SVP"
    auxiliary_basis: Optional[str] = "def2/J"
    dispersion: Optional[str] = "D3BJ"
    solvent_model: Optional[str] = "CPCM"
    solvent: Optional[str] = "water"
    multiplicity: int = 1
    max_core: int = 2000  # MB per core
    n_cores: int = 4
    additional_keywords: List[str] = field(default_factory=list)
    
    def to_input_string(self, charge: int) -> str:
        """Generate ORCA input file header."""
        keywords = [self.method, self.basis_set]
        
        if self.auxiliary_basis:
            keywords.append(f"RIJCOSX {self.auxiliary_basis}")
            
        if self.dispersion:
            keywords.append(self.dispersion)
            
        if self.solvent_model and self.solvent:
            keywords.append(f"{self.solvent_model}({self.solvent})")
            
        keywords.extend(self.additional_keywords)
        
        # Add computational settings
        input_str = f"! {' '.join(keywords)}\n"
        input_str += f"%maxcore {self.max_core}\n"
        input_str += f"%pal nprocs {self.n_cores} end\n"
        input_str += f"* xyz {charge} {self.multiplicity}\n"
        
        return input_str


@dataclass
class ComplexConfig:
    """Main configuration for metal-ligand complex analysis."""
    # Experiment info
    experiment_name: str = "metal_ligand_binding"
    
    # Structure generation
    max_poses: int = 50
    n_conformers: int = 30
    rmsd_threshold: float = 0.5  # Å
    
    # Optimization settings
    optimize_metal: bool = True
    optimize_ligand: bool = True
    optimize_complex: bool = True
    
    # XTB settings
    xtb_config: XTBConfig = field(default_factory=XTBConfig)
    
    # ORCA settings
    prepare_orca: bool = True
    orca_config: ORCAConfig = field(default_factory=ORCAConfig)
    
    # Analysis settings
    energy_window: float = 10.0  # kcal/mol
    keep_top_n: int = 5
    
    # Computational resources
    n_workers: Optional[int] = None  # Auto-detect if None
    
    # Validation
    validate_geometries: bool = True
    max_bond_deviation: float = 0.5  # Å
    min_metal_ligand_distance: float = 1.5  # Å
    max_metal_ligand_distance: float = 3.0  # Å
    
    def __post_init__(self):
        """Post-initialization setup."""
        pass  # No setup needed since we use current working directory