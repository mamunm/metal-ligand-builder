"""Data models for metal-ligand complex analysis."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from enum import Enum


class CoordinationGeometry(Enum):
    """Standard coordination geometries."""
    LINEAR = "linear"
    TRIGONAL_PLANAR = "trigonal_planar"
    TETRAHEDRAL = "tetrahedral"
    SQUARE_PLANAR = "square_planar"
    TRIGONAL_BIPYRAMIDAL = "trigonal_bipyramidal"
    SQUARE_PYRAMIDAL = "square_pyramidal"
    OCTAHEDRAL = "octahedral"
    PENTAGONAL_BIPYRAMIDAL = "pentagonal_bipyramidal"
    CAPPED_OCTAHEDRAL = "capped_octahedral"
    DODECAHEDRAL = "dodecahedral"


class BindingSiteType(Enum):
    """Types of binding sites."""
    CARBOXYLATE = "carboxylate"
    AMINE = "amine"
    HYDROXYL = "hydroxyl"
    IMIDAZOLE = "imidazole"
    PHOSPHATE = "phosphate"
    SULFUR = "sulfur"
    OTHER = "other"


@dataclass
class Metal:
    """Metal ion properties."""
    symbol: str
    charge: int
    coordination_numbers: List[int] = field(default_factory=list)
    preferred_geometries: List[CoordinationGeometry] = field(default_factory=list)
    typical_bond_lengths: Dict[str, float] = field(default_factory=dict)  # Element -> distance (Å)
    ionic_radius: Optional[float] = None
    
    def __post_init__(self):
        """Set default values based on metal type."""
        if not self.coordination_numbers:
            # Common coordination numbers
            defaults = {
                "Co": [4, 6],
                "Ni": [4, 6],
                "Cu": [4, 5, 6],
                "Zn": [4, 6],
                "Fe": [4, 6],
                "Mn": [4, 6],
                "Ca": [6, 7, 8],
                "Mg": [6],
                "Na": [2, 3, 4, 5, 6],  # Na+ can have variable coordination
                "K": [4, 6, 7, 8],
                "Li": [2, 3, 4]
            }
            self.coordination_numbers = defaults.get(self.symbol, [4, 6])
        
        if not self.preferred_geometries:
            # Default geometries based on coordination number
            if 2 in self.coordination_numbers:
                self.preferred_geometries.append(CoordinationGeometry.LINEAR)
            if 3 in self.coordination_numbers:
                self.preferred_geometries.append(CoordinationGeometry.TRIGONAL_PLANAR)
            if 4 in self.coordination_numbers:
                self.preferred_geometries.append(CoordinationGeometry.TETRAHEDRAL)
                if self.symbol in ["Ni", "Pd", "Pt", "Cu"]:
                    self.preferred_geometries.append(CoordinationGeometry.SQUARE_PLANAR)
            if 5 in self.coordination_numbers:
                self.preferred_geometries.extend([
                    CoordinationGeometry.TRIGONAL_BIPYRAMIDAL,
                    CoordinationGeometry.SQUARE_PYRAMIDAL
                ])
            if 6 in self.coordination_numbers:
                self.preferred_geometries.append(CoordinationGeometry.OCTAHEDRAL)


@dataclass
class Ligand:
    """Ligand properties."""
    name: str
    smiles: str
    charge: int
    protonation_state: str = "deprotonated"  # or "neutral", "protonated"
    xyz_path: Optional[Path] = None
    atoms: List[str] = field(default_factory=list)
    coordinates: Optional[np.ndarray] = None


@dataclass
class BindingSite:
    """Information about a potential binding site."""
    atom_indices: List[int]
    site_type: BindingSiteType
    score: float
    position: np.ndarray  # Average position
    functional_group_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Geometry:
    """3D geometry representation."""
    atoms: List[str]
    coordinates: np.ndarray
    title: str = ""
    energy: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_xyz_string(self) -> str:
        """Convert to XYZ format string."""
        lines = [f"{len(self.atoms)}", self.title]
        for atom, coord in zip(self.atoms, self.coordinates):
            lines.append(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
        return "\n".join(lines)
    
    def save_xyz(self, filepath: Path) -> None:
        """Save geometry to XYZ file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_xyz_string())


@dataclass
class OptimizationResult:
    """Results from geometry optimization."""
    success: bool
    initial_geometry: Geometry
    optimized_geometry: Optional[Geometry] = None
    energy: Optional[float] = None
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)  # HOMO-LUMO, dipole, etc.
    output_files: Dict[str, Path] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class BindingEnergyResult:
    """Binding energy calculation results."""
    binding_energy: float  # in kcal/mol
    binding_energy_hartree: float
    complex_energy: float
    metal_energy: float
    ligand_energy: float
    complex_geometry: Geometry
    metal_geometry: Geometry
    ligand_geometry: Geometry
    bsse_correction: Optional[float] = None
    solvent_correction: Optional[float] = None