"""ORCA input file writer."""

import logging
from pathlib import Path
from typing import List, Optional

from ..core.data_models import Geometry, Metal, Ligand
from ..core.config import ORCAConfig

logger = logging.getLogger(__name__)


class ORCAWriter:
    """Write ORCA input files."""
    
    def __init__(self, config: Optional[ORCAConfig] = None):
        """
        Initialize ORCA writer.
        
        Args:
            config: ORCA configuration
        """
        self.config = config or ORCAConfig()
    
    def write_input(
        self,
        geometry: Geometry,
        charge: int,
        multiplicity: int,
        output_path: Path,
        title: Optional[str] = None
    ) -> Path:
        """
        Write ORCA input file.
        
        Args:
            geometry: Molecular geometry
            charge: Total charge
            multiplicity: Spin multiplicity
            output_path: Output file path
            title: Optional title for the calculation
            
        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate input content
        content = []
        
        # Title
        if title:
            content.append(f"# {title}")
            content.append("")
        
        # Keywords and settings
        keywords = self._generate_keywords()
        content.append(f"! {keywords}")
        
        # Additional blocks
        content.extend(self._generate_blocks())
        
        # Coordinates
        content.append(f"* xyz {charge} {multiplicity}")
        for atom, coord in zip(geometry.atoms, geometry.coordinates):
            content.append(
                f"  {atom:<2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}"
            )
        content.append("*")
        
        # Write file
        output_path.write_text('\n'.join(content))
        logger.info(f"Wrote ORCA input to {output_path}")
        
        return output_path
    
    def write_complex_input(
        self,
        geometry: Geometry,
        metal: Metal,
        ligand: Ligand,
        output_path: Path
    ) -> Path:
        """
        Write ORCA input for metal-ligand complex.
        
        Args:
            geometry: Complex geometry
            metal: Metal properties
            ligand: Ligand properties
            output_path: Output file path
            
        Returns:
            Path to written file
        """
        # Calculate total charge and multiplicity
        total_charge = metal.charge + ligand.charge
        
        # Determine multiplicity based on metal
        multiplicity = self._determine_multiplicity(metal, ligand)
        
        title = f"{metal.symbol}-{ligand.name} complex"
        
        return self.write_input(
            geometry, total_charge, multiplicity, output_path, title
        )
    
    def write_batch_inputs(
        self,
        geometries: List[Geometry],
        charges: List[int],
        multiplicities: List[int],
        output_dir: Path,
        prefix: str = "orca"
    ) -> List[Path]:
        """
        Write multiple ORCA input files.
        
        Args:
            geometries: List of geometries
            charges: List of charges
            multiplicities: List of multiplicities
            output_dir: Output directory
            prefix: File prefix
            
        Returns:
            List of written file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for i, (geom, charge, mult) in enumerate(
            zip(geometries, charges, multiplicities)
        ):
            filename = f"{prefix}_{i:04d}.inp"
            path = self.write_input(
                geom, charge, mult, 
                output_dir / filename,
                title=f"{prefix} structure {i}"
            )
            paths.append(path)
        
        return paths
    
    def _generate_keywords(self) -> str:
        """Generate ORCA keyword line."""
        keywords = [self.config.method, self.config.basis_set]
        
        if self.config.auxiliary_basis:
            keywords.append(f"RIJCOSX {self.config.auxiliary_basis}")
        
        if self.config.dispersion:
            keywords.append(self.config.dispersion)
        
        if self.config.solvent_model and self.config.solvent:
            keywords.append(f"{self.config.solvent_model}({self.config.solvent})")
        
        keywords.extend(self.config.additional_keywords)
        
        return " ".join(keywords)
    
    def _generate_blocks(self) -> List[str]:
        """Generate ORCA input blocks."""
        blocks = []
        
        # MaxCore
        blocks.append(f"%maxcore {self.config.max_core}")
        
        # Parallel
        if self.config.n_cores > 1:
            blocks.append(f"%pal nprocs {self.config.n_cores} end")
        
        # SCF settings for difficult convergence
        blocks.append("%scf")
        blocks.append("  MaxIter 200")
        blocks.append("  CNVDIIS 15")
        blocks.append("end")
        
        blocks.append("")
        
        return blocks
    
    def _determine_multiplicity(
        self, 
        metal: Metal, 
        ligand: Ligand
    ) -> int:
        """
        Determine multiplicity for metal-ligand complex.
        
        Simple heuristic based on metal d-electron count.
        """
        # D-electron counts for common oxidation states
        d_electrons = {
            ("Fe", 2): 6,  # Fe(II) - d6
            ("Fe", 3): 5,  # Fe(III) - d5
            ("Co", 2): 7,  # Co(II) - d7
            ("Co", 3): 6,  # Co(III) - d6
            ("Ni", 2): 8,  # Ni(II) - d8
            ("Cu", 2): 9,  # Cu(II) - d9
            ("Zn", 2): 10, # Zn(II) - d10
            ("Mn", 2): 5,  # Mn(II) - d5
        }
        
        n_d = d_electrons.get((metal.symbol, metal.charge), 0)
        
        # Simple rules (can be refined)
        if n_d in [0, 10]:  # d0 or d10
            return 1  # Singlet
        elif n_d in [5]:  # d5 high spin
            return 6  # Sextet for high spin d5
        elif n_d in [4, 6]:  # d4 or d6
            return 1  # Often low spin singlet
        else:
            # Default: assume high spin
            return n_d + 1 if n_d <= 5 else 11 - n_d