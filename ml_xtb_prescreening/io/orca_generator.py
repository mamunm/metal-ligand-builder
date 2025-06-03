"""ORCA input generation with multiplicity handling and structure selection."""

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import shutil

from ..core.data_models import Geometry, OptimizationResult, Metal, Ligand
from ..core.config import ORCAConfig

logger = logging.getLogger(__name__)


class ORCAGenerator:
    """
    Generate ORCA input files for DFT calculations.
    
    This class handles:
    - Selection of lowest energy structures
    - Multiplicity determination and folder organization
    - UHF/RHF selection based on multiplicity
    - Input file generation with proper settings
    """
    
    def __init__(self, base_dir: Path, config: Optional[ORCAConfig] = None):
        """
        Initialize ORCA generator.
        
        Args:
            base_dir: Base directory for output
            config: ORCA configuration
        """
        self.base_dir = Path(base_dir)
        self.config = config or ORCAConfig()
        self.orca_dir = self.base_dir / "03_orca_inputs"
        self.orca_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_from_optimization_results(
        self,
        optimization_results: Dict[str, List[OptimizationResult]],
        metal: Metal,
        ligand: Ligand,
        n_lowest: int = 5,
        multiplicities: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate ORCA inputs from optimization results.
        
        Args:
            optimization_results: Dictionary with optimization results by type
            metal: Metal properties
            ligand: Ligand properties
            n_lowest: Number of lowest energy structures to use
            multiplicities: List of multiplicities to consider (auto if None)
            
        Returns:
            Dictionary with generation summary
        """
        logger.info(f"Generating ORCA inputs for {n_lowest} lowest energy structures")
        
        # Determine multiplicities if not provided
        if multiplicities is None:
            multiplicities = self._determine_multiplicities(metal, ligand)
        
        logger.info(f"Considering multiplicities: {multiplicities}")
        
        results = {
            "metal_inputs": [],
            "ligand_inputs": [],
            "complex_inputs": [],
            "summary": {
                "n_requested": n_lowest,
                "multiplicities": multiplicities,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Process each structure type
        for struct_type in ["metals", "ligands", "complexes"]:
            if struct_type not in optimization_results:
                continue
            
            # Get successful optimizations
            successful = [r for r in optimization_results[struct_type] 
                         if r.success and r.optimized_geometry is not None]
            
            if not successful:
                logger.warning(f"No successful optimizations found for {struct_type}")
                continue
            
            # Sort by energy and take n_lowest
            # Filter out results without energy (e.g., metals with only single-point)
            successful_with_energy = [r for r in successful if r.energy is not None]
            
            if successful_with_energy:
                successful_with_energy.sort(key=lambda r: r.energy)
                selected = successful_with_energy[:n_lowest]
            else:
                # If no energies available (e.g., for metals), just take first n
                selected = successful[:n_lowest]
            
            logger.info(f"Processing {len(selected)} {struct_type}")
            
            # Determine charge for this structure type
            if struct_type == "metals":
                charge = metal.charge
                struct_multiplicities = self._get_metal_multiplicities(metal)
            elif struct_type == "ligands":
                charge = ligand.charge
                struct_multiplicities = [1]  # Usually singlet
            else:  # complexes
                charge = metal.charge + ligand.charge
                struct_multiplicities = multiplicities
            
            # Generate inputs
            type_results = self._generate_inputs_for_type(
                struct_type, selected, charge, struct_multiplicities
            )
            
            results[f"{struct_type[:-1]}_inputs"] = type_results
        
        # Save summary
        self._save_generation_summary(results)
        
        return results
    
    def _generate_inputs_for_type(
        self,
        struct_type: str,
        optimized_results: List[OptimizationResult],
        charge: int,
        multiplicities: List[int]
    ) -> List[Dict[str, Any]]:
        """Generate ORCA inputs for a structure type."""
        type_dir = self.orca_dir / struct_type
        type_dir.mkdir(exist_ok=True)
        
        generated_inputs = []
        
        for i, result in enumerate(optimized_results):
            geometry = result.optimized_geometry
            
            # Create base name and structure directory
            base_name = f"{struct_type[:-1]}_{i+1:03d}"
            struct_dir = type_dir / base_name
            struct_dir.mkdir(exist_ok=True)
            
            # Generate input for each multiplicity
            for mult in multiplicities:
                mult_dir = struct_dir / f"mult_{mult}"
                mult_dir.mkdir(exist_ok=True)
                
                # Determine if UHF is needed
                use_uhf = mult > 1
                
                # Create input file
                input_file = mult_dir / f"{base_name}_m{mult}.inp"
                
                self._write_orca_input(
                    geometry=geometry,
                    charge=charge,
                    multiplicity=mult,
                    output_path=input_file,
                    use_uhf=use_uhf,
                    title=f"{base_name} Multiplicity={mult}"
                )
                
                # Store info
                input_info = {
                    "structure_index": i,
                    "original_title": geometry.title,
                    "xtb_energy": result.energy,
                    "charge": charge,
                    "multiplicity": mult,
                    "input_file": str(input_file.relative_to(self.base_dir)),
                    "use_uhf": use_uhf
                }
                
                generated_inputs.append(input_info)
        
        return generated_inputs
    
    def _write_orca_input(
        self,
        geometry: Geometry,
        charge: int,
        multiplicity: int,
        output_path: Path,
        use_uhf: bool,
        title: str
    ) -> None:
        """Write ORCA input file with proper settings."""
        lines = []
        
        # Header comment
        lines.append(f"# {title}")
        lines.append(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Main keywords line
        keywords = [self.config.method, self.config.basis_set]
        
        # Add UHF if needed
        if use_uhf:
            keywords.insert(0, "UHF")
        
        # Add auxiliary basis if RIJCOSX
        if self.config.auxiliary_basis:
            keywords.append(f"RIJCOSX {self.config.auxiliary_basis}")
        
        # Add dispersion
        if self.config.dispersion:
            keywords.append(self.config.dispersion)
        
        # Add solvation
        if self.config.solvent_model and self.config.solvent:
            keywords.append(f"{self.config.solvent_model}({self.config.solvent})")
        
        # Add optimization
        keywords.append("Opt")
        
        # Add additional keywords
        keywords.extend(self.config.additional_keywords)
        
        lines.append(f"! {' '.join(keywords)}")
        lines.append("")
        
        # Add blocks
        # MaxCore
        lines.append(f"%maxcore {self.config.max_core}")
        lines.append("")
        
        # Parallel settings
        if self.config.n_cores > 1:
            lines.append("%pal")
            lines.append(f"  nprocs {self.config.n_cores}")
            lines.append("end")
            lines.append("")
        
        # SCF settings for better convergence
        lines.append("%scf")
        lines.append("  MaxIter 300")
        if use_uhf:
            lines.append("  UHF true")
        lines.append("  ConvTol 1e-7")
        lines.append("end")
        lines.append("")
        
        # Geometry optimization settings
        lines.append("%geom")
        lines.append("  MaxIter 100")
        lines.append("  Trust 0.3")
        lines.append("end")
        lines.append("")
        
        # Output settings
        lines.append("%output")
        lines.append("  Print[P_MOs] 1")
        lines.append("  Print[P_Overlap] 1")
        lines.append("end")
        lines.append("")
        
        # Coordinates
        lines.append(f"* xyz {charge} {multiplicity}")
        for atom, coord in zip(geometry.atoms, geometry.coordinates):
            lines.append(f"  {atom:<2s} {coord[0]:16.10f} {coord[1]:16.10f} {coord[2]:16.10f}")
        lines.append("*")
        lines.append("")
        
        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines))
        
        logger.debug(f"Wrote ORCA input: {output_path}")
    
    def _determine_multiplicities(self, metal: Metal, ligand: Ligand) -> List[int]:
        """
        Determine possible multiplicities for metal-ligand complex.
        
        Returns list of multiplicities to consider.
        Note: multiplicities are consistent - either all odd (1,3,5,7,9) or all even (2,4,6,8).
        """
        # Get d-electron count
        d_electrons = self._get_d_electron_count(metal)
        
        # Calculate total electrons for the complex
        # This determines if we need odd or even multiplicities
        # Total charge of complex
        complex_charge = metal.charge + ligand.charge
        
        # For transition metal complexes, we typically consider:
        # - If total electrons is even -> odd multiplicities (1,3,5,7,9)
        # - If total electrons is odd -> even multiplicities (2,4,6,8)
        
        # Common multiplicity patterns based on d-electron count
        if d_electrons == 0:  # d0
            return [1]
        elif d_electrons == 1:  # d1
            return [2]
        elif d_electrons == 2:  # d2
            return [1, 3]
        elif d_electrons == 3:  # d3
            return [2, 4]
        elif d_electrons == 4:  # d4
            return [1, 3, 5]
        elif d_electrons == 5:  # d5
            return [2, 4, 6]
        elif d_electrons == 6:  # d6
            return [1, 3, 5]
        elif d_electrons == 7:  # d7
            return [2, 4]
        elif d_electrons == 8:  # d8
            return [1, 3]
        elif d_electrons == 9:  # d9
            return [2]
        elif d_electrons == 10:  # d10
            return [1]
        else:
            # Default for main group or unknown
            return [1]
    
    def _get_metal_multiplicities(self, metal: Metal) -> List[int]:
        """Get possible multiplicities for isolated metal ion."""
        d_electrons = self._get_d_electron_count(metal)
        
        # For isolated ions, usually high spin
        if d_electrons <= 5:
            return [d_electrons + 1]  # All unpaired
        else:
            return [11 - d_electrons]  # After half-filled
    
    def _get_d_electron_count(self, metal: Metal) -> int:
        """Get d-electron count for transition metal."""
        # Atomic numbers for transition metals
        atomic_numbers = {
            "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
            "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
            "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
            "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48
        }
        
        if metal.symbol not in atomic_numbers:
            return 0  # Not a transition metal
        
        # Calculate d-electrons
        z = atomic_numbers[metal.symbol]
        
        # First row transition metals
        if 21 <= z <= 30:
            neutral_d = z - 20  # Sc is 3d1, Ti is 3d2, etc.
        # Second row
        elif 39 <= z <= 48:
            neutral_d = z - 38
        else:
            return 0
        
        # Adjust for charge
        d_electrons = neutral_d - metal.charge
        
        # Handle special cases (Cr, Cu have different configurations)
        if metal.symbol == "Cr" and metal.charge == 0:
            d_electrons = 5  # 3d5 4s1
        elif metal.symbol == "Cu" and metal.charge == 0:
            d_electrons = 10  # 3d10 4s1
        
        return max(0, d_electrons)
    
    def _save_generation_summary(self, results: Dict[str, Any]) -> None:
        """Save summary of ORCA input generation."""
        summary = {
            "generation_info": {
                "timestamp": results["summary"]["timestamp"],
                "orca_config": {
                    "method": self.config.method,
                    "basis_set": self.config.basis_set,
                    "dispersion": self.config.dispersion,
                    "solvent": self.config.solvent,
                    "n_cores": self.config.n_cores
                },
                "multiplicities_considered": results["summary"]["multiplicities"]
            },
            "generated_files": {
                "metals": len(results["metal_inputs"]),
                "ligands": len(results["ligand_inputs"]),
                "complexes": len(results["complex_inputs"]),
                "total": (len(results["metal_inputs"]) + 
                         len(results["ligand_inputs"]) + 
                         len(results["complex_inputs"]))
            },
            "file_details": {
                "metals": results["metal_inputs"],
                "ligands": results["ligand_inputs"],
                "complexes": results["complex_inputs"]
            }
        }
        
        summary_file = self.orca_dir / "orca_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved generation summary to {self._get_relative_path(summary_file)}")
    
    def _get_relative_path(self, path: Path) -> str:
        """Get relative path from current working directory."""
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            # If path is not relative to cwd, just return the name
            return path.name