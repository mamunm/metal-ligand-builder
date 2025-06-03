"""Energy analysis and binding energy calculations."""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from ..core.data_models import (
    OptimizationResult, BindingEnergyResult, Geometry
)

logger = logging.getLogger(__name__)


class EnergyAnalyzer:
    """Analyze energies and calculate binding energies."""
    
    # Conversion factors
    HARTREE_TO_KCAL = 627.5094740631  # Hartree to kcal/mol
    
    def calculate_binding_energy(
        self,
        complex_result: OptimizationResult,
        metal_result: OptimizationResult,
        ligand_result: OptimizationResult,
        bsse_correction: Optional[float] = None,
        solvent_correction: Optional[float] = None
    ) -> BindingEnergyResult:
        """
        Calculate binding energy.
        
        Binding Energy = E(complex) - E(metal) - E(ligand)
        
        Args:
            complex_result: Optimized complex
            metal_result: Optimized metal
            ligand_result: Optimized ligand
            bsse_correction: Basis set superposition error correction
            solvent_correction: Solvation energy correction
            
        Returns:
            Binding energy result
        """
        if not all([r.success for r in [complex_result, metal_result, ligand_result]]):
            raise ValueError("All optimizations must be successful")
        
        # Get energies
        e_complex = complex_result.energy
        e_metal = metal_result.energy
        e_ligand = ligand_result.energy
        
        # Calculate binding energy
        binding_energy_hartree = e_complex - e_metal - e_ligand
        
        # Apply corrections if provided
        if bsse_correction:
            binding_energy_hartree += bsse_correction
        if solvent_correction:
            binding_energy_hartree += solvent_correction
        
        # Convert to kcal/mol
        binding_energy_kcal = binding_energy_hartree * self.HARTREE_TO_KCAL
        
        return BindingEnergyResult(
            binding_energy=binding_energy_kcal,
            binding_energy_hartree=binding_energy_hartree,
            complex_energy=e_complex,
            metal_energy=e_metal,
            ligand_energy=e_ligand,
            bsse_correction=bsse_correction,
            solvent_correction=solvent_correction,
            complex_geometry=complex_result.optimized_geometry,
            metal_geometry=metal_result.optimized_geometry,
            ligand_geometry=ligand_result.optimized_geometry
        )
    
    def calculate_relative_energies(
        self,
        results: List[OptimizationResult],
        reference_idx: int = 0,
        unit: str = "kcal/mol"
    ) -> List[float]:
        """
        Calculate relative energies.
        
        Args:
            results: List of optimization results
            reference_idx: Index of reference structure (default: lowest energy)
            unit: Energy unit ("hartree" or "kcal/mol")
            
        Returns:
            List of relative energies
        """
        # Extract energies
        energies = []
        for result in results:
            if result.success and result.energy is not None:
                energies.append(result.energy)
            else:
                energies.append(float('inf'))
        
        # Find reference (lowest energy if reference_idx is 0)
        if reference_idx == 0:
            ref_energy = min(e for e in energies if e != float('inf'))
        else:
            ref_energy = energies[reference_idx]
        
        # Calculate relative energies
        rel_energies = []
        for energy in energies:
            if energy == float('inf'):
                rel_energies.append(float('inf'))
            else:
                rel_e = energy - ref_energy
                if unit == "kcal/mol":
                    rel_e *= self.HARTREE_TO_KCAL
                rel_energies.append(rel_e)
        
        return rel_energies
    
    def rank_by_energy(
        self,
        results: List[OptimizationResult]
    ) -> List[Tuple[int, float]]:
        """
        Rank structures by energy.
        
        Args:
            results: List of optimization results
            
        Returns:
            List of (index, energy) tuples sorted by energy
        """
        indexed_energies = []
        
        for i, result in enumerate(results):
            if result.success and result.energy is not None:
                indexed_energies.append((i, result.energy))
        
        # Sort by energy
        indexed_energies.sort(key=lambda x: x[1])
        
        return indexed_energies
    
    def calculate_boltzmann_populations(
        self,
        energies: List[float],
        temperature: float = 298.15
    ) -> List[float]:
        """
        Calculate Boltzmann populations.
        
        Args:
            energies: List of energies in kcal/mol
            temperature: Temperature in Kelvin
            
        Returns:
            List of populations (sum to 1)
        """
        # Constants
        R = 1.987e-3  # kcal/mol/K
        
        # Remove infinite energies
        finite_energies = [e for e in energies if e != float('inf')]
        if not finite_energies:
            return [0.0] * len(energies)
        
        # Reference to lowest energy
        min_energy = min(finite_energies)
        
        # Calculate Boltzmann factors
        populations = []
        for energy in energies:
            if energy == float('inf'):
                populations.append(0.0)
            else:
                delta_e = energy - min_energy
                populations.append(np.exp(-delta_e / (R * temperature)))
        
        # Normalize
        total = sum(populations)
        if total > 0:
            populations = [p / total for p in populations]
        
        return populations
    
    def analyze_energy_distribution(
        self,
        results: List[OptimizationResult],
        energy_window: float = 10.0
    ) -> Dict[str, Any]:
        """
        Analyze energy distribution of structures.
        
        Args:
            results: List of optimization results
            energy_window: Energy window in kcal/mol
            
        Returns:
            Dictionary with distribution statistics
        """
        # Get successful results
        successful = [r for r in results if r.success and r.energy is not None]
        
        if not successful:
            return {
                'n_successful': 0,
                'n_failed': len(results),
                'lowest_energy': None,
                'highest_energy': None,
                'energy_range': None,
                'n_within_window': 0,
                'populations': []
            }
        
        # Calculate relative energies
        rel_energies = self.calculate_relative_energies(successful)
        
        # Find structures within energy window
        n_within_window = sum(1 for e in rel_energies if e <= energy_window)
        
        # Calculate populations
        populations = self.calculate_boltzmann_populations(rel_energies)
        
        return {
            'n_successful': len(successful),
            'n_failed': len(results) - len(successful),
            'lowest_energy': min(r.energy for r in successful),
            'highest_energy': max(r.energy for r in successful),
            'energy_range': max(rel_energies) if rel_energies else 0,
            'n_within_window': n_within_window,
            'populations': populations,
            'mean_energy': np.mean([r.energy for r in successful]),
            'std_energy': np.std([r.energy for r in successful])
        }