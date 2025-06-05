"""Structure validation and sanity checks for optimized geometries."""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy.spatial.distance import cdist

from ..core.data_models import Geometry, OptimizationResult
from .ligand_validator import LigandValidator

logger = logging.getLogger(__name__)


class StructureValidator:
    """
    Validate optimized structures for chemical sanity.
    
    Checks include:
    - Dissociation detection
    - Connectivity analysis
    - Unreasonable bond distances
    - Fragment analysis
    """
    
    # Covalent radii in Angstrom (from Cordero et al., 2008)
    COVALENT_RADII = {
        'H': 0.31, 'He': 0.28,
        'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
        'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
        'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32,
        'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20,
        'Br': 1.20, 'Kr': 1.16, 'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
        'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39,
        'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40
    }
    
    def __init__(
        self,
        bond_tolerance: float = 1.3,
        min_bond_distance: float = 0.4,
        max_bond_distance: float = 4.0,
        metal_coord_distance: float = 3.5
    ):
        """
        Initialize validator with distance criteria.
        
        Args:
            bond_tolerance: Factor to multiply sum of covalent radii for bond detection
            min_bond_distance: Minimum reasonable bond distance (Å)
            max_bond_distance: Maximum reasonable bond distance for organic bonds (Å)
            metal_coord_distance: Maximum metal-ligand coordination distance (Å)
        """
        self.bond_tolerance = bond_tolerance
        self.min_bond_distance = min_bond_distance
        self.max_bond_distance = max_bond_distance
        self.metal_coord_distance = metal_coord_distance
        
        # Initialize ligand validator for specific ligand checks
        self.ligand_validator = LigandValidator()
        
    def validate_optimization_result(
        self,
        result: OptimizationResult,
        is_complex: bool = False,
        metal_indices: Optional[List[int]] = None
    ) -> Tuple[bool, str]:
        """
        Validate an optimization result.
        
        Args:
            result: The optimization result to validate
            is_complex: Whether this is a metal-ligand complex
            metal_indices: Indices of metal atoms (if known)
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not result.success or result.optimized_geometry is None:
            return False, "Optimization failed"
            
        geometry = result.optimized_geometry
        
        # Check for reasonable bond distances
        is_valid, reason = self.check_bond_distances(geometry)
        if not is_valid:
            return False, f"Unreasonable distances: {reason}"
        
        # Check connectivity
        fragments = self.get_fragments(geometry)
        n_fragments = len(fragments)
        
        # For complexes, check if structure is dissociated
        if is_complex:
            if n_fragments > 1:
                # Check if any fragment contains metal
                if metal_indices:
                    metal_fragment = None
                    for i, fragment in enumerate(fragments):
                        if any(idx in fragment for idx in metal_indices):
                            metal_fragment = i
                            break
                    
                    if metal_fragment is not None:
                        # Check if metal is isolated
                        if len(fragments[metal_fragment]) == len(metal_indices):
                            return False, "Metal completely dissociated from ligand"
                        # Check if significant ligand parts dissociated
                        ligand_atoms = sum(len(f) for i, f in enumerate(fragments) if i != metal_fragment)
                        if ligand_atoms > 2:  # More than just H atoms dissociated
                            return False, f"Ligand fragmented ({n_fragments} fragments)"
                else:
                    # Without metal info, just check fragment count
                    if n_fragments > 2:
                        return False, f"Structure fragmented ({n_fragments} fragments)"
        else:
            # For non-complexes (ligands, single molecules)
            if n_fragments > 1:
                # Check if it's just a proton dissociation
                small_fragments = [f for f in fragments if len(f) == 1 
                                 and geometry.atoms[f[0]] == 'H']
                if len(small_fragments) == n_fragments - 1:
                    # Only H atoms dissociated
                    if len(small_fragments) > 2:
                        return False, f"Multiple H atoms dissociated ({len(small_fragments)})"
                else:
                    return False, f"Structure fragmented ({n_fragments} fragments)"
        
        # Additional checks for complexes
        if is_complex and metal_indices:
            is_valid, reason = self.check_metal_coordination(geometry, metal_indices)
            if not is_valid:
                return False, reason
        
        return True, "Structure is valid"
    
    def check_bond_distances(self, geometry: Geometry) -> Tuple[bool, str]:
        """Check for unreasonable bond distances."""
        coords = geometry.coordinates
        n_atoms = len(geometry.atoms)
        
        # Check for atoms too close
        dist_matrix = cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_dist = np.min(dist_matrix)
        if min_dist < self.min_bond_distance:
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            return False, f"Atoms {i} and {j} too close ({min_dist:.2f} Å)"
        
        # Check for unreasonably long bonds in connected atoms
        connectivity = self.get_connectivity_matrix(geometry)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if connectivity[i, j]:
                    dist = dist_matrix[i, j]
                    # Special handling for metal bonds
                    if self._is_metal(geometry.atoms[i]) or self._is_metal(geometry.atoms[j]):
                        if dist > self.metal_coord_distance:
                            return False, f"Metal bond too long ({geometry.atoms[i]}-{geometry.atoms[j]}: {dist:.2f} Å)"
                    else:
                        if dist > self.max_bond_distance:
                            return False, f"Bond too long ({geometry.atoms[i]}-{geometry.atoms[j]}: {dist:.2f} Å)"
        
        return True, ""
    
    def get_connectivity_matrix(self, geometry: Geometry) -> np.ndarray:
        """
        Get connectivity matrix based on covalent radii.
        
        Returns:
            Boolean matrix where True indicates bonded atoms
        """
        n_atoms = len(geometry.atoms)
        coords = geometry.coordinates
        connectivity = np.zeros((n_atoms, n_atoms), dtype=bool)
        
        # Calculate distance matrix
        dist_matrix = cdist(coords, coords)
        
        # Determine bonds based on covalent radii
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                atom_i = geometry.atoms[i]
                atom_j = geometry.atoms[j]
                
                # Get covalent radii
                r_i = self.COVALENT_RADII.get(atom_i, 1.5)
                r_j = self.COVALENT_RADII.get(atom_j, 1.5)
                
                # Check if bonded
                max_dist = (r_i + r_j) * self.bond_tolerance
                
                # Special handling for metal-ligand bonds
                if self._is_metal(atom_i) or self._is_metal(atom_j):
                    max_dist = max(max_dist, self.metal_coord_distance)
                
                if dist_matrix[i, j] <= max_dist:
                    connectivity[i, j] = True
                    connectivity[j, i] = True
        
        return connectivity
    
    def get_fragments(self, geometry: Geometry) -> List[List[int]]:
        """
        Get disconnected fragments in the structure.
        
        Returns:
            List of fragments, where each fragment is a list of atom indices
        """
        n_atoms = len(geometry.atoms)
        connectivity = self.get_connectivity_matrix(geometry)
        
        # Find connected components using DFS
        visited = [False] * n_atoms
        fragments = []
        
        def dfs(atom_idx: int, fragment: List[int]):
            visited[atom_idx] = True
            fragment.append(atom_idx)
            
            # Visit all connected atoms
            for j in range(n_atoms):
                if connectivity[atom_idx, j] and not visited[j]:
                    dfs(j, fragment)
        
        # Find all fragments
        for i in range(n_atoms):
            if not visited[i]:
                fragment = []
                dfs(i, fragment)
                fragments.append(fragment)
        
        return fragments
    
    def check_metal_coordination(
        self,
        geometry: Geometry,
        metal_indices: List[int]
    ) -> Tuple[bool, str]:
        """
        Check metal coordination environment.
        
        Args:
            geometry: The geometry to check
            metal_indices: Indices of metal atoms
            
        Returns:
            Tuple of (is_valid, reason)
        """
        coords = geometry.coordinates
        
        for metal_idx in metal_indices:
            if metal_idx >= len(geometry.atoms):
                continue
                
            metal_atom = geometry.atoms[metal_idx]
            metal_coord = coords[metal_idx]
            
            # Count coordinating atoms
            coordinating_atoms = []
            for i, (atom, coord) in enumerate(zip(geometry.atoms, coords)):
                if i == metal_idx:
                    continue
                    
                dist = np.linalg.norm(coord - metal_coord)
                
                # Check if within coordination distance
                if dist <= self.metal_coord_distance:
                    # Typically only N, O, S, P, Cl, etc. coordinate to metals
                    if atom in ['N', 'O', 'S', 'P', 'Cl', 'Br', 'I', 'F']:
                        coordinating_atoms.append((i, atom, dist))
            
            # Check coordination number
            if len(coordinating_atoms) == 0:
                return False, f"Metal {metal_atom} has no coordinating atoms"
            
            # Check for reasonable coordination numbers
            expected_cn_range = self._get_expected_coordination_number(metal_atom)
            if expected_cn_range and len(coordinating_atoms) not in expected_cn_range:
                return False, (f"Metal {metal_atom} has unusual coordination number "
                             f"({len(coordinating_atoms)}, expected {expected_cn_range})")
        
        return True, ""
    
    def _is_metal(self, atom: str) -> bool:
        """Check if atom is a metal."""
        metals = {
            'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
            'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
            'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf',
            'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
        }
        return atom in metals
    
    def _get_expected_coordination_number(self, metal: str) -> Optional[range]:
        """Get expected coordination number range for common metals."""
        cn_ranges = {
            # Alkali metals
            'Li': range(4, 7), 'Na': range(4, 9), 'K': range(6, 13),
            # Alkaline earth
            'Mg': range(4, 7), 'Ca': range(6, 9),
            # First row transition metals
            'Sc': range(6, 8), 'Ti': range(4, 7), 'V': range(4, 7),
            'Cr': range(4, 7), 'Mn': range(4, 8), 'Fe': range(4, 7),
            'Co': range(4, 7), 'Ni': range(4, 7), 'Cu': range(2, 7),
            'Zn': range(4, 7),
            # Second row transition metals
            'Y': range(6, 10), 'Zr': range(6, 9), 'Mo': range(4, 9),
            'Ru': range(4, 7), 'Rh': range(4, 7), 'Pd': range(4, 5),
            'Ag': range(2, 5), 'Cd': range(4, 7),
            # Other metals
            'Al': range(4, 7), 'Ga': range(4, 7), 'In': range(4, 8),
            'Sn': range(4, 7), 'Pb': range(4, 9), 'Bi': range(4, 9)
        }
        return cn_ranges.get(metal)
    
    def validate_complex_two_stage(
        self,
        result: OptimizationResult,
        ligand_name: str = "edta",
        metal_indices: Optional[List[int]] = None,
        reference_ligand: Optional[Geometry] = None
    ) -> Tuple[bool, str]:
        """
        Two-stage validation for metal-ligand complexes.
        
        Stage 1: Validate ligand structural integrity
        Stage 2: Validate metal binding geometry
        
        Args:
            result: Optimization result to validate
            ligand_name: Name of the ligand (currently supports "edta")
            metal_indices: Indices of metal atoms
            reference_ligand: Reference ligand structure for comparison
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not result.success or result.optimized_geometry is None:
            return False, "Optimization failed"
        
        geometry = result.optimized_geometry
        
        # If metal indices not provided, try to identify them
        if metal_indices is None:
            metal_indices = []
            for i, atom in enumerate(geometry.atoms):
                if self._is_metal(atom):
                    metal_indices.append(i)
        
        if not metal_indices:
            return False, "No metal atoms found in complex"
        
        # Stage 1: Check ligand integrity
        if ligand_name.lower() == "edta":
            is_valid, message, details = self.ligand_validator.validate_edta_integrity(
                geometry, reference_ligand, metal_indices
            )
            if not is_valid:
                return False, f"Ligand validation failed: {message}"
        else:
            # For other ligands, use generic validation
            logger.warning(f"No specific validator for ligand '{ligand_name}', using generic validation")
        
        # Stage 2: Check metal coordination
        is_valid, message = self.ligand_validator.validate_ligand_for_metal_binding(
            geometry, metal_indices
        )
        if not is_valid:
            return False, f"Metal binding validation failed: {message}"
        
        # Additional general checks
        # Check for unreasonable distances
        is_valid, reason = self.check_bond_distances(geometry)
        if not is_valid:
            return False, f"Distance check failed: {reason}"
        
        return True, "Complex structure validated successfully"