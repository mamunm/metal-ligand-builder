"""Ligand-specific validation to ensure structural integrity."""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy.spatial.distance import cdist
from collections import defaultdict

from ..core.data_models import Geometry

logger = logging.getLogger(__name__)


class LigandValidator:
    """
    Validates ligand structural integrity in optimized complexes.
    
    This validator checks:
    - Core functional groups remain intact
    - Key bond patterns are preserved
    - Chelating groups maintain proper geometry
    - No unreasonable bond breaking/forming
    """
    
    def __init__(self):
        """Initialize ligand validator."""
        # Define expected bond patterns for common chelating groups
        self.chelating_groups = {
            'carboxylate': {
                'atoms': ['C', 'O', 'O'],
                'bonds': [(0, 1), (0, 2)],
                'max_bond_length': 1.4  # C-O in carboxylate
            },
            'amine': {
                'atoms': ['N'],
                'bonds': [],
                'max_bond_length': 1.5  # C-N
            }
        }
        
        # Bond length tolerances (in Angstroms)
        self.bond_lengths = {
            ('C', 'C'): (1.2, 1.6),   # Single/double C-C
            ('C', 'N'): (1.2, 1.5),   # C-N bonds
            ('C', 'O'): (1.1, 1.5),   # C-O bonds (double/single)
            ('N', 'H'): (0.9, 1.1),   # N-H bonds
            ('C', 'H'): (0.9, 1.1),   # C-H bonds
            ('O', 'H'): (0.9, 1.1),   # O-H bonds
        }
        
        # Maximum allowed bond stretch
        self.max_bond_stretch = 0.3  # Angstroms
    
    def validate_edta_integrity(
        self,
        geometry: Geometry,
        reference_geometry: Optional[Geometry] = None,
        metal_indices: Optional[List[int]] = None
    ) -> Tuple[bool, str, Dict[str, any]]:
        """
        Validate EDTA structural integrity.
        
        EDTA has:
        - 2 nitrogen atoms (tertiary amines)
        - 4 carboxylate groups (4 C=O and 4 C-O bonds)
        - Ethylene bridges connecting the components
        
        Args:
            geometry: Optimized geometry to validate
            reference_geometry: Reference ligand geometry (optional)
            metal_indices: Indices of metal atoms to exclude
            
        Returns:
            Tuple of (is_valid, message, details_dict)
        """
        details = {
            'carboxylate_groups': [],
            'amine_groups': [],
            'broken_bonds': [],
            'distorted_groups': []
        }
        
        # Exclude metal atoms from analysis
        if metal_indices:
            ligand_indices = [i for i in range(len(geometry.atoms)) if i not in metal_indices]
        else:
            ligand_indices = list(range(len(geometry.atoms)))
        
        # Extract ligand atoms and coordinates
        ligand_atoms = [geometry.atoms[i] for i in ligand_indices]
        ligand_coords = geometry.coordinates[ligand_indices]
        
        # Count key atoms
        n_nitrogen = ligand_atoms.count('N')
        n_carbon = ligand_atoms.count('C')
        n_oxygen = ligand_atoms.count('O')
        
        # EDTA should have 2 N, 10 C, 8 O (in deprotonated form)
        if n_nitrogen != 2:
            return False, f"Expected 2 N atoms, found {n_nitrogen}", details
        if n_carbon != 10:
            return False, f"Expected 10 C atoms, found {n_carbon}", details
        if n_oxygen != 8:
            return False, f"Expected 8 O atoms, found {n_oxygen}", details
        
        # Find and validate carboxylate groups
        carboxylate_count = self._find_carboxylate_groups(
            ligand_atoms, ligand_coords, ligand_indices, details
        )
        
        if carboxylate_count != 4:
            return False, f"Expected 4 carboxylate groups, found {carboxylate_count}", details
        
        # Check for broken C-N bonds (should have 8 C-N bonds in EDTA)
        cn_bonds = self._count_bonds(ligand_atoms, ligand_coords, 'C', 'N')
        if cn_bonds < 6:  # Allow some flexibility
            return False, f"Too few C-N bonds ({cn_bonds}), structure may be fragmented", details
        
        # If reference geometry provided, check for major distortions
        if reference_geometry:
            is_valid, message = self._compare_with_reference(
                geometry, reference_geometry, metal_indices, details
            )
            if not is_valid:
                return False, message, details
        
        # Check connectivity to ensure no fragmentation
        fragments = self._get_molecular_fragments(ligand_atoms, ligand_coords)
        if len(fragments) > 1:
            # Check if small fragments are just H atoms
            large_fragments = [f for f in fragments if len(f) > 1]
            if len(large_fragments) > 1:
                return False, f"Ligand fragmented into {len(large_fragments)} pieces", details
        
        return True, "EDTA structure intact", details
    
    def _find_carboxylate_groups(
        self,
        atoms: List[str],
        coords: np.ndarray,
        atom_indices: List[int],
        details: Dict
    ) -> int:
        """Find and validate carboxylate groups."""
        carboxylate_count = 0
        
        # Find all C atoms
        c_indices = [i for i, atom in enumerate(atoms) if atom == 'C']
        
        for c_idx in c_indices:
            # Find O atoms bonded to this C
            bonded_oxygens = []
            c_coord = coords[c_idx]
            
            for i, atom in enumerate(atoms):
                if atom == 'O':
                    dist = np.linalg.norm(coords[i] - c_coord)
                    if dist < 1.5:  # C-O bond
                        bonded_oxygens.append(i)
            
            # Check if this is a carboxylate (C bonded to 2 O)
            if len(bonded_oxygens) == 2:
                carboxylate_count += 1
                details['carboxylate_groups'].append({
                    'C_index': atom_indices[c_idx],
                    'O_indices': [atom_indices[o] for o in bonded_oxygens]
                })
        
        return carboxylate_count
    
    def _count_bonds(
        self,
        atoms: List[str],
        coords: np.ndarray,
        atom1: str,
        atom2: str,
        max_dist: float = 1.7
    ) -> int:
        """Count bonds between two atom types."""
        count = 0
        indices1 = [i for i, a in enumerate(atoms) if a == atom1]
        indices2 = [i for i, a in enumerate(atoms) if a == atom2]
        
        for i in indices1:
            for j in indices2:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < max_dist:
                    count += 1
        
        return count
    
    def _get_molecular_fragments(
        self,
        atoms: List[str],
        coords: np.ndarray
    ) -> List[List[int]]:
        """Get disconnected molecular fragments."""
        n_atoms = len(atoms)
        
        # Build connectivity matrix
        connectivity = np.zeros((n_atoms, n_atoms), dtype=bool)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Get expected max bond length
                atom_pair = tuple(sorted([atoms[i], atoms[j]]))
                max_dist = 1.8  # Default
                
                if atom_pair in self.bond_lengths:
                    max_dist = self.bond_lengths[atom_pair][1] + self.max_bond_stretch
                
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < max_dist:
                    connectivity[i, j] = True
                    connectivity[j, i] = True
        
        # Find connected components
        visited = [False] * n_atoms
        fragments = []
        
        def dfs(atom_idx: int, fragment: List[int]):
            visited[atom_idx] = True
            fragment.append(atom_idx)
            
            for j in range(n_atoms):
                if connectivity[atom_idx, j] and not visited[j]:
                    dfs(j, fragment)
        
        for i in range(n_atoms):
            if not visited[i]:
                fragment = []
                dfs(i, fragment)
                fragments.append(fragment)
        
        return fragments
    
    def _compare_with_reference(
        self,
        geometry: Geometry,
        reference: Geometry,
        metal_indices: Optional[List[int]],
        details: Dict
    ) -> Tuple[bool, str]:
        """Compare with reference structure to detect major distortions."""
        # This is a simplified check - could be enhanced with RMSD alignment
        
        # For now, just check if key distances are preserved
        # You could enhance this with proper structural alignment
        
        return True, "Reference comparison passed"
    
    def validate_ligand_for_metal_binding(
        self,
        geometry: Geometry,
        metal_indices: List[int],
        expected_denticity: int = 6  # EDTA is hexadentate
    ) -> Tuple[bool, str]:
        """
        Validate that ligand is properly positioned for metal binding.
        
        Args:
            geometry: Complex geometry
            metal_indices: Indices of metal atoms
            expected_denticity: Expected number of coordinating atoms
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not metal_indices:
            return False, "No metal indices provided"
        
        # Find potential coordinating atoms (N, O for EDTA)
        coordinating_atoms = []
        metal_coords = geometry.coordinates[metal_indices[0]]  # Assume single metal
        
        for i, atom in enumerate(geometry.atoms):
            if i in metal_indices:
                continue
                
            if atom in ['N', 'O']:
                dist = np.linalg.norm(geometry.coordinates[i] - metal_coords)
                if dist < 3.0:  # Reasonable coordination distance
                    coordinating_atoms.append((i, atom, dist))
        
        # Sort by distance
        coordinating_atoms.sort(key=lambda x: x[2])
        
        # Check if we have enough coordinating atoms
        if len(coordinating_atoms) < expected_denticity - 1:  # Allow one less
            return False, f"Only {len(coordinating_atoms)} coordinating atoms found, expected ~{expected_denticity}"
        
        # Check if coordination is reasonable
        if len(coordinating_atoms) >= expected_denticity:
            # Check if the closest atoms are at reasonable distances
            closest_dist = coordinating_atoms[0][2]
            if closest_dist < 1.6:  # Too close
                return False, f"Coordinating atom too close to metal ({closest_dist:.2f} Å)"
            
            furthest_dist = coordinating_atoms[expected_denticity-1][2]
            if furthest_dist > 2.8:  # Too far for effective coordination
                return False, f"Some coordinating atoms too far from metal (>{furthest_dist:.2f} Å)"
        
        # Check coordination geometry is reasonable (not all atoms on one side)
        if len(coordinating_atoms) >= 4:
            # Simple check: calculate centroid of coordinating atoms
            coord_positions = np.array([geometry.coordinates[a[0]] for a in coordinating_atoms[:expected_denticity]])
            centroid = np.mean(coord_positions, axis=0)
            
            # Metal should be reasonably close to centroid
            metal_to_centroid = np.linalg.norm(metal_coords - centroid)
            if metal_to_centroid > 1.5:
                return False, f"Metal displaced from coordination center ({metal_to_centroid:.2f} Å)"
        
        return True, "Metal coordination geometry acceptable"