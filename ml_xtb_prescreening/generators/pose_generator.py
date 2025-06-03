"""Metal-ligand pose generation."""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from itertools import combinations

from ..core.data_models import (
    Geometry, BindingSite, Metal, 
    CoordinationGeometry, BindingSiteType
)

logger = logging.getLogger(__name__)


class PoseGenerator:
    """Generate metal-ligand complex poses based on coordination chemistry."""
    
    def __init__(self):
        """Initialize pose generator."""
        # Define ideal coordination geometries
        self.ideal_geometries = {
            CoordinationGeometry.LINEAR: self._linear_vectors(),
            CoordinationGeometry.TRIGONAL_PLANAR: self._trigonal_planar_vectors(),
            CoordinationGeometry.TETRAHEDRAL: self._tetrahedral_vectors(),
            CoordinationGeometry.SQUARE_PLANAR: self._square_planar_vectors(),
            CoordinationGeometry.TRIGONAL_BIPYRAMIDAL: self._trigonal_bipyramidal_vectors(),
            CoordinationGeometry.SQUARE_PYRAMIDAL: self._square_pyramidal_vectors(),
            CoordinationGeometry.OCTAHEDRAL: self._octahedral_vectors(),
        }
    
    def generate_poses(
        self,
        ligand_geometry: Geometry,
        metal: Metal,
        binding_sites: List[BindingSite],
        max_poses: int = 50,
        rmsd_threshold: float = 0.5
    ) -> List[Geometry]:
        """
        Generate metal-ligand complex poses.
        
        Args:
            ligand_geometry: Ligand structure
            metal: Metal properties
            binding_sites: Available binding sites
            max_poses: Maximum number of poses to generate
            rmsd_threshold: RMSD threshold for removing duplicates
            
        Returns:
            List of complex geometries
        """
        poses = []
        
        # Try each preferred geometry
        for coord_geom in metal.preferred_geometries:
            if coord_geom not in self.ideal_geometries:
                continue
            
            coord_number = self._get_coordination_number(coord_geom)
            
            # Skip if not enough binding sites
            if len(binding_sites) < coord_number:
                continue
            
            # Generate poses for this geometry
            geom_poses = self._generate_poses_for_geometry(
                ligand_geometry, metal, binding_sites, 
                coord_geom, max_poses // len(metal.preferred_geometries)
            )
            
            poses.extend(geom_poses)
        
        # Remove duplicates
        if rmsd_threshold > 0:
            poses = self._remove_duplicate_poses(poses, rmsd_threshold)
        
        # Limit to max_poses
        if len(poses) > max_poses:
            poses = poses[:max_poses]
        
        logger.info(f"Generated {len(poses)} metal-ligand poses")
        return poses
    
    def _generate_poses_for_geometry(
        self,
        ligand_geometry: Geometry,
        metal: Metal,
        binding_sites: List[BindingSite],
        coord_geom: CoordinationGeometry,
        max_poses: int
    ) -> List[Geometry]:
        """Generate poses for a specific coordination geometry."""
        poses = []
        coord_number = self._get_coordination_number(coord_geom)
        
        # Get all possible combinations of binding sites
        site_combinations = list(combinations(binding_sites, coord_number))
        
        # Limit combinations if too many
        if len(site_combinations) > max_poses * 2:
            # Prioritize by score
            site_combinations.sort(
                key=lambda x: sum(site.score for site in x), 
                reverse=True
            )
            site_combinations = site_combinations[:max_poses * 2]
        
        for sites in site_combinations:
            # Get coordinating atoms
            coord_atoms = []
            for site in sites:
                # Use first atom of each site for simplicity
                coord_atoms.append(site.atom_indices[0])
            
            # Calculate metal position
            metal_pos = self._calculate_metal_position(
                ligand_geometry, coord_atoms, metal, coord_geom
            )
            
            if metal_pos is not None:
                # Create complex geometry
                complex_geom = self._create_complex_geometry(
                    ligand_geometry, metal, metal_pos, coord_geom
                )
                poses.append(complex_geom)
            
            if len(poses) >= max_poses:
                break
        
        return poses
    
    def _calculate_metal_position(
        self,
        ligand_geometry: Geometry,
        coord_atom_indices: List[int],
        metal: Metal,
        coord_geom: CoordinationGeometry
    ) -> Optional[np.ndarray]:
        """Calculate optimal metal position."""
        # Get positions of coordinating atoms
        coord_positions = ligand_geometry.coordinates[coord_atom_indices]
        
        # Get ideal vectors for this geometry
        ideal_vectors = self.ideal_geometries[coord_geom]
        
        # Estimate metal-ligand distance
        typical_distance = 2.1  # Default
        for idx in coord_atom_indices:
            atom_type = ligand_geometry.atoms[idx]
            if atom_type in metal.typical_bond_lengths:
                typical_distance = metal.typical_bond_lengths[atom_type]
                break
        
        # Find optimal metal position
        # Start from centroid of coordinating atoms
        centroid = np.mean(coord_positions, axis=0)
        
        # Simple approach: place metal at centroid + offset
        # More sophisticated: optimize to match ideal geometry
        metal_pos = centroid + np.array([0, 0, typical_distance])
        
        return metal_pos
    
    def _create_complex_geometry(
        self,
        ligand_geometry: Geometry,
        metal: Metal,
        metal_position: np.ndarray,
        coord_geom: CoordinationGeometry
    ) -> Geometry:
        """Create complex geometry with metal."""
        # Combine ligand atoms with metal
        complex_atoms = ligand_geometry.atoms + [metal.symbol]
        
        # Combine coordinates
        complex_coords = np.vstack([
            ligand_geometry.coordinates,
            metal_position.reshape(1, 3)
        ])
        
        return Geometry(
            atoms=complex_atoms,
            coordinates=complex_coords,
            title=f"{ligand_geometry.title}_{metal.symbol}_{coord_geom.value}"
        )
    
    def _get_coordination_number(self, geom: CoordinationGeometry) -> int:
        """Get coordination number for geometry type."""
        coord_numbers = {
            CoordinationGeometry.LINEAR: 2,
            CoordinationGeometry.TRIGONAL_PLANAR: 3,
            CoordinationGeometry.TETRAHEDRAL: 4,
            CoordinationGeometry.SQUARE_PLANAR: 4,
            CoordinationGeometry.TRIGONAL_BIPYRAMIDAL: 5,
            CoordinationGeometry.SQUARE_PYRAMIDAL: 5,
            CoordinationGeometry.OCTAHEDRAL: 6,
            CoordinationGeometry.PENTAGONAL_BIPYRAMIDAL: 7,
            CoordinationGeometry.CAPPED_OCTAHEDRAL: 7,
            CoordinationGeometry.DODECAHEDRAL: 8,
        }
        return coord_numbers.get(geom, 6)
    
    def _remove_duplicate_poses(
        self,
        poses: List[Geometry],
        rmsd_threshold: float
    ) -> List[Geometry]:
        """Remove duplicate poses based on metal position."""
        unique = []
        
        for pose in poses:
            is_unique = True
            metal_pos = pose.coordinates[-1]  # Metal is last atom
            
            for unique_pose in unique:
                unique_metal_pos = unique_pose.coordinates[-1]
                dist = np.linalg.norm(metal_pos - unique_metal_pos)
                
                if dist < rmsd_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(pose)
        
        return unique
    
    # Ideal geometry vectors
    def _linear_vectors(self) -> np.ndarray:
        """Linear coordination vectors."""
        return np.array([
            [1, 0, 0],
            [-1, 0, 0]
        ])
    
    def _trigonal_planar_vectors(self) -> np.ndarray:
        """Trigonal planar coordination vectors."""
        return np.array([
            [1, 0, 0],
            [-0.5, 0.866, 0],
            [-0.5, -0.866, 0]
        ])
    
    def _tetrahedral_vectors(self) -> np.ndarray:
        """Tetrahedral coordination vectors."""
        return np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ]) / np.sqrt(3)
    
    def _square_planar_vectors(self) -> np.ndarray:
        """Square planar coordination vectors."""
        return np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ])
    
    def _octahedral_vectors(self) -> np.ndarray:
        """Octahedral coordination vectors."""
        return np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]
        ])
    
    def _trigonal_bipyramidal_vectors(self) -> np.ndarray:
        """Trigonal bipyramidal coordination vectors."""
        return np.array([
            [1, 0, 0],
            [-0.5, 0.866, 0],
            [-0.5, -0.866, 0],
            [0, 0, 1],
            [0, 0, -1]
        ])
    
    def _square_pyramidal_vectors(self) -> np.ndarray:
        """Square pyramidal coordination vectors."""
        return np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])