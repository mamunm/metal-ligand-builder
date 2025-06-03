"""Binding site detection for ligands."""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from ..core.data_models import BindingSite, BindingSiteType, Geometry

logger = logging.getLogger(__name__)


class BindingSiteDetector:
    """Detect and analyze potential metal binding sites in ligands."""
    
    def __init__(self):
        """Initialize binding site detector."""
        # Define binding atom types and their properties
        self.binding_atoms = {
            'O': {'vdw_radius': 1.52, 'electronegativity': 3.44},
            'N': {'vdw_radius': 1.55, 'electronegativity': 3.04},
            'S': {'vdw_radius': 1.80, 'electronegativity': 2.58},
            'P': {'vdw_radius': 1.80, 'electronegativity': 2.19},
        }
        
        # Distance thresholds for functional group detection
        self.bond_thresholds = {
            ('C', 'O'): 1.5,   # C-O single bond
            ('C', 'N'): 1.5,   # C-N single bond
            ('C', 'C'): 1.6,   # C-C single bond
            ('O', 'H'): 1.2,   # O-H bond
            ('N', 'H'): 1.2,   # N-H bond
        }
    
    def detect_sites(self, geometry: Geometry) -> List[BindingSite]:
        """
        Detect all potential binding sites in a ligand.
        
        Args:
            geometry: Ligand geometry
            
        Returns:
            List of binding sites
        """
        sites = []
        
        # Find all potential binding atoms
        binding_indices = []
        for i, atom in enumerate(geometry.atoms):
            if atom in self.binding_atoms:
                binding_indices.append(i)
        
        if not binding_indices:
            logger.warning("No potential binding atoms found")
            return sites
        
        # Detect functional groups
        functional_groups = self._detect_functional_groups(geometry)
        
        # Create binding sites for each functional group
        for fg_type, atom_indices in functional_groups.items():
            if fg_type == BindingSiteType.CARBOXYLATE:
                # For carboxylates, use both oxygens
                for idx_group in atom_indices:
                    site = self._create_carboxylate_site(geometry, idx_group)
                    if site:
                        sites.append(site)
                        
            elif fg_type == BindingSiteType.AMINE:
                # For amines, use nitrogen
                for idx in atom_indices:
                    site = self._create_amine_site(geometry, idx)
                    if site:
                        sites.append(site)
                        
            elif fg_type == BindingSiteType.HYDROXYL:
                # For hydroxyls, use oxygen
                for idx in atom_indices:
                    site = self._create_hydroxyl_site(geometry, idx)
                    if site:
                        sites.append(site)
                        
            elif fg_type == BindingSiteType.SULFUR:
                # For sulfur atoms
                for idx in atom_indices:
                    site = self._create_sulfur_site(geometry, idx)
                    if site:
                        sites.append(site)
        
        # Cluster nearby sites
        sites = self._cluster_sites(sites, threshold=3.0)
        
        logger.info(f"Found {len(sites)} binding sites")
        return sites
    
    def _detect_functional_groups(
        self, 
        geometry: Geometry
    ) -> Dict[BindingSiteType, List[Any]]:
        """Detect functional groups in the ligand."""
        groups = {
            BindingSiteType.CARBOXYLATE: [],
            BindingSiteType.AMINE: [],
            BindingSiteType.HYDROXYL: [],
            BindingSiteType.IMIDAZOLE: [],
            BindingSiteType.PHOSPHATE: [],
            BindingSiteType.SULFUR: []
        }
        
        # Build connectivity
        connectivity = self._build_connectivity(geometry)
        
        # Detect carboxylates (COO-)
        for i, atom in enumerate(geometry.atoms):
            if atom == 'C':
                neighbors = connectivity.get(i, [])
                oxygens = [n for n in neighbors if geometry.atoms[n] == 'O']
                
                if len(oxygens) >= 2:
                    # Check if it's a carboxylate (C with 2 O neighbors)
                    groups[BindingSiteType.CARBOXYLATE].append(oxygens)
        
        # Detect amines
        for i, atom in enumerate(geometry.atoms):
            if atom == 'N':
                neighbors = connectivity.get(i, [])
                hydrogens = [n for n in neighbors if geometry.atoms[n] == 'H']
                
                if len(neighbors) <= 3:  # Primary, secondary, or tertiary amine
                    groups[BindingSiteType.AMINE].append(i)
        
        # Detect hydroxyls
        for i, atom in enumerate(geometry.atoms):
            if atom == 'O':
                neighbors = connectivity.get(i, [])
                
                if len(neighbors) == 2:  # O with 2 neighbors (likely -OH)
                    has_h = any(geometry.atoms[n] == 'H' for n in neighbors)
                    if has_h:
                        groups[BindingSiteType.HYDROXYL].append(i)
        
        # Detect sulfur groups
        for i, atom in enumerate(geometry.atoms):
            if atom == 'S':
                groups[BindingSiteType.SULFUR].append(i)
        
        return groups
    
    def _build_connectivity(self, geometry: Geometry) -> Dict[int, List[int]]:
        """Build connectivity graph based on distances."""
        connectivity = {}
        n_atoms = len(geometry.atoms)
        
        for i in range(n_atoms):
            connectivity[i] = []
            
            for j in range(n_atoms):
                if i == j:
                    continue
                
                dist = np.linalg.norm(
                    geometry.coordinates[i] - geometry.coordinates[j]
                )
                
                # Check if within bonding distance
                atom_i = geometry.atoms[i]
                atom_j = geometry.atoms[j]
                
                threshold = self.bond_thresholds.get(
                    (atom_i, atom_j),
                    self.bond_thresholds.get((atom_j, atom_i), 1.8)
                )
                
                if dist < threshold:
                    connectivity[i].append(j)
        
        return connectivity
    
    def _create_carboxylate_site(
        self, 
        geometry: Geometry, 
        oxygen_indices: List[int]
    ) -> BindingSite:
        """Create binding site for carboxylate group."""
        # Use center of oxygens as binding position
        positions = geometry.coordinates[oxygen_indices]
        center = np.mean(positions, axis=0)
        
        return BindingSite(
            atom_indices=oxygen_indices,
            site_type=BindingSiteType.CARBOXYLATE,
            score=0.9,  # Carboxylates are excellent binding sites
            position=center,
            functional_group_info={
                'n_oxygens': len(oxygen_indices),
                'denticity': 2  # Bidentate
            }
        )
    
    def _create_amine_site(
        self, 
        geometry: Geometry, 
        nitrogen_index: int
    ) -> BindingSite:
        """Create binding site for amine group."""
        return BindingSite(
            atom_indices=[nitrogen_index],
            site_type=BindingSiteType.AMINE,
            score=0.7,  # Amines are good binding sites
            position=geometry.coordinates[nitrogen_index],
            functional_group_info={
                'denticity': 1  # Monodentate
            }
        )
    
    def _create_hydroxyl_site(
        self, 
        geometry: Geometry, 
        oxygen_index: int
    ) -> BindingSite:
        """Create binding site for hydroxyl group."""
        return BindingSite(
            atom_indices=[oxygen_index],
            site_type=BindingSiteType.HYDROXYL,
            score=0.5,  # Hydroxyls are moderate binding sites
            position=geometry.coordinates[oxygen_index],
            functional_group_info={
                'denticity': 1  # Monodentate
            }
        )
    
    def _create_sulfur_site(
        self, 
        geometry: Geometry, 
        sulfur_index: int
    ) -> BindingSite:
        """Create binding site for sulfur atom."""
        return BindingSite(
            atom_indices=[sulfur_index],
            site_type=BindingSiteType.SULFUR,
            score=0.6,  # Sulfur is a soft donor, good for soft metals
            position=geometry.coordinates[sulfur_index],
            functional_group_info={
                'denticity': 1  # Monodentate
            }
        )
    
    def _cluster_sites(
        self, 
        sites: List[BindingSite], 
        threshold: float = 3.0
    ) -> List[BindingSite]:
        """
        Cluster binding sites that are close together.
        
        This helps identify chelating binding modes.
        """
        if len(sites) <= 1:
            return sites
        
        # Calculate distances between sites
        positions = np.array([site.position for site in sites])
        distances = cdist(positions, positions)
        
        # Simple clustering: group sites within threshold
        clustered = []
        used = set()
        
        for i, site_i in enumerate(sites):
            if i in used:
                continue
            
            # Find all sites within threshold
            cluster_indices = [i]
            for j in range(i + 1, len(sites)):
                if j not in used and distances[i, j] < threshold:
                    cluster_indices.append(j)
            
            # Mark as used
            used.update(cluster_indices)
            
            if len(cluster_indices) > 1:
                # Create merged site
                all_indices = []
                for idx in cluster_indices:
                    all_indices.extend(sites[idx].atom_indices)
                
                center = np.mean([sites[idx].position for idx in cluster_indices], axis=0)
                
                merged_site = BindingSite(
                    atom_indices=all_indices,
                    site_type=sites[cluster_indices[0]].site_type,
                    score=max(sites[idx].score for idx in cluster_indices),
                    position=center,
                    functional_group_info={
                        'merged': True,
                        'n_sites': len(cluster_indices),
                        'denticity': len(cluster_indices)
                    }
                )
                clustered.append(merged_site)
            else:
                clustered.append(site_i)
        
        return clustered