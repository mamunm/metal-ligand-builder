"""Metal ion geometry generation."""

import logging
from typing import List
import numpy as np

from ..core.data_models import Geometry, Metal

logger = logging.getLogger(__name__)


class MetalGenerator:
    """Generate metal ion geometries."""
    
    def generate_metal_geometry(self, metal: Metal) -> Geometry:
        """
        Generate a simple metal ion geometry.
        
        For bare metal ions, this is just a single atom at the origin.
        
        Args:
            metal: Metal properties
            
        Returns:
            Metal geometry
        """
        return Geometry(
            atoms=[metal.symbol],
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            title=f"{metal.symbol}{metal.charge}+"
        )
    
    def generate_hydrated_metal(
        self, 
        metal: Metal, 
        n_water: int = 6
    ) -> Geometry:
        """
        Generate hydrated metal ion geometry.
        
        Args:
            metal: Metal properties
            n_water: Number of water molecules
            
        Returns:
            Hydrated metal geometry
        """
        # Start with metal at origin
        atoms = [metal.symbol]
        coords = [[0.0, 0.0, 0.0]]
        
        # Add water molecules in octahedral arrangement
        water_distance = metal.typical_bond_lengths.get('O', 2.1)
        
        if n_water == 6:
            # Octahedral
            water_positions = [
                [water_distance, 0, 0],
                [-water_distance, 0, 0],
                [0, water_distance, 0],
                [0, -water_distance, 0],
                [0, 0, water_distance],
                [0, 0, -water_distance]
            ]
        elif n_water == 4:
            # Tetrahedral
            angle = np.arccos(-1/3)
            water_positions = [
                [water_distance, 0, 0],
                [water_distance * np.cos(angle), water_distance * np.sin(angle), 0],
                [water_distance * np.cos(angle), water_distance * np.sin(angle) * np.cos(2*np.pi/3), 
                 water_distance * np.sin(angle) * np.sin(2*np.pi/3)],
                [water_distance * np.cos(angle), water_distance * np.sin(angle) * np.cos(4*np.pi/3),
                 water_distance * np.sin(angle) * np.sin(4*np.pi/3)]
            ]
        else:
            # Simple arrangement
            water_positions = []
            for i in range(n_water):
                angle = 2 * np.pi * i / n_water
                water_positions.append([
                    water_distance * np.cos(angle),
                    water_distance * np.sin(angle),
                    0
                ])
        
        # Add water molecules
        for i, water_pos in enumerate(water_positions[:n_water]):
            # Add oxygen
            atoms.append('O')
            coords.append(water_pos)
            
            # Add hydrogens (simplified geometry)
            h_distance = 0.96  # O-H bond length
            h_angle = 104.5 * np.pi / 180  # H-O-H angle
            
            # Place hydrogens
            direction = np.array(water_pos) / np.linalg.norm(water_pos)
            h1_pos = np.array(water_pos) + h_distance * direction
            atoms.append('H')
            coords.append(h1_pos.tolist())
            
            # Second hydrogen (perpendicular)
            perp = np.cross(direction, [0, 0, 1])
            if np.linalg.norm(perp) < 0.1:
                perp = np.cross(direction, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            
            h2_pos = np.array(water_pos) + h_distance * (
                np.cos(h_angle/2) * direction + np.sin(h_angle/2) * perp
            )
            atoms.append('H')
            coords.append(h2_pos.tolist())
        
        return Geometry(
            atoms=atoms,
            coordinates=np.array(coords),
            title=f"{metal.symbol}{metal.charge}+_{n_water}H2O"
        )