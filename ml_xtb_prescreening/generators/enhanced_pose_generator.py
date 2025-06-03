"""Enhanced metal-ligand pose generation with force field optimization."""

import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from itertools import combinations
import tempfile
import subprocess
import shutil
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom, rdMolAlign
    from rdkit.Chem import rdMolDescriptors
    from rdkit import RDLogger
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.data_models import (
    Geometry, BindingSite, Metal, 
    CoordinationGeometry, BindingSiteType
)
from ..core.logger import logger
from .pose_generator import PoseGenerator


class EnhancedPoseGenerator(PoseGenerator):
    """
    Enhanced pose generator with force field optimization and conformational search.
    
    This generator improves upon the basic pose generator by:
    1. Using force field optimization to refine initial poses
    2. Generating multiple conformations for each pose
    3. Better handling of metal coordination geometry
    """
    
    def __init__(self):
        """Initialize enhanced pose generator."""
        super().__init__()
        self.rdkit_available = RDKIT_AVAILABLE
        self.obabel_path = shutil.which("obabel")
        
        if self.rdkit_available:
            logger.debug("RDKit available for pose optimization")
        elif self.obabel_path:
            logger.debug("OpenBabel available for pose optimization")
        else:
            logger.warning("No force field optimization available")
    
    def generate_poses(
        self,
        ligand_geometry: Geometry,
        metal: Metal,
        binding_sites: List[BindingSite],
        max_poses_per_conformer: int = 25,
        rmsd_threshold: float = 0.5,
        optimize_with_ff: bool = True,
        ff_method: str = "auto"
    ) -> List[Geometry]:
        """
        Generate optimized metal-ligand complex poses.
        
        Args:
            ligand_geometry: Ligand structure
            metal: Metal properties
            binding_sites: Available binding sites
            max_poses_per_conformer: Maximum poses to generate per conformer
            rmsd_threshold: RMSD threshold for removing duplicates
            optimize_with_ff: Whether to optimize with force field
            ff_method: Force field method ("rdkit", "obabel", or "auto")
            
        Returns:
            List of optimized complex geometries
        """
        # Generate initial poses using parent method
        initial_poses = super().generate_poses(
            ligand_geometry, metal, binding_sites, 
            max_poses_per_conformer * 2,  # Generate more initial poses
            rmsd_threshold=0.0  # Don't remove duplicates yet
        )
        
        if not initial_poses:
            logger.warning("No initial poses generated")
            return []
        
        logger.info(f"Generated {len(initial_poses)} initial poses")
        
        # Optimize poses if requested
        if optimize_with_ff and (self.rdkit_available or self.obabel_path):
            # Check if we should use OpenBabel instead for metals
            use_obabel_for_metal = (
                ff_method == "obabel" or 
                (ff_method == "auto" and self.obabel_path and metal.symbol in ["Co", "Fe", "Ni", "Cu", "Zn", "Mn"])
            )
            
            if use_obabel_for_metal and self.obabel_path:
                optimized_poses = self._optimize_poses_with_ff(
                    initial_poses, metal, "obabel"
                )
            else:
                optimized_poses = self._optimize_poses_with_ff(
                    initial_poses, metal, ff_method
                )
            
            # Generate conformations for each optimized pose
            all_conformations = []
            for pose in optimized_poses[:max_poses_per_conformer // 5]:  # Limit base poses
                conformations = self._generate_pose_conformations(
                    pose, metal, n_conformers=5, ff_method=ff_method
                )
                all_conformations.extend(conformations)
            
            # Add original optimized poses
            all_conformations.extend(optimized_poses)
            
            # Remove duplicates
            final_poses = self._remove_duplicate_poses(all_conformations, rmsd_threshold)
            
            # Limit to requested number
            if len(final_poses) > max_poses_per_conformer:
                final_poses = final_poses[:max_poses_per_conformer]
            
            logger.info(f"Generated {len(final_poses)} optimized poses")
            return final_poses
        else:
            # Just remove duplicates and return
            final_poses = self._remove_duplicate_poses(initial_poses, rmsd_threshold)
            if len(final_poses) > max_poses_per_conformer:
                final_poses = final_poses[:max_poses_per_conformer]
            return final_poses
    
    def _optimize_poses_with_ff(
        self,
        poses: List[Geometry],
        metal: Metal,
        ff_method: str = "auto"
    ) -> List[Geometry]:
        """Optimize poses using force field."""
        if ff_method == "auto":
            if self.rdkit_available:
                return self._optimize_with_rdkit(poses, metal)
            elif self.obabel_path:
                return self._optimize_with_obabel(poses, metal)
            else:
                logger.warning("No force field method available")
                return poses
        elif ff_method == "rdkit" and self.rdkit_available:
            return self._optimize_with_rdkit(poses, metal)
        elif ff_method == "obabel" and self.obabel_path:
            return self._optimize_with_obabel(poses, metal)
        else:
            logger.warning(f"Force field method {ff_method} not available")
            return poses
    
    def _optimize_with_rdkit(
        self,
        poses: List[Geometry],
        metal: Metal
    ) -> List[Geometry]:
        """Optimize poses using RDKit UFF with custom metal parameters."""
        optimized = []
        
        for pose in poses:
            try:
                # Convert to RDKit molecule
                mol = self._geometry_to_rdkit_mol(pose, metal)
                if mol is None:
                    logger.warning(f"Could not convert pose {pose.title} to RDKit molecule")
                    optimized.append(pose)
                    continue
                
                # Add constraints to keep metal coordination
                conf = mol.GetConformer()
                metal_idx = mol.GetNumAtoms() - 1  # Metal is last atom
                
                # Find coordinating atoms
                coord_atoms = []
                metal_pos = conf.GetAtomPosition(metal_idx)
                for i in range(mol.GetNumAtoms() - 1):  # Exclude metal
                    atom_pos = conf.GetAtomPosition(i)
                    dist = metal_pos.Distance(atom_pos)
                    if dist < 3.0:  # Within coordination distance
                        coord_atoms.append(i)
                
                # Try optimization with UFF
                try:
                    ff = AllChem.UFFGetMoleculeForceField(mol)
                    if ff is None:
                        # UFF failed, try without metal in force field
                        logger.debug(f"UFF failed for {pose.title}, trying alternative approach")
                        # Just do a simple optimization of ligand atoms
                        ff_props = AllChem.MMFFGetMoleculeProperties(mol)
                        if ff_props:
                            ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props)
                        
                    if ff is not None:
                        # Add distance constraints to maintain coordination
                        for coord_atom in coord_atoms:
                            current_dist = metal_pos.Distance(conf.GetAtomPosition(coord_atom))
                            ff.AddDistanceConstraint(metal_idx, coord_atom, 
                                                   current_dist - 0.1, current_dist + 0.1, 100.0)
                        
                        # Optimize
                        converged = ff.Minimize(maxIts=500)
                    else:
                        logger.debug(f"No force field available for {pose.title}")
                        optimized.append(pose)
                        continue
                        
                except Exception as ff_error:
                    logger.debug(f"Force field optimization error for {pose.title}: {ff_error}")
                    optimized.append(pose)
                    continue
                
                # Convert back to Geometry
                opt_geom = self._rdkit_mol_to_geometry(mol, pose.title + "_opt")
                optimized.append(opt_geom)
                
            except Exception as e:
                logger.warning(f"RDKit optimization failed for {pose.title}: {e}")
                optimized.append(pose)
        
        return optimized
    
    def _optimize_with_obabel(
        self,
        poses: List[Geometry],
        metal: Metal
    ) -> List[Geometry]:
        """Optimize poses using OpenBabel."""
        optimized = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, pose in enumerate(poses):
                try:
                    # Write pose to XYZ file
                    input_file = temp_path / f"pose_{i}.xyz"
                    pose.save_xyz(input_file)
                    
                    # Create output file name
                    output_file = temp_path / f"pose_{i}_opt.xyz"
                    
                    # Run OpenBabel with UFF
                    cmd = [
                        self.obabel_path,
                        str(input_file),
                        "-O", str(output_file),
                        "--minimize",  # Minimize with force field
                        "--ff", "UFF",  # Use UFF force field
                        "--steps", "500",  # Max optimization steps
                        "--sd"  # Steepest descent
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and output_file.exists():
                        # Read optimized geometry
                        opt_geom = self._read_xyz(output_file)
                        opt_geom.title = pose.title + "_opt"
                        optimized.append(opt_geom)
                    else:
                        logger.warning(f"OpenBabel optimization failed for pose {i}")
                        optimized.append(pose)
                        
                except Exception as e:
                    logger.warning(f"OpenBabel optimization error for pose {i}: {e}")
                    optimized.append(pose)
        
        return optimized
    
    def _generate_pose_conformations(
        self,
        pose: Geometry,
        metal: Metal,
        n_conformers: int = 5,
        ff_method: str = "auto"
    ) -> List[Geometry]:
        """Generate conformational variations of a pose."""
        conformations = []
        
        if ff_method == "auto":
            if self.rdkit_available:
                return self._generate_conformations_rdkit(pose, metal, n_conformers)
            elif self.obabel_path:
                return self._generate_conformations_obabel(pose, metal, n_conformers)
        elif ff_method == "rdkit" and self.rdkit_available:
            return self._generate_conformations_rdkit(pose, metal, n_conformers)
        elif ff_method == "obabel" and self.obabel_path:
            return self._generate_conformations_obabel(pose, metal, n_conformers)
        
        return [pose]  # Return original if no method available
    
    def _generate_conformations_rdkit(
        self,
        pose: Geometry,
        metal: Metal,
        n_conformers: int
    ) -> List[Geometry]:
        """Generate conformations using RDKit."""
        try:
            # Convert to RDKit molecule
            mol = self._geometry_to_rdkit_mol(pose, metal)
            if mol is None:
                return [pose]
            
            # Embed multiple conformers
            metal_idx = mol.GetNumAtoms() - 1
            
            # Find coordinating atoms
            coord_atoms = []
            conf0 = mol.GetConformer(0)
            metal_pos = conf0.GetAtomPosition(metal_idx)
            
            for i in range(mol.GetNumAtoms() - 1):
                atom_pos = conf0.GetAtomPosition(i)
                dist = metal_pos.Distance(atom_pos)
                if dist < 3.0:
                    coord_atoms.append(i)
            
            # Generate conformers with constraints
            params = rdDistGeom.ETKDGv3() if hasattr(rdDistGeom, 'ETKDGv3') else None
            if params:
                params.randomSeed = 42
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
            else:
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)
            
            conformations = []
            for conf_id in conf_ids:
                try:
                    # Try to optimize with constraints
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    if ff is None:
                        # Try MMFF as fallback
                        ff_props = AllChem.MMFFGetMoleculeProperties(mol)
                        if ff_props:
                            ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props, confId=conf_id)
                    
                    if ff:
                        # Add constraints
                        for coord_atom in coord_atoms:
                            metal_pos = mol.GetConformer(conf_id).GetAtomPosition(metal_idx)
                            atom_pos = mol.GetConformer(conf_id).GetAtomPosition(coord_atom)
                            dist = metal_pos.Distance(atom_pos)
                            ff.AddDistanceConstraint(metal_idx, coord_atom, dist - 0.2, dist + 0.2, 50.0)
                        
                        ff.Minimize(maxIts=200)
                except Exception as e:
                    logger.debug(f"Conformer optimization failed: {e}")
                
                # Convert to Geometry
                conf_geom = self._rdkit_mol_to_geometry(mol, f"{pose.title}_conf{conf_id}", conf_id)
                conformations.append(conf_geom)
            
            return conformations if conformations else [pose]
            
        except Exception as e:
            logger.warning(f"RDKit conformation generation failed: {e}")
            return [pose]
    
    def _generate_conformations_obabel(
        self,
        pose: Geometry,
        metal: Metal,
        n_conformers: int
    ) -> List[Geometry]:
        """Generate conformations using OpenBabel."""
        conformations = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Write pose
                input_file = temp_path / "pose.xyz"
                pose.save_xyz(input_file)
                
                # Generate conformers
                for i in range(n_conformers):
                    output_file = temp_path / f"conf_{i}.xyz"
                    
                    cmd = [
                        self.obabel_path,
                        str(input_file),
                        "-O", str(output_file),
                        "--conformer",  # Generate conformer
                        "--nconf", "1",  # One at a time
                        "--weighted"  # Use weighted rotor search
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and output_file.exists():
                        conf_geom = self._read_xyz(output_file)
                        conf_geom.title = f"{pose.title}_conf{i}"
                        conformations.append(conf_geom)
                
            except Exception as e:
                logger.warning(f"OpenBabel conformation generation failed: {e}")
        
        return conformations if conformations else [pose]
    
    def _geometry_to_rdkit_mol(self, geometry: Geometry, metal: Metal) -> Optional['Chem.Mol']:
        """Convert Geometry to RDKit molecule."""
        try:
            # Create editable molecule
            mol = Chem.RWMol()
            
            # Add atoms
            for i, atom_symbol in enumerate(geometry.atoms):
                atom = Chem.Atom(atom_symbol)
                # Set formal charge for metal
                if atom_symbol == metal.symbol:
                    atom.SetFormalCharge(metal.charge)
                mol.AddAtom(atom)
            
            # Set 3D coordinates
            conf = Chem.Conformer(len(geometry.atoms))
            for i, coord in enumerate(geometry.coordinates):
                conf.SetAtomPosition(i, coord.tolist())
            mol.AddConformer(conf)
            
            # Try to infer bonds (simple distance-based)
            for i in range(len(geometry.atoms)):
                for j in range(i + 1, len(geometry.atoms)):
                    dist = np.linalg.norm(geometry.coordinates[i] - geometry.coordinates[j])
                    
                    # Skip metal bonds for now
                    if geometry.atoms[i] == metal.symbol or geometry.atoms[j] == metal.symbol:
                        continue
                    
                    # Simple distance criteria
                    if dist < 1.7:  # Typical single bond
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
            
            # Sanitize molecule (skip property sanitization)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)
            
            # Add hydrogens
            mol = Chem.AddHs(mol, addCoords=True)
            
            return mol
            
        except Exception as e:
            logger.debug(f"Failed to convert geometry to RDKit mol: {e}")
            return None
    
    def _read_xyz(self, filepath: Path) -> Geometry:
        """Read XYZ file and return Geometry object."""
        lines = filepath.read_text().strip().split('\n')
        n_atoms = int(lines[0])
        title = lines[1] if len(lines) > 1 else ""
        
        atoms = []
        coords = []
        
        for i in range(2, min(2 + n_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return Geometry(
            atoms=atoms,
            coordinates=np.array(coords),
            title=title
        )
    
    def _rdkit_mol_to_geometry(self, mol: 'Chem.Mol', title: str, conf_id: int = 0) -> Geometry:
        """Convert RDKit molecule to Geometry."""
        conf = mol.GetConformer(conf_id)
        
        atoms = []
        coordinates = []
        
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            atoms.append(atom.GetSymbol())
            
            pos = conf.GetAtomPosition(i)
            coordinates.append([pos.x, pos.y, pos.z])
        
        return Geometry(
            atoms=atoms,
            coordinates=np.array(coordinates),
            title=title
        )