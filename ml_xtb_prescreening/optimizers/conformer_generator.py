"""Conformer generation using RDKit and Open Babel."""

import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from ..core.data_models import Geometry
from ..core.logger import logger


class ConformerGenerator:
    """Generate conformers using RDKit (preferred) or Open Babel (fallback)."""
    
    def __init__(self):
        """Initialize conformer generator."""
        self.rdkit_available = RDKIT_AVAILABLE
        self.obabel_path = shutil.which("obabel")
        
        if self.rdkit_available:
            logger.info("Using RDKit for conformer generation")
        elif self.obabel_path:
            logger.info("Using Open Babel for conformer generation")
        else:
            logger.warning("Neither RDKit nor Open Babel found. Conformer generation unavailable.")
    
    def check_available(self) -> bool:
        """Check if conformer generation is available."""
        return self.rdkit_available or (self.obabel_path is not None)
    
    def from_smiles(
        self,
        smiles: str,
        n_conformers: int = 30,
        energy_window: float = 50.0,  # kcal/mol
        rmsd_threshold: float = 0.5,   # Angstrom
        work_dir: Optional[Path] = None
    ) -> List[Geometry]:
        """
        Generate conformers from SMILES string.
        
        Args:
            smiles: SMILES string
            n_conformers: Number of conformers to generate
            energy_window: Energy window for conformer selection (kcal/mol)
            rmsd_threshold: RMSD threshold for removing duplicates
            work_dir: Working directory
            
        Returns:
            List of conformer geometries
        """
        if not self.check_available():
            logger.error("No conformer generation methods available")
            return []
        
        if work_dir is None:
            work_dir = Path.cwd() / "conformers"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Try RDKit first
        if self.rdkit_available:
            try:
                return self._generate_with_rdkit(smiles, n_conformers, energy_window, rmsd_threshold, work_dir)
            except Exception as e:
                logger.warning(f"RDKit conformer generation failed: {e}")
                logger.info("Falling back to Open Babel")
        
        # Fallback to Open Babel
        if self.obabel_path:
            return self._generate_with_obabel(smiles, n_conformers, energy_window, rmsd_threshold, work_dir)
        
        logger.error("All conformer generation methods failed")
        return []
    
    def _generate_with_rdkit(
        self,
        smiles: str,
        n_conformers: int,
        energy_window: float,
        rmsd_threshold: float,
        work_dir: Path
    ) -> List[Geometry]:
        """Generate conformers using RDKit."""
        logger.info(f"Generating {n_conformers} conformers with RDKit for SMILES: {smiles}")
        
        try:
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            logger.debug(f"Created molecule with {mol.GetNumAtoms()} atoms")
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            logger.debug(f"Added hydrogens: {mol.GetNumAtoms()} atoms total")
            
        except Exception as e:
            logger.error(f"Failed to create molecule from SMILES {smiles}: {e}")
            raise
        
        # Generate conformers
        conf_ids = None
        try:
            # Try ETKDGv3 first (newer RDKit versions)
            logger.debug("Trying ETKDGv3 parameters...")
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = 42  # For reproducibility
            if hasattr(params, 'pruneRmsThresh'):
                params.pruneRmsThresh = rmsd_threshold
            
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=n_conformers,
                params=params
            )
            logger.debug(f"ETKDGv3 generated {len(conf_ids) if conf_ids else 0} conformers")
            
        except (AttributeError, TypeError) as e:
            # Fallback for older RDKit versions
            logger.debug(f"ETKDGv3 failed: {e}")
            logger.debug("Using ETKDG parameters for older RDKit version")
            try:
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers,
                    randomSeed=42,
                    pruneRmsThresh=rmsd_threshold
                )
                logger.debug(f"ETKDG generated {len(conf_ids) if conf_ids else 0} conformers")
            except Exception as e2:
                logger.debug(f"ETKDG also failed: {e2}")
        
        if not conf_ids:
            # Try with more relaxed parameters
            logger.warning("Initial conformer generation failed, trying with relaxed parameters")
            try:
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=n_conformers,
                    randomSeed=42,
                    useRandomCoords=True
                )
                logger.debug(f"Random coords generated {len(conf_ids) if conf_ids else 0} conformers")
            except Exception as e:
                logger.debug(f"Random coords failed: {e}")
                # Final fallback - just try basic embedding
                try:
                    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers)
                    logger.debug(f"Basic embedding generated {len(conf_ids) if conf_ids else 0} conformers")
                except Exception as e2:
                    logger.debug(f"Basic embedding failed: {e2}")
        
        if not conf_ids:
            raise ValueError("Could not generate any conformers with any method")
        
        logger.info(f"Generated {len(conf_ids)} initial conformers")
        
        # Optimize conformers with MMFF
        energies = []
        for conf_id in conf_ids:
            try:
                # Use MMFF94 force field for optimization
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                
                # Calculate energy
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if mmff_props:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                    energy = ff.CalcEnergy()
                else:
                    # Fallback to UFF if MMFF fails
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    energy = ff.CalcEnergy()
                
                energies.append((conf_id, energy))
                
            except Exception as e:
                logger.warning(f"Optimization failed for conformer {conf_id}: {e}")
                energies.append((conf_id, float('inf')))
        
        # Filter by energy window
        energies.sort(key=lambda x: x[1])
        min_energy = energies[0][1]
        
        # Convert kcal/mol to energy units (assuming energy is already in kcal/mol)
        filtered_conformers = []
        for conf_id, energy in energies:
            if energy - min_energy <= energy_window:
                filtered_conformers.append((conf_id, energy))
        
        logger.info(f"After energy filtering: {len(filtered_conformers)} conformers within {energy_window} kcal/mol")
        
        # Convert to Geometry objects
        conformers = []
        for i, (conf_id, energy) in enumerate(filtered_conformers):
            conf = mol.GetConformer(conf_id)
            
            # Extract atoms and coordinates
            atoms = []
            coords = []
            
            for atom_idx in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(atom_idx)
                atoms.append(atom.GetSymbol())
                
                pos = conf.GetAtomPosition(atom_idx)
                coords.append([pos.x, pos.y, pos.z])
            
            geometry = Geometry(
                atoms=atoms,
                coordinates=np.array(coords),
                title=f"conformer_{i:03d}",
                energy=energy,
                properties={"force_field_energy": energy}
            )
            
            conformers.append(geometry)
            
            # Save individual conformer
            xyz_file = work_dir / f"conformer_{i:03d}.xyz"
            geometry.save_xyz(xyz_file)
        
        logger.info(f"Generated {len(conformers)} final conformers")
        return conformers
    
    def _generate_with_obabel(
        self,
        smiles: str,
        n_conformers: int,
        energy_window: float,
        rmsd_threshold: float,
        work_dir: Path
    ) -> List[Geometry]:
        """Generate conformers using Open Babel."""
        logger.info(f"Generating {n_conformers} conformers with Open Babel for SMILES: {smiles}")
        
        # First, generate initial 3D structure
        initial_xyz = work_dir / "initial.xyz"
        cmd = [
            self.obabel_path,
            f"-:{smiles}",
            "-oxyz",
            "--gen3d", "best",
            "-O", str(initial_xyz)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate initial structure: {e.stderr}")
            return []
        
        # Convert to SDF for conformer generation
        sdf_file = work_dir / "molecule.sdf"
        cmd = [self.obabel_path, str(initial_xyz), "-O", str(sdf_file)]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Generate conformers
        conf_sdf = work_dir / "conformers.sdf"
        cmd = [
            self.obabel_path,
            str(sdf_file),
            "-O", str(conf_sdf),
            "--conformer",
            "--nconf", str(n_conformers),
            "--score", "energy",
            "--writeconformers"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Generated conformers: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Conformer generation failed: {e.stderr}")
            return []
        
        # Convert conformers to individual XYZ files
        xyz_prefix = work_dir / "conf"
        cmd = [
            self.obabel_path,
            str(conf_sdf),
            "-oxyz",
            "-m",
            "-O", str(xyz_prefix)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        
        # Read generated conformers
        conformers = []
        for i, xyz_file in enumerate(sorted(work_dir.glob("conf*.xyz"))):
            geom = self._read_xyz(xyz_file)
            geom.title = f"conformer_{i:03d}"
            conformers.append(geom)
        
        logger.info(f"Generated {len(conformers)} conformers")
        
        # Remove duplicates based on RMSD
        if rmsd_threshold > 0:
            conformers = self._remove_duplicates(conformers, rmsd_threshold)
            logger.info(f"After duplicate removal: {len(conformers)} conformers")
        
        return conformers
    
    def _read_xyz(self, filepath: Path) -> Geometry:
        """Read XYZ file."""
        lines = filepath.read_text().strip().split('\n')
        n_atoms = int(lines[0])
        title = lines[1] if len(lines) > 1 else ""
        
        atoms = []
        coords = []
        
        for i in range(2, min(2 + n_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[j]) for j in range(1, 4)])
        
        return Geometry(
            atoms=atoms,
            coordinates=np.array(coords),
            title=title
        )
    
    def _remove_duplicates(
        self, 
        conformers: List[Geometry], 
        rmsd_threshold: float
    ) -> List[Geometry]:
        """Remove duplicate conformers based on RMSD."""
        unique = []
        
        for conf in conformers:
            is_unique = True
            
            for unique_conf in unique:
                if self._calculate_rmsd(conf, unique_conf) < rmsd_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(conf)
        
        return unique
    
    def _calculate_rmsd(self, geom1: Geometry, geom2: Geometry) -> float:
        """Calculate RMSD between two geometries."""
        if len(geom1.atoms) != len(geom2.atoms):
            return float('inf')
        
        # Simple RMSD without alignment
        diff = geom1.coordinates - geom2.coordinates
        rmsd = np.sqrt(np.mean(diff**2))
        
        return rmsd