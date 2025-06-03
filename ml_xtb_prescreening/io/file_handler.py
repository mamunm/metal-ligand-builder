"""File and directory management utilities."""

import logging
from pathlib import Path
from typing import Dict, Optional
import shutil
import json

logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file and directory operations for metal-ligand analysis."""
    
    def __init__(self, base_dir: Path, experiment_name: str):
        """
        Initialize file handler.
        
        Args:
            base_dir: Base output directory
            experiment_name: Name of the experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.dirs: Dict[str, Path] = {}
        
        # Create directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self) -> None:
        """Create the standard directory structure."""
        # Main experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Define directory structure
        dir_structure = {
            "01_initial_structures": {
                "metals": None,
                "ligands": None,
                "complexes": None
            },
            "02_optimized_structures": {
                "metals": None,
                "ligands": None,
                "complexes": None
            },
            "03_binding_analysis": None,
            "04_orca_inputs": None,
            "05_reports": None
        }
        
        # Create directories
        self._create_nested_dirs(self.experiment_dir, dir_structure)
        
        # Store commonly used paths
        self.dirs["initial"] = self.experiment_dir / "01_initial_structures"
        self.dirs["initial_metals"] = self.dirs["initial"] / "metals"
        self.dirs["initial_ligands"] = self.dirs["initial"] / "ligands"
        self.dirs["initial_complexes"] = self.dirs["initial"] / "complexes"
        
        self.dirs["optimized"] = self.experiment_dir / "02_optimized_structures"
        self.dirs["optimized_metals"] = self.dirs["optimized"] / "metals"
        self.dirs["optimized_ligands"] = self.dirs["optimized"] / "ligands"
        self.dirs["optimized_complexes"] = self.dirs["optimized"] / "complexes"
        
        self.dirs["analysis"] = self.experiment_dir / "03_binding_analysis"
        self.dirs["orca"] = self.experiment_dir / "04_orca_inputs"
        self.dirs["reports"] = self.experiment_dir / "05_reports"
        
        logger.info(f"Created directory structure at: {self.experiment_dir}")
    
    def _create_nested_dirs(self, parent: Path, structure: Dict) -> None:
        """Recursively create nested directory structure."""
        for name, substructure in structure.items():
            dir_path = parent / name
            dir_path.mkdir(exist_ok=True)
            
            if isinstance(substructure, dict):
                self._create_nested_dirs(dir_path, substructure)
    
    def get_metal_filename(self, metal_symbol: str, index: int = 0) -> str:
        """
        Generate filename for metal structure.
        
        Args:
            metal_symbol: Metal element symbol
            index: Structure index (for multiple geometries)
            
        Returns:
            Filename string
        """
        if index == 0:
            return f"{metal_symbol}_ion.xyz"
        return f"{metal_symbol}_ion_{index:03d}.xyz"
    
    def get_ligand_filename(self, ligand_name: str, conformer_id: int) -> str:
        """
        Generate filename for ligand conformer.
        
        Args:
            ligand_name: Name of the ligand
            conformer_id: Conformer index
            
        Returns:
            Filename string
        """
        return f"{ligand_name}_conf_{conformer_id:03d}.xyz"
    
    def get_complex_filename(
        self, 
        ligand_name: str, 
        metal_symbol: str,
        pose_id: int,
        geometry_type: Optional[str] = None
    ) -> str:
        """
        Generate filename for metal-ligand complex.
        
        Args:
            ligand_name: Name of the ligand
            metal_symbol: Metal element symbol
            pose_id: Pose index
            geometry_type: Optional geometry descriptor
            
        Returns:
            Filename string
        """
        base = f"{ligand_name}_{metal_symbol}_pose_{pose_id:03d}"
        if geometry_type:
            base += f"_{geometry_type}"
        return f"{base}.xyz"
    
    def save_metadata(self, metadata: Dict, filename: str) -> Path:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filename: Name of the metadata file
            
        Returns:
            Path to saved file
        """
        metadata_path = self.experiment_dir / filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to: {metadata_path}")
        return metadata_path
    
    def clean_directory(self, dir_name: str) -> None:
        """
        Clean (empty) a specific directory.
        
        Args:
            dir_name: Name of directory to clean
        """
        if dir_name in self.dirs:
            dir_path = self.dirs[dir_name]
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Cleaned directory: {dir_path}")
    
    def copy_file(self, source: Path, dest_dir: str, new_name: Optional[str] = None) -> Path:
        """
        Copy file to a destination directory.
        
        Args:
            source: Source file path
            dest_dir: Destination directory name
            new_name: Optional new filename
            
        Returns:
            Path to copied file
        """
        if dest_dir not in self.dirs:
            raise ValueError(f"Unknown directory: {dest_dir}")
        
        dest_path = self.dirs[dest_dir] / (new_name or source.name)
        shutil.copy2(source, dest_path)
        
        return dest_path