"""XTB optimization interface."""

import subprocess
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..core.data_models import Geometry, OptimizationResult
from ..core.config import XTBConfig
from ..core.logger import logger


class XTBOptimizer:
    """Interface for xTB calculations and optimizations."""
    
    def __init__(self, config: Optional[XTBConfig] = None):
        """
        Initialize XTB optimizer.
        
        Args:
            config: XTB configuration settings
        """
        self.config = config or XTBConfig()
        self.xtb_path = shutil.which("xtb")
        
        if not self.xtb_path:
            logger.warning("xTB not found in PATH. Some features may not work.")
    
    def check_available(self) -> bool:
        """Check if xTB is available."""
        return self.xtb_path is not None
    
    def _get_relative_path(self, path: Path) -> str:
        """Get relative path from current working directory."""
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            # If path is not relative to cwd, just return the name
            return path.name
    
    def optimize(
        self, 
        geometry: Geometry,
        charge: int,
        multiplicity: int = 1,
        work_dir: Optional[Path] = None
    ) -> OptimizationResult:
        """
        Optimize a single geometry with xTB.
        
        Args:
            geometry: Input geometry
            charge: Total charge
            multiplicity: Spin multiplicity (2S+1)
            work_dir: Working directory for calculation
            
        Returns:
            Optimization result
        """
        if not self.check_available():
            return OptimizationResult(
                success=False,
                initial_geometry=geometry,
                error_message="xTB not available"
            )
        
        # Set up working directory
        if work_dir is None:
            work_dir = Path.cwd() / f"xtb_opt_{geometry.title}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Write input geometry
        input_xyz = work_dir / "input.xyz"
        geometry.save_xyz(input_xyz)
        
        # Prepare xTB command
        cmd = [self.xtb_path, str(input_xyz)]
        cmd.extend(["--opt", "normal"])  # Geometry optimization
        cmd.extend(["--chrg", str(charge)])
        
        # Check if this is a metal ion
        is_metal = len(geometry.atoms) == 1 and geometry.atoms[0] in [
            'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi'
        ]
        
        if is_metal:
            # For metal ions, always use --uhf 1
            cmd.extend(["--uhf", "1"])
            logger.debug(f"Using --uhf 1 for metal ion {geometry.atoms[0]}")
        elif multiplicity > 1:
            cmd.extend(["--uhf", str(multiplicity - 1)])
        
        # Add configuration options
        cmd.extend(self.config.to_cmd_args())
        
        # Log the full command with relative paths
        cmd_display = []
        for part in cmd:
            if '/' in part and Path(part).exists():
                cmd_display.append(self._get_relative_path(Path(part)))
            else:
                cmd_display.append(part)
        cmd_str_display = ' '.join(cmd_display)
        logger.debug(f"Running xTB command: {cmd_str_display}")
        logger.debug(f"Working directory: {self._get_relative_path(work_dir)}")
        
        # Run xTB
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Save stdout and stderr to files for debugging
            stdout_file = work_dir / "xtb.out"
            stdout_file.write_text(result.stdout)
            if result.stderr:
                stderr_file = work_dir / "xtb.err"
                stderr_file.write_text(result.stderr)
            
            # Check for successful termination
            # xTB may output "normal termination" to either stdout or stderr
            all_output = result.stdout + "\n" + result.stderr
            terminated_normally = "normal termination of xtb" in all_output
            
            # If xTB terminated normally, it's successful regardless of return code or stderr content
            if not terminated_normally:
                error_msg = result.stderr if result.stderr else "xTB did not terminate normally"
                if "Multiplicity missmatch" in result.stdout:
                    error_msg += " (Warning: Multiplicity mismatch in restart file)"
                return OptimizationResult(
                    success=False,
                    initial_geometry=geometry,
                    error_message=f"xTB failed: {error_msg}"
                )
            
            # Log warnings if present but xTB succeeded
            if result.stderr and "Note:" in result.stderr:
                logger.debug(f"xTB succeeded with warnings: {result.stderr.strip()}")
            
            # Parse results
            opt_xyz = work_dir / "xtbopt.xyz"
            if not opt_xyz.exists():
                return OptimizationResult(
                    success=False,
                    initial_geometry=geometry,
                    error_message="Optimized geometry not found"
                )
            
            # Read optimized geometry
            opt_geometry = self._read_xyz(opt_xyz)
            opt_geometry.title = f"{geometry.title}_opt"
            
            # Parse properties
            properties = self._parse_xtb_output(work_dir)
            opt_geometry.energy = properties.get("total_energy")
            
            return OptimizationResult(
                success=True,
                initial_geometry=geometry,
                optimized_geometry=opt_geometry,
                energy=properties.get("total_energy"),
                properties=properties,
                output_files={
                    "optimized_xyz": opt_xyz,
                    "output": work_dir / "xtb.out",
                    "properties": work_dir / "xtbout.json"
                }
            )
            
        except subprocess.TimeoutExpired:
            return OptimizationResult(
                success=False,
                initial_geometry=geometry,
                error_message="xTB optimization timed out"
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                initial_geometry=geometry,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def single_point(
        self,
        geometry: Geometry,
        charge: int,
        multiplicity: int = 1,
        work_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Perform single-point energy calculation.
        
        Args:
            geometry: Input geometry
            charge: Total charge
            multiplicity: Spin multiplicity
            work_dir: Working directory for calculation
            
        Returns:
            Dictionary with energy and properties
        """
        if not self.check_available():
            logger.error("xTB not available for single-point calculation")
            return {}
        
        # Set up working directory
        if work_dir is None:
            work_dir = Path.cwd() / f"xtb_sp_{geometry.title}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Write input geometry
        input_xyz = work_dir / "input.xyz"
        geometry.save_xyz(input_xyz)
        
        # Prepare xTB command (no --opt flag for single-point)
        cmd = [self.xtb_path, str(input_xyz)]
        cmd.extend(["--chrg", str(charge)])
        
        # Check if this is a metal ion
        is_metal = len(geometry.atoms) == 1 and geometry.atoms[0] in [
            'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi'
        ]
        
        if is_metal:
            # For metal ions, always use --uhf 1
            cmd.extend(["--uhf", "1"])
            logger.debug(f"Using --uhf 1 for metal ion {geometry.atoms[0]}")
        elif multiplicity > 1:
            cmd.extend(["--uhf", str(multiplicity - 1)])
        
        # Add configuration options
        cmd.extend(self.config.to_cmd_args())
        
        # Log the full command with relative paths
        cmd_display = []
        for part in cmd:
            if '/' in part and Path(part).exists():
                cmd_display.append(self._get_relative_path(Path(part)))
            else:
                cmd_display.append(part)
        cmd_str_display = ' '.join(cmd_display)
        logger.info(f"Running xTB single-point command: {cmd_str_display}")
        logger.debug(f"Working directory: {self._get_relative_path(work_dir)}")
        
        # Run xTB
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for single-point
            )
            
            # Save stdout and stderr to files for debugging
            stdout_file = work_dir / "xtb.out"
            stdout_file.write_text(result.stdout)
            if result.stderr:
                stderr_file = work_dir / "xtb.err"
                stderr_file.write_text(result.stderr)
            
            # Check for successful termination
            # xTB may output "normal termination" to either stdout or stderr
            all_output = result.stdout + "\n" + result.stderr
            terminated_normally = "normal termination of xtb" in all_output
            
            # If xTB terminated normally, it's successful regardless of return code or stderr content
            if not terminated_normally:
                error_msg = result.stderr if result.stderr else "xTB did not terminate normally"
                if "Multiplicity missmatch" in result.stdout:
                    error_msg += " (Warning: Multiplicity mismatch in restart file)"
                logger.error(f"xTB single-point failed: {error_msg}")
                return {}
            
            # Log warnings if present but xTB succeeded
            if result.stderr and "Note:" in result.stderr:
                logger.debug(f"xTB single-point succeeded with warnings: {result.stderr.strip()}")
            
            # Parse properties
            properties = self._parse_xtb_output(work_dir)
            
            logger.info(f"Single-point energy: {properties.get('total_energy', 'N/A')} Hartree")
            
            return properties
            
        except subprocess.TimeoutExpired:
            logger.error("xTB single-point calculation timed out")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in single-point calculation: {str(e)}")
            return {}
    
    def _read_xyz(self, filepath: Path) -> Geometry:
        """Read XYZ file."""
        lines = filepath.read_text().strip().split('\n')
        n_atoms = int(lines[0])
        title = lines[1]
        
        atoms = []
        coords = []
        
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms.append(parts[0])
            coords.append([float(parts[j]) for j in range(1, 4)])
        
        return Geometry(
            atoms=atoms,
            coordinates=np.array(coords),
            title=title
        )
    
    def _parse_xtb_output(self, work_dir: Path) -> Dict[str, float]:
        """Parse xTB output files for properties."""
        properties = {}
        
        # Try to read JSON output (newer xTB versions)
        json_file = work_dir / "xtbout.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    xtb_data = json.load(f)
                    
                properties["total_energy"] = xtb_data.get("total energy", 0.0)
                properties["homo_lumo_gap"] = xtb_data.get("HOMO-LUMO gap/eV", 0.0)
                properties["dipole_moment"] = xtb_data.get("dipole", 0.0)
                
                return properties
            except:
                pass
        
        # Fallback: parse text output
        output_file = work_dir / "xtb.out"
        if output_file.exists():
            content = output_file.read_text()
            
            # Initialize variables for HOMO-LUMO gap calculation
            homo = None
            lumo = None
            
            # Parse total energy
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if "TOTAL ENERGY" in line:
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == "Eh" and j > 0:
                            properties["total_energy"] = float(parts[j-1])
                            break
                
                # Parse HOMO-LUMO gap
                if "(HOMO)" in line:
                    try:
                        homo = float(line.split()[-2])
                    except (IndexError, ValueError):
                        pass
                        
                if "(LUMO)" in line:
                    try:
                        lumo = float(line.split()[-2])
                    except (IndexError, ValueError):
                        pass
                
                # Parse dipole moment
                if "molecular dipole:" in line and i + 3 < len(lines):
                    try:
                        dipole_line = lines[i + 3]
                        properties["dipole_moment"] = float(dipole_line.split()[-1])
                    except (IndexError, ValueError):
                        pass
            
            # Calculate HOMO-LUMO gap if both values are available
            if homo is not None and lumo is not None:
                properties["homo_lumo_gap"] = lumo - homo
        
        return properties