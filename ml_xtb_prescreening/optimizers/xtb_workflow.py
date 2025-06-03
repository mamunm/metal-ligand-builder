"""XTB optimization workflow manager."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import tempfile

try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..core.data_models import Geometry, OptimizationResult
from ..core.config import XTBConfig
from ..core.logger import logger
from .xtb_optimizer import XTBOptimizer


class XTBWorkflowManager:
    """
    Manages XTB optimization workflow for batch processing.
    
    This class handles:
    1. Reading structures from initial folders
    2. Running XTB optimizations in parallel
    3. Organizing results in output folders
    4. Storing optimization data in JSON
    """
    
    def __init__(
        self,
        base_dir: Path,
        config: Optional[XTBConfig] = None,
        n_workers: int = 4,
        create_folders: bool = True
    ):
        """
        Initialize XTB workflow manager.
        
        Args:
            base_dir: Base directory containing experiment
            config: XTB configuration
            n_workers: Number of parallel workers
            create_folders: If True, create folder for each optimization
        """
        self.base_dir = Path(base_dir)
        self.config = config or XTBConfig()
        self.n_workers = n_workers
        self.create_folders = create_folders
        
        # Set up directories
        self.input_dir = self.base_dir / "01_initial_structures"
        self.output_dir = self.base_dir / "02_optimized_structures"
        
        # Initialize optimizer
        self.optimizer = XTBOptimizer(self.config)
        
        # Check if XTB is available
        if not self.optimizer.check_available():
            raise RuntimeError("xTB not found. Please install xTB to use optimization features.")
    
    def _get_relative_path(self, path: Path) -> str:
        """Get relative path from current working directory."""
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            # If path is not relative to cwd, just return the name
            return path.name
    
    def optimize_all_structures(
        self,
        input_folder: Optional[str] = None,
        charge_map: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[OptimizationResult]]:
        """
        Optimize all structures in the input folder.
        
        Args:
            input_folder: Input folder name (default: "01_initial_structures")
            charge_map: Dictionary mapping structure types to charges
                       e.g., {"metal": 2, "ligand": -4, "complex": -2}
        
        Returns:
            Dictionary with optimization results for each structure type
        """
        # Use provided folder or default
        if input_folder:
            self.input_dir = self.base_dir / input_folder
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Default charge map if not provided
        if charge_map is None:
            charge_map = {
                "metals": 2,    # Default metal charge
                "ligands": 0,   # Default ligand charge
                "complexes": 2  # Default complex charge
            }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting XTB optimization workflow")
        logger.debug(f"Input: {self._get_relative_path(self.input_dir)}, Output: {self._get_relative_path(self.output_dir)}")
        logger.debug(f"Workers: {self.n_workers}, Create folders: {self.create_folders}")
        
        results = {}
        
        # Process each structure type
        for struct_type in ["metals", "ligands", "complexes"]:
            struct_dir = self.input_dir / struct_type
            
            if not struct_dir.exists():
                logger.warning(f"Directory not found: {struct_dir}")
                continue
            
            logger.info(f"Processing {struct_type}...")
            
            # Get charge for this structure type
            charge = charge_map.get(struct_type, 0)
            
            # Note if we'll be doing single-point for metals
            if struct_type == "metals":
                logger.debug("Note: Metal ions will use single-point calculations instead of optimization")
            
            # Optimize structures
            type_results = self._optimize_structure_type(
                struct_type, struct_dir, charge
            )
            
            results[struct_type] = type_results
            
            # Log summary only
            successful = len([r for r in type_results if r.success])
            total = len(type_results)
            if successful == total:
                logger.info(f"Completed {struct_type}: {successful}/{total} successful ✓")
            else:
                logger.info(f"Completed {struct_type}: {successful}/{total} successful ({total - successful} failed)")
        
        # Save overall summary
        self._save_workflow_summary(results)
        
        return results
    
    def _optimize_structure_type(
        self,
        struct_type: str,
        input_dir: Path,
        charge: int
    ) -> List[OptimizationResult]:
        """
        Optimize all structures of a given type.
        
        Args:
            struct_type: Type of structure (metals/ligands/complexes)
            input_dir: Directory containing XYZ files
            charge: Charge for this structure type
            
        Returns:
            List of optimization results
        """
        # Create output directory for this type
        output_type_dir = self.output_dir / struct_type
        output_type_dir.mkdir(exist_ok=True)
        
        # Find all XYZ files
        xyz_files = list(input_dir.glob("*.xyz"))
        
        if not xyz_files:
            logger.warning(f"No XYZ files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(xyz_files)} {struct_type} to optimize")
        
        # Prepare optimization tasks
        tasks = []
        for xyz_file in xyz_files:
            tasks.append({
                'xyz_file': xyz_file,
                'struct_type': struct_type,
                'charge': charge,
                'output_dir': output_type_dir
            })
        
        # Run optimizations in parallel
        results = []
        
        if RICH_AVAILABLE and len(tasks) > 1:
            # Use rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=Console(stderr=True),  # Use stderr to not interfere with stdout
                transient=True  # Remove progress bar when done
            ) as progress:
                task_id = progress.add_task(f"[cyan]Optimizing {struct_type}...", total=len(tasks))
                
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    # Submit all tasks
                    future_to_task = {
                        executor.submit(
                            self._optimize_single_structure,
                            task['xyz_file'],
                            task['charge'],
                            task['output_dir'],
                            struct_type
                        ): task
                        for task in tasks
                    }
                    
                    # Process completed tasks
                    completed = 0
                    failed = 0
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result.success:
                                completed += 1
                            else:
                                failed += 1
                                
                        except Exception as e:
                            failed += 1
                            # Create failed result
                            geom = self._read_xyz(task['xyz_file'])
                            results.append(OptimizationResult(
                                success=False,
                                initial_geometry=geom,
                                error_message=str(e)
                            ))
                        
                        # Update progress
                        progress.update(task_id, advance=1, 
                                      description=f"[cyan]Optimizing {struct_type}... [green]{completed} done[/green], [red]{failed} failed[/red]")
        else:
            # Fallback to original implementation for single files or no rich
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(
                        self._optimize_single_structure,
                        task['xyz_file'],
                        task['charge'],
                        task['output_dir'],
                        struct_type
                    ): task
                    for task in tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result.success:
                            logger.info(f"✓ {task['xyz_file'].name}")
                        else:
                            logger.warning(f"✗ {task['xyz_file'].name}: {result.error_message}")
                            
                    except Exception as e:
                        logger.error(f"Failed to optimize {task['xyz_file'].name}: {str(e)}")
                        
                        # Create failed result
                        geom = self._read_xyz(task['xyz_file'])
                        results.append(OptimizationResult(
                            success=False,
                            initial_geometry=geom,
                            error_message=str(e)
                        ))
        
        # Save results summary for this structure type
        self._save_type_summary(struct_type, results, output_type_dir)
        
        return results
    
    def _optimize_single_structure(
        self,
        xyz_file: Path,
        charge: int,
        output_dir: Path,
        struct_type: str = None
    ) -> OptimizationResult:
        """
        Optimize a single structure (or run single-point for metals).
        
        Args:
            xyz_file: Path to XYZ file
            charge: Molecular charge
            output_dir: Output directory
            struct_type: Type of structure ('metals', 'ligands', 'complexes')
            
        Returns:
            Optimization result
        """
        # Read initial geometry
        geometry = self._read_xyz(xyz_file)
        
        # Check if this is a single atom (metal ion)
        is_single_atom = len(geometry.atoms) == 1
        is_metal_type = struct_type == "metals" if struct_type else False
        
        # Determine work directory
        if self.create_folders:
            # Create folder for this optimization
            work_dir = output_dir / xyz_file.stem
            work_dir.mkdir(exist_ok=True)
        else:
            # Use temporary directory
            work_dir = Path(tempfile.mkdtemp(prefix=f"xtb_{xyz_file.stem}_"))
        
        try:
            # For single atoms (typically metal ions), run single-point calculation
            if is_single_atom or is_metal_type:
                logger.info(f"Running single-point calculation for {xyz_file.stem} (single atom)")
                
                # Determine multiplicity for metal ions
                # For most transition metals with +2 charge, we need specific multiplicities
                metal_multiplicity = 1
                if is_single_atom and len(geometry.atoms) == 1:
                    atom_symbol = geometry.atoms[0]
                    # Common transition metals with their typical multiplicities for +2 state
                    metal_multiplicities = {
                        'Co': 4,  # Co2+ is d7, high spin quartet
                        'Ni': 3,  # Ni2+ is d8, triplet
                        'Cu': 2,  # Cu2+ is d9, doublet
                        'Fe': 5,  # Fe2+ is d6, high spin quintet
                        'Mn': 6,  # Mn2+ is d5, high spin sextet
                        'Cr': 5,  # Cr2+ is d4, high spin quintet
                        'V':  4,  # V2+ is d3, quartet
                        'Ti': 3,  # Ti2+ is d2, triplet
                        'Sc': 2,  # Sc2+ is d1, doublet
                        'Zn': 1,  # Zn2+ is d10, singlet
                    }
                    metal_multiplicity = metal_multiplicities.get(atom_symbol, 1)
                    logger.debug(f"Using multiplicity {metal_multiplicity} for {atom_symbol}{charge}+")
                
                # Run single-point calculation
                properties = self.optimizer.single_point(
                    geometry=geometry,
                    charge=charge,
                    multiplicity=metal_multiplicity,
                    work_dir=work_dir
                )
                
                if properties:
                    # Create OptimizationResult from single-point
                    result = OptimizationResult(
                        success=True,
                        initial_geometry=geometry,
                        optimized_geometry=geometry,  # Same geometry for single-point
                        energy=properties.get('total_energy'),
                        properties=properties,
                        output_files={
                            'output': work_dir / 'xtb.out',
                            'properties': work_dir / 'xtbout.json'
                        }
                    )
                else:
                    result = OptimizationResult(
                        success=False,
                        initial_geometry=geometry,
                        error_message="Single-point calculation failed"
                    )
            else:
                # Run optimization for multi-atom structures
                result = self.optimizer.optimize(
                    geometry=geometry,
                    charge=charge,
                    multiplicity=1,  # Default to singlet
                    work_dir=work_dir
                )
            
            # If successful and not creating folders, copy key files
            if result.success and not self.create_folders:
                if result.optimized_geometry:
                    # Save geometry (optimized or original for single-point)
                    suffix = "_sp" if is_single_atom else "_opt"
                    out_xyz = output_dir / f"{xyz_file.stem}{suffix}.xyz"
                    result.optimized_geometry.save_xyz(out_xyz)
                    
                    # Update result paths
                    result.output_files['optimized_xyz'] = out_xyz
                
                # Clean up temp directory
                shutil.rmtree(work_dir, ignore_errors=True)
            
            return result
            
        except Exception as e:
            # Clean up on error
            if not self.create_folders and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
            
            return OptimizationResult(
                success=False,
                initial_geometry=geometry,
                error_message=str(e)
            )
    
    def _read_xyz(self, filepath: Path) -> Geometry:
        """Read XYZ file into Geometry object."""
        lines = filepath.read_text().strip().split('\n')
        n_atoms = int(lines[0])
        title = lines[1] if len(lines) > 1 else filepath.stem
        
        atoms = []
        coords = []
        
        for i in range(2, min(2 + n_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                atoms.append(parts[0])
                coords.append([float(parts[j]) for j in range(1, 4)])
        
        import numpy as np
        return Geometry(
            atoms=atoms,
            coordinates=np.array(coords),
            title=title
        )
    
    def _save_type_summary(
        self,
        struct_type: str,
        results: List[OptimizationResult],
        output_dir: Path
    ) -> None:
        """Save summary for a structure type."""
        summary = {
            "structure_type": struct_type,
            "timestamp": datetime.now().isoformat(),
            "n_structures": len(results),
            "n_successful": len([r for r in results if r.success]),
            "n_failed": len([r for r in results if not r.success]),
            "structures": []
        }
        
        for result in results:
            struct_data = {
                "name": result.initial_geometry.title,
                "success": result.success,
                "initial_energy": None,
                "final_energy": result.energy,
                "energy_change": None,
                "convergence": result.convergence_info if result.success else None,
                "properties": result.properties if result.success else None,
                "error": result.error_message if not result.success else None
            }
            
            if result.success and result.energy is not None:
                # Calculate energy change if possible
                if hasattr(result.initial_geometry, 'energy') and result.initial_geometry.energy:
                    struct_data["initial_energy"] = result.initial_geometry.energy
                    struct_data["energy_change"] = result.energy - result.initial_geometry.energy
            
            summary["structures"].append(struct_data)
        
        # Sort by energy if available
        summary["structures"].sort(
            key=lambda x: x["final_energy"] if x["final_energy"] is not None else float('inf')
        )
        
        # Save summary
        summary_file = output_dir / f"{struct_type}_optimization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {struct_type} summary to {self._get_relative_path(summary_file)}")
    
    def _save_workflow_summary(self, results: Dict[str, List[OptimizationResult]]) -> None:
        """Save overall workflow summary."""
        summary = {
            "workflow": "XTB Optimization/Single-Point",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "method": self.config.method,
                "solvent": self.config.solvent,
                "convergence": self.config.convergence,
                "n_workers": self.n_workers,
                "create_folders": self.create_folders
            },
            "notes": {
                "metals": "Single-point calculations (no optimization for single atoms)",
                "ligands": "Full geometry optimization",
                "complexes": "Full geometry optimization"
            },
            "results": {}
        }
        
        for struct_type, type_results in results.items():
            summary["results"][struct_type] = {
                "total": len(type_results),
                "successful": len([r for r in type_results if r.success]),
                "failed": len([r for r in type_results if not r.success])
            }
        
        # Save summary
        summary_file = self.output_dir / "optimization_workflow_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Workflow summary saved to {self._get_relative_path(summary_file)}")