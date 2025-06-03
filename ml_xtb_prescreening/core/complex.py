"""Core class for metal-ligand complex analysis."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import json

from .config import ComplexConfig
from .data_models import (
    Metal, Ligand, BindingSite, Geometry, 
    OptimizationResult, BindingEnergyResult
)
from .logger import logger
from ..io.file_handler import FileHandler
from ..io.orca_generator import ORCAGenerator
from ..optimizers.conformer_generator import ConformerGenerator
from ..optimizers.xtb_workflow import XTBWorkflowManager
from ..generators.metal_generator import MetalGenerator
from ..generators.binding_site_detector import BindingSiteDetector
from ..generators.pose_generator import PoseGenerator
from ..generators.enhanced_pose_generator import EnhancedPoseGenerator
from ..analysis.report_generator import ReportGenerator


class MetalLigandComplex:
    """
    Main class for analyzing metal-ligand complexes.
    
    This class handles the complete workflow:
    1. Structure generation (metal, ligand, complex)
    2. XTB optimization
    3. Binding energy calculation
    4. ORCA input preparation
    
    Attributes:
        metal: Metal ion properties
        ligand: Ligand properties
        config: Configuration settings
        binding_sites: Detected binding sites
        results: Dictionary storing all results
    """
    
    def __init__(
        self,
        ligand_name: str,
        ligand_smiles: str,
        metal_symbol: str,
        metal_charge: int,
        ligand_charge: int,
        ligand_protonation_state: str = "deprotonated",
        experiment_name: Optional[str] = None,
        config: Optional[ComplexConfig] = None
    ):
        """
        Initialize metal-ligand complex analysis.
        
        Args:
            ligand_name: Name of the ligand
            ligand_smiles: SMILES string of the ligand
            metal_symbol: Metal element symbol (e.g., 'Co', 'Ni')
            metal_charge: Charge of the metal ion
            ligand_charge: Charge of the ligand
            ligand_protonation_state: Protonation state ('neutral', 'protonated', 'deprotonated')
            experiment_name: Optional name for the experiment
            config: Configuration object (uses defaults if None)
        """
        self.ligand = Ligand(
            name=ligand_name,
            smiles=ligand_smiles,
            charge=ligand_charge,
            protonation_state=ligand_protonation_state
        )
        
        self.metal = Metal(
            symbol=metal_symbol,
            charge=metal_charge
        )
        
        self.config = config or ComplexConfig()
        
        if experiment_name:
            self.config.experiment_name = experiment_name
            
        # Set up file handler for directory management
        self.file_handler = FileHandler(
            Path.cwd(),  # Use current working directory
            self.config.experiment_name
        )
        self.work_dir = self.file_handler.experiment_dir
        
        # Initialize generators
        self.conformer_generator = ConformerGenerator()
        self.metal_generator = MetalGenerator()
        self.binding_site_detector = BindingSiteDetector()
        # Use enhanced pose generator if force field optimization is enabled
        if self.config.optimize_poses_with_ff:
            self.pose_generator = EnhancedPoseGenerator()
        else:
            self.pose_generator = PoseGenerator()
        
        # Initialize result storage
        self.binding_sites: List[BindingSite] = []
        self.results: Dict[str, Any] = {
            'ligand_conformers': [],
            'metal_geometries': [],
            'complex_poses': [],
            'optimized_ligands': [],
            'optimized_metals': [],
            'optimized_complexes': [],
            'binding_energies': [],
            'orca_inputs': []
        }
        
        logger.info(f"Initialized {self.metal.symbol}-{self.ligand.name} complex analysis")
        logger.info(f"Working directory: {self._get_relative_path(self.work_dir)}")
        logger.debug(f"Metal: {self.metal.symbol}{self.metal.charge:+d}, Ligand: {self.ligand.name}{self.ligand.charge:+d}")
    
    def _get_relative_path(self, path: Path) -> str:
        """Get relative path from current working directory."""
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            # If path is not relative to cwd, just return the name
            return path.name
    
    def generate_ligand_conformers(self) -> List[Geometry]:
        """
        Generate ligand conformers using Open Babel.
        
        This method:
        1. Generates 3D conformers from SMILES
        2. Saves them to 01_initial_structures/ligands/
        3. Updates the ligand object with the initial structure
        
        Returns:
            List of ligand conformer geometries
        """
        logger.info(f"Generating {self.config.n_conformers} ligand conformers for {self.ligand.name}...")
        
        # Check if conformer generator is available
        if not self.conformer_generator.check_available():
            logger.error("Open Babel not available for conformer generation")
            return []
        
        # Generate conformers from SMILES
        work_dir = self.file_handler.dirs["initial_ligands"] / f"{self.ligand.name}_workdir"
        work_dir.mkdir(exist_ok=True)
        
        conformers = self.conformer_generator.from_smiles(
            smiles=self.ligand.smiles,
            n_conformers=self.config.n_conformers,
            energy_window=50.0,  # kcal/mol - wide window for initial generation
            rmsd_threshold=self.config.rmsd_threshold,
            work_dir=work_dir
        )
        
        if not conformers:
            logger.error(f"Failed to generate conformers for {self.ligand.name}")
            return []
        
        # Save conformers to proper directory
        saved_conformers = []
        for i, conformer in enumerate(conformers):
            # Update conformer title
            conformer.title = f"{self.ligand.name}_conformer_{i}"
            
            # Generate filename and save
            filename = self.file_handler.get_ligand_filename(self.ligand.name, i)
            output_path = self.file_handler.dirs["initial_ligands"] / filename
            conformer.save_xyz(output_path)
            
            saved_conformers.append(conformer)
            
            # Store first conformer as reference
            if i == 0:
                self.ligand.xyz_path = output_path
                self.ligand.atoms = conformer.atoms
                self.ligand.coordinates = conformer.coordinates
        
        logger.info(f"Generated and saved {len(saved_conformers)} conformers to {self._get_relative_path(self.file_handler.dirs['initial_ligands'])}")
        
        # Clean up work directory
        import shutil
        if work_dir.exists():
            shutil.rmtree(work_dir)
        
        # Store in results
        self.results['ligand_conformers'] = saved_conformers
        
        # Save metadata
        metadata = {
            "ligand_name": self.ligand.name,
            "smiles": self.ligand.smiles,
            "charge": self.ligand.charge,
            "protonation_state": self.ligand.protonation_state,
            "n_conformers": len(saved_conformers),
            "conformer_files": [
                str(self.file_handler.dirs["initial_ligands"] / self.file_handler.get_ligand_filename(self.ligand.name, i))
                for i in range(len(saved_conformers))
            ]
        }
        self.file_handler.save_metadata(metadata, "ligand_generation_metadata.json")
        
        return saved_conformers
    
    def generate_metal_geometries(self) -> List[Geometry]:
        """
        Generate metal ion geometries in vacuum.
        
        This method:
        1. Creates bare metal ion geometry (single atom)
        2. Optionally creates hydrated metal complexes
        3. Saves them to 01_initial_structures/metals/
        
        For ions, this typically returns a single geometry,
        but can generate multiple coordination environments if needed.
        
        Returns:
            List of metal geometries
        """
        logger.info(f"Generating metal geometries for {self.metal.symbol}{self.metal.charge}+...")
        
        metal_geometries = []
        
        # 1. Generate bare metal ion (always)
        bare_metal = self.metal_generator.generate_metal_geometry(self.metal)
        bare_metal.title = f"{self.metal.symbol}{self.metal.charge}+_bare"
        
        # Save bare metal
        filename = self.file_handler.get_metal_filename(self.metal.symbol, 0)
        output_path = self.file_handler.dirs["initial_metals"] / filename
        bare_metal.save_xyz(output_path)
        metal_geometries.append(bare_metal)
        
        logger.info(f"Generated bare metal ion: {filename}")
        
        # 2. Optionally generate hydrated metal complexes
        # This can be useful for comparison and for metals that are never bare in solution
        if hasattr(self.config, 'generate_hydrated_metals') and self.config.generate_hydrated_metals:
            # Common hydration numbers based on metal
            hydration_numbers = {
                "Li": [4],
                "Na": [6],
                "K": [6, 8],
                "Mg": [6],
                "Ca": [6, 8],
                "Fe": [6],
                "Co": [6],
                "Ni": [6],
                "Cu": [4, 6],  # Cu can have Jahn-Teller distortion
                "Zn": [4, 6],
                "Al": [6],
                "Cr": [6],
                "Mn": [6]
            }
            
            n_waters = hydration_numbers.get(self.metal.symbol, [6])
            
            for i, n_water in enumerate(n_waters, 1):
                hydrated = self.metal_generator.generate_hydrated_metal(self.metal, n_water)
                hydrated.title = f"{self.metal.symbol}{self.metal.charge}+_{n_water}H2O"
                
                filename = self.file_handler.get_metal_filename(f"{self.metal.symbol}_{n_water}H2O", i)
                output_path = self.file_handler.dirs["initial_metals"] / filename
                hydrated.save_xyz(output_path)
                metal_geometries.append(hydrated)
                
                logger.info(f"Generated hydrated metal complex: {filename}")
        
        # Store in results
        self.results['metal_geometries'] = metal_geometries
        
        # Save metadata
        metadata = {
            "metal_symbol": self.metal.symbol,
            "metal_charge": self.metal.charge,
            "coordination_numbers": self.metal.coordination_numbers,
            "preferred_geometries": [g.value for g in self.metal.preferred_geometries],
            "n_geometries": len(metal_geometries),
            "geometry_files": [
                str(self.file_handler.dirs["initial_metals"] / self.file_handler.get_metal_filename(
                    self.metal.symbol if i == 0 else f"{self.metal.symbol}_{geom.title.split('_')[-1]}", i
                ))
                for i, geom in enumerate(metal_geometries)
            ],
            "includes_hydrated": len(metal_geometries) > 1
        }
        self.file_handler.save_metadata(metadata, "metal_generation_metadata.json")
        
        logger.info(f"Generated {len(metal_geometries)} metal geometries")
        
        return metal_geometries
    
    def detect_binding_sites(self) -> List[BindingSite]:
        """
        Detect potential binding sites on the ligand.
        
        This method analyzes the ligand structure to find:
        - Carboxylate groups
        - Amine groups
        - Hydroxyl groups
        - Other potential coordinating atoms
        
        Returns:
            List of binding sites with scores
        """
        logger.info("Detecting binding sites on ligand...")
        
        # Make sure we have ligand structure
        if not self.ligand.atoms or self.ligand.coordinates is None:
            if not self.results['ligand_conformers']:
                logger.error("No ligand structure available. Run generate_ligand_conformers first.")
                return []
            
            # Use first conformer
            first_conformer = self.results['ligand_conformers'][0]
            self.ligand.atoms = first_conformer.atoms
            self.ligand.coordinates = first_conformer.coordinates
        
        # Create geometry object for binding site detection
        ligand_geometry = Geometry(
            atoms=self.ligand.atoms,
            coordinates=self.ligand.coordinates,
            title=self.ligand.name
        )
        
        # Detect binding sites
        self.binding_sites = self.binding_site_detector.detect_sites(ligand_geometry)
        
        logger.info(f"Found {len(self.binding_sites)} potential binding sites:")
        for i, site in enumerate(self.binding_sites):
            logger.info(f"  Site {i+1}: {site.site_type.value} "
                       f"(atoms: {site.atom_indices}, score: {site.score:.2f})")
        
        # Save binding site information
        metadata = {
            "ligand_name": self.ligand.name,
            "n_binding_sites": len(self.binding_sites),
            "binding_sites": [
                {
                    "index": i,
                    "type": site.site_type.value,
                    "atom_indices": site.atom_indices,
                    "score": site.score,
                    "position": site.position.tolist(),
                    "denticity": site.functional_group_info.get('denticity', 1)
                }
                for i, site in enumerate(self.binding_sites)
            ]
        }
        self.file_handler.save_metadata(metadata, "binding_sites_metadata.json")
        
        return self.binding_sites
    
    def generate_complex_poses(self) -> List[Geometry]:
        """
        Generate metal-ligand complex poses.
        
        This method:
        1. Uses detected binding sites
        2. Generates poses based on coordination chemistry
        3. Creates multiple poses for each ligand conformer
        4. Saves them to 01_initial_structures/complexes/
        
        Returns:
            List of complex geometries
        """
        logger.info(f"Generating metal-ligand complex poses...")
        
        # Make sure we have binding sites
        if not self.binding_sites:
            self.binding_sites = self.detect_binding_sites()
            if not self.binding_sites:
                logger.error("No binding sites found. Cannot generate poses.")
                return []
        
        # Make sure we have ligand conformers
        if not self.results['ligand_conformers']:
            logger.error("No ligand conformers available. Run generate_ligand_conformers first.")
            return []
        
        all_poses = []
        pose_counter = 0
        
        # Generate poses for each ligand conformer
        n_conformers_to_use = min(
            len(self.results['ligand_conformers']), 
            self.config.n_conformers if hasattr(self.config, 'n_conformers_for_posing') else 5
        )
        
        for conf_idx in range(n_conformers_to_use):
            conformer = self.results['ligand_conformers'][conf_idx]
            logger.info(f"Generating poses for conformer {conf_idx + 1}/{n_conformers_to_use}")
            
            # Generate poses for this conformer
            if isinstance(self.pose_generator, EnhancedPoseGenerator):
                # Use enhanced generator with force field optimization
                poses = self.pose_generator.generate_poses(
                    ligand_geometry=conformer,
                    metal=self.metal,
                    binding_sites=self.binding_sites,
                    max_poses_per_conformer=self.config.max_poses_per_conformer,
                    rmsd_threshold=self.config.rmsd_threshold,
                    optimize_with_ff=self.config.optimize_poses_with_ff,
                    ff_method=self.config.ff_method
                )
            else:
                # Use basic generator (backward compatibility)
                poses = self.pose_generator.generate_poses(
                    ligand_geometry=conformer,
                    metal=self.metal,
                    binding_sites=self.binding_sites,
                    max_poses=self.config.max_poses_per_conformer,
                    rmsd_threshold=self.config.rmsd_threshold
                )
            
            # Save poses
            for pose in poses:
                # Update pose title
                pose.title = f"{self.ligand.name}_{self.metal.symbol}_conf{conf_idx}_pose{pose_counter}"
                
                # Determine geometry type from pose if available
                geom_type = None
                if hasattr(pose, 'properties') and 'geometry' in pose.properties:
                    geom_type = pose.properties['geometry']
                
                # Generate filename and save
                filename = self.file_handler.get_complex_filename(
                    self.ligand.name, 
                    self.metal.symbol, 
                    pose_counter,
                    geom_type
                )
                output_path = self.file_handler.dirs["initial_complexes"] / filename
                pose.save_xyz(output_path)
                
                all_poses.append(pose)
                pose_counter += 1
            
            logger.info(f"  Generated {len(poses)} poses for conformer {conf_idx + 1}")
        
        logger.info(f"Generated total of {len(all_poses)} complex poses")
        
        # Store in results
        self.results['complex_poses'] = all_poses
        
        # Save metadata
        metadata = {
            "ligand_name": self.ligand.name,
            "metal_symbol": self.metal.symbol,
            "metal_charge": self.metal.charge,
            "ligand_charge": self.ligand.charge,
            "total_charge": self.metal.charge + self.ligand.charge,
            "n_conformers_used": n_conformers_to_use,
            "n_binding_sites": len(self.binding_sites),
            "n_poses_generated": len(all_poses),
            "pose_files": [
                str(self.file_handler.dirs["initial_complexes"] / self.file_handler.get_complex_filename(
                    self.ligand.name, self.metal.symbol, i
                ))
                for i in range(len(all_poses))
            ],
            "coordination_geometries": list(set(
                pose.properties.get('geometry', 'unknown') 
                for pose in all_poses 
                if hasattr(pose, 'properties')
            ))
        }
        self.file_handler.save_metadata(metadata, "complex_generation_metadata.json")
        
        return all_poses
    
    def generate_all_structures(self) -> Dict[str, List[Geometry]]:
        """
        Generate all initial structures (ligand, metal, and complex).
        
        This is a convenience method that calls:
        1. generate_ligand_conformers()
        2. generate_metal_geometries()
        3. detect_binding_sites()
        4. generate_complex_poses()
        
        Returns:
            Dictionary with 'ligands', 'metals', and 'complexes' keys
        """
        logger.info("Generating all initial structures")
        
        # Step 1: Generate ligand conformers
        logger.info("Step 1: Generating ligand conformers")
        ligand_conformers = self.generate_ligand_conformers()
        
        # Step 2: Generate metal geometries
        logger.info("Step 2: Generating metal geometries")
        metal_geometries = self.generate_metal_geometries()
        
        # Step 3: Detect binding sites
        logger.info("Step 3: Detecting binding sites")
        binding_sites = self.detect_binding_sites()
        
        # Step 4: Generate complex poses
        logger.info("Step 4: Generating complex poses")
        complex_poses = self.generate_complex_poses()
        
        # Summary
        logger.info(f"Structure generation complete: {len(ligand_conformers)} conformers, {len(metal_geometries)} metals, {len(complex_poses)} poses")
        
        # Save overall generation summary
        summary = {
            "experiment_name": self.config.experiment_name,
            "timestamp": str(Path.cwd()),
            "ligand": {
                "name": self.ligand.name,
                "smiles": self.ligand.smiles,
                "charge": self.ligand.charge,
                "n_conformers": len(ligand_conformers)
            },
            "metal": {
                "symbol": self.metal.symbol,
                "charge": self.metal.charge,
                "n_geometries": len(metal_geometries)
            },
            "binding_sites": {
                "n_sites": len(binding_sites),
                "types": [site.site_type.value for site in binding_sites]
            },
            "complexes": {
                "n_poses": len(complex_poses),
                "total_charge": self.metal.charge + self.ligand.charge
            },
            "directories": {
                "experiment": str(self.work_dir),
                "initial_structures": str(self.file_handler.dirs["initial"]),
                "metals": str(self.file_handler.dirs["initial_metals"]),
                "ligands": str(self.file_handler.dirs["initial_ligands"]),
                "complexes": str(self.file_handler.dirs["initial_complexes"])
            }
        }
        self.file_handler.save_metadata(summary, "structure_generation_summary.json")
        
        return {
            "ligands": ligand_conformers,
            "metals": metal_geometries,
            "complexes": complex_poses
        }
    
    def optimize_structures_xtb(
        self, 
        input_folder: Optional[str] = None,
        create_folders: bool = True,
        custom_charges: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[OptimizationResult]]:
        """
        Optimize all structures using xTB.
        
        This method:
        1. Reads structures from initial structures folder
        2. Runs XTB optimization for each structure
        3. Saves results to 02_optimized_structures/
        4. Stores optimization data in JSON files
        
        Args:
            input_folder: Input folder name (default: "01_initial_structures")
            create_folders: If True, create folder for each optimization
                          If False, run in temp dir and save only results
            custom_charges: Custom charges for structure types
                          Default: {"metals": metal.charge, "ligands": ligand.charge, 
                                   "complexes": metal.charge + ligand.charge}
            
        Returns:
            Dictionary with optimization results for each structure type
        """
        logger.info("Starting XTB Optimization Workflow")
        
        # Set up charge map
        if custom_charges is None:
            charge_map = {
                "metals": self.metal.charge,
                "ligands": self.ligand.charge,
                "complexes": self.metal.charge + self.ligand.charge
            }
        else:
            charge_map = custom_charges
        
        logger.info(f"Charge settings:")
        for struct_type, charge in charge_map.items():
            logger.info(f"  {struct_type}: {charge:+d}")
        
        # Create workflow manager
        workflow_manager = XTBWorkflowManager(
            base_dir=self.work_dir,
            config=self.config.xtb_config,
            n_workers=self.config.n_workers or 4,
            create_folders=create_folders
        )
        
        # Run optimizations
        try:
            results = workflow_manager.optimize_all_structures(
                input_folder=input_folder,
                charge_map=charge_map
            )
            
            # Store results
            self.results['optimization_results'] = results
            
            # Update stored optimized structures
            if 'metals' in results:
                self.results['optimized_metals'] = [r for r in results['metals'] if r.success]
            if 'ligands' in results:
                self.results['optimized_ligands'] = [r for r in results['ligands'] if r.success]
            if 'complexes' in results:
                self.results['optimized_complexes'] = [r for r in results['complexes'] if r.success]
            
            # Print summary
            logger.info("Optimization Summary:")
            
            total_structures = 0
            total_successful = 0
            
            for struct_type, type_results in results.items():
                n_total = len(type_results)
                n_success = len([r for r in type_results if r.success])
                total_structures += n_total
                total_successful += n_success
                
                logger.info(f"{struct_type.capitalize()}: {n_success}/{n_total} successful")
                
                # Show failed optimizations
                failed = [r for r in type_results if not r.success]
                if failed:
                    logger.warning(f"  Failed {struct_type}:")
                    for r in failed[:5]:  # Show first 5 failures
                        logger.warning(f"    - {r.initial_geometry.title}: {r.error_message}")
                    if len(failed) > 5:
                        logger.warning(f"    ... and {len(failed) - 5} more")
            
            logger.info(f"Total: {total_successful}/{total_structures} successful")
            logger.info(f"Output directory: {self._get_relative_path(self.work_dir / '02_optimized_structures')}")
            
            # Save optimization metadata
            metadata = {
                "timestamp": str(datetime.now()),
                "xtb_config": {
                    "method": self.config.xtb_config.method,
                    "solvent": self.config.xtb_config.solvent,
                    "convergence": self.config.xtb_config.convergence
                },
                "charges": charge_map,
                "create_folders": create_folders,
                "summary": {
                    struct_type: {
                        "total": len(type_results),
                        "successful": len([r for r in type_results if r.success]),
                        "failed": len([r for r in type_results if not r.success])
                    }
                    for struct_type, type_results in results.items()
                }
            }
            self.file_handler.save_metadata(metadata, "xtb_optimization_metadata.json")
            
            return results
            
        except Exception as e:
            logger.error(f"XTB optimization workflow failed: {str(e)}")
            raise
    
    def optimize_all_structures(self, create_folders: bool = True) -> Dict[str, List[OptimizationResult]]:
        """
        Convenience method to optimize all generated structures.
        
        This runs XTB optimization on all structures in 01_initial_structures.
        
        Args:
            create_folders: If True, create folder for each optimization
            
        Returns:
            Dictionary with optimization results
        """
        # Make sure structures have been generated
        if not (self.work_dir / "01_initial_structures").exists():
            logger.error("No initial structures found. Run generate_all_structures() first.")
            return {}
        
        return self.optimize_structures_xtb(
            input_folder="01_initial_structures",
            create_folders=create_folders
        )
    
    def calculate_binding_energies(self) -> List[BindingEnergyResult]:
        """
        Calculate binding energies for optimized complexes.
        
        Binding Energy = E(complex) - E(metal) - E(ligand)
        
        Returns:
            List of binding energy results
        """
        logger.info("Calculating binding energies for optimized complexes...")
        
        # Check if optimization results exist
        if "optimization_results" not in self.results:
            logger.error("No optimization results found. Run optimize_structures_xtb first.")
            return []
        
        opt_results = self.results["optimization_results"]
        
        # Get successful optimizations
        successful_complexes = [r for r in opt_results.get("complexes", []) if r.success and r.energy is not None]
        successful_metals = [r for r in opt_results.get("metals", []) if r.success and r.energy is not None]
        successful_ligands = [r for r in opt_results.get("ligands", []) if r.success and r.energy is not None]
        
        if not successful_complexes:
            logger.error("No successful complex optimizations found.")
            return []
        
        if not successful_metals:
            logger.error("No successful metal optimizations found.")
            return []
        
        if not successful_ligands:
            logger.error("No successful ligand optimizations found.")
            return []
        
        logger.debug(f"Found {len(successful_complexes)} successful complexes, {len(successful_metals)} metals, {len(successful_ligands)} ligands")
        
        # Use lowest energy metal and ligand as reference
        reference_metal = min(successful_metals, key=lambda r: r.energy)
        reference_ligand = min(successful_ligands, key=lambda r: r.energy)
        
        logger.info(f"Using reference metal energy: {reference_metal.energy:.6f} Hartree")
        logger.info(f"Using reference ligand energy: {reference_ligand.energy:.6f} Hartree")
        
        binding_energy_results = []
        
        for complex_result in successful_complexes:
            try:
                # Calculate binding energy in Hartree
                binding_energy_hartree = (
                    complex_result.energy - reference_metal.energy - reference_ligand.energy
                )
                
                # Convert to kcal/mol (1 Hartree = 627.5094740631 kcal/mol)
                binding_energy_kcal = binding_energy_hartree * 627.5094740631
                
                # Create BindingEnergyResult
                binding_result = BindingEnergyResult(
                    binding_energy=binding_energy_kcal,
                    binding_energy_hartree=binding_energy_hartree,
                    complex_energy=complex_result.energy,
                    metal_energy=reference_metal.energy,
                    ligand_energy=reference_ligand.energy,
                    complex_geometry=complex_result.optimized_geometry,
                    metal_geometry=reference_metal.optimized_geometry,
                    ligand_geometry=reference_ligand.optimized_geometry
                )
                
                binding_energy_results.append(binding_result)
                
                logger.debug(f"Complex {complex_result.initial_geometry.title}: "
                           f"ΔE = {binding_energy_kcal:.2f} kcal/mol")
                
            except Exception as e:
                logger.error(f"Failed to calculate binding energy for {complex_result.initial_geometry.title}: {e}")
                continue
        
        # Sort by binding energy (most negative = most favorable)
        binding_energy_results.sort(key=lambda r: r.binding_energy)
        
        logger.info(f"Calculated {len(binding_energy_results)} binding energies")
        if binding_energy_results:
            best_energy = binding_energy_results[0].binding_energy
            worst_energy = binding_energy_results[-1].binding_energy
            logger.info(f"Energy range: {best_energy:.2f} to {worst_energy:.2f} kcal/mol")
            logger.debug(f"Most favorable complex: {binding_energy_results[0].complex_geometry.title} (ΔE = {best_energy:.2f} kcal/mol)")
        
        # Store results
        self.results['binding_energies'] = binding_energy_results
        
        return binding_energy_results
    
    def rank_structures(self, criterion: str = "binding_energy") -> List[Tuple[int, float]]:
        """
        Rank structures by specified criterion.
        
        Args:
            criterion: Ranking criterion ('binding_energy', 'complex_energy', etc.)
            
        Returns:
            List of (index, value) tuples sorted by criterion
        """
        logger.info(f"Ranking structures by {criterion}...")
        
        rankings = []
        
        if criterion == "binding_energy":
            # Rank by binding energy (requires binding energy calculations)
            if "binding_energies" not in self.results:
                logger.error("No binding energies calculated. Run calculate_binding_energies first.")
                return []
            
            binding_energies = self.results["binding_energies"]
            rankings = [(i, result.binding_energy) for i, result in enumerate(binding_energies)]
            # Sort by binding energy (most negative = most favorable)
            rankings.sort(key=lambda x: x[1])
            
        elif criterion == "complex_energy":
            # Rank by absolute complex energy
            if "optimization_results" not in self.results or "complexes" not in self.results["optimization_results"]:
                logger.error("No complex optimization results found.")
                return []
            
            complex_results = self.results["optimization_results"]["complexes"]
            successful_complexes = [(i, r.energy) for i, r in enumerate(complex_results) 
                                  if r.success and r.energy is not None]
            # Sort by energy (lowest = most stable)
            rankings = sorted(successful_complexes, key=lambda x: x[1])
            
        elif criterion == "ligand_energy":
            # Rank by ligand energy
            if "optimization_results" not in self.results or "ligands" not in self.results["optimization_results"]:
                logger.error("No ligand optimization results found.")
                return []
            
            ligand_results = self.results["optimization_results"]["ligands"]
            successful_ligands = [(i, r.energy) for i, r in enumerate(ligand_results) 
                                if r.success and r.energy is not None]
            rankings = sorted(successful_ligands, key=lambda x: x[1])
            
        elif criterion == "metal_energy":
            # Rank by metal energy
            if "optimization_results" not in self.results or "metals" not in self.results["optimization_results"]:
                logger.error("No metal optimization results found.")
                return []
            
            metal_results = self.results["optimization_results"]["metals"]
            successful_metals = [(i, r.energy) for i, r in enumerate(metal_results) 
                               if r.success and r.energy is not None]
            rankings = sorted(successful_metals, key=lambda x: x[1])
            
        else:
            logger.error(f"Unknown ranking criterion: {criterion}")
            logger.info("Available criteria: 'binding_energy', 'complex_energy', 'ligand_energy', 'metal_energy'")
            return []
        
        if rankings:
            logger.info(f"Ranked {len(rankings)} structures by {criterion}")
            logger.info(f"Best value: {rankings[0][1]:.6f}")
            logger.info(f"Worst value: {rankings[-1][1]:.6f}")
            
            # Show top 5
            logger.debug("Top 5 structures:")
            for i, (idx, value) in enumerate(rankings[:5]):
                logger.debug(f"  {i+1}. Index {idx}: {value:.6f}")
        
        # Store rankings in results
        if "rankings" not in self.results:
            self.results["rankings"] = {}
        self.results["rankings"][criterion] = rankings
        
        return rankings
    
    def prepare_orca_inputs(
        self, 
        n_best: Optional[int] = None,
        multiplicities: Optional[List[int]] = None,
        structure_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare ORCA input files for best structures.
        
        This method:
        1. Selects the n lowest energy structures from XTB optimization
        2. Creates ORCA input files for different multiplicities
        3. Organizes files by structure type and multiplicity
        4. Generates submission scripts
        
        Args:
            n_best: Number of best structures to prepare (default: config.keep_top_n)
            multiplicities: List of multiplicities to consider (auto-determined if None)
            structure_types: Types to process (default: ["metals", "ligands", "complexes"])
            
        Returns:
            Dictionary with ORCA generation results
        """
        n_best = n_best or self.config.keep_top_n
        
        logger.info(f"Preparing ORCA inputs for top {n_best} structures")
        
        # Check if optimization results exist
        if "optimization_results" not in self.results:
            logger.error("No optimization results found. Run optimize_structures_xtb first.")
            return {}
        
        # Create ORCA generator
        orca_generator = ORCAGenerator(
            base_dir=self.work_dir,
            config=self.config.orca_config
        )
        
        # Filter structure types if specified
        opt_results = self.results["optimization_results"]
        if structure_types:
            opt_results = {k: v for k, v in opt_results.items() if k in structure_types}
        
        # Generate ORCA inputs
        try:
            results = orca_generator.generate_from_optimization_results(
                optimization_results=opt_results,
                metal=self.metal,
                ligand=self.ligand,
                n_lowest=n_best,
                multiplicities=multiplicities
            )
            
            # Store results
            self.results['orca_inputs'] = results
            
            # Print summary
            logger.info("ORCA Input Generation Summary:")
            logger.info("-" * 40)
            
            total_files = 0
            for struct_type in ["metal", "ligand", "complex"]:
                key = f"{struct_type}_inputs"
                if key in results:
                    n_files = len(results[key])
                    total_files += n_files
                    logger.info(f"{struct_type.capitalize()}s: {n_files} input files")
            
            logger.info(f"Total: {total_files} ORCA input files")
            logger.info(f"Output directory: {self._get_relative_path(self.work_dir / '03_orca_inputs')}")
            
            # Show multiplicity information
            if results.get("summary", {}).get("multiplicities"):
                mults = results["summary"]["multiplicities"]
                logger.info(f"Multiplicities considered: {mults}")
            
            logger.info("ORCA input preparation complete!")
            
            return results
            
        except Exception as e:
            logger.error(f"ORCA input generation failed: {str(e)}")
            raise
    
    def prepare_orca_for_complexes_only(
        self,
        n_best: Optional[int] = None,
        multiplicities: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to prepare ORCA inputs only for complexes.
        
        Args:
            n_best: Number of best structures (default: config.keep_top_n)
            multiplicities: List of multiplicities to consider
            
        Returns:
            Dictionary with ORCA generation results
        """
        return self.prepare_orca_inputs(
            n_best=n_best,
            multiplicities=multiplicities,
            structure_types=["complexes"]
        )
    
    def run_workflow(self) -> Dict[str, Any]:
        """
        Run the complete analysis workflow.
        
        Steps:
        1. Generate ligand conformers
        2. Generate metal geometries
        3. Detect binding sites
        4. Generate complex poses
        5. Optimize all structures with xTB
        6. Calculate binding energies
        7. Rank structures
        8. Prepare ORCA inputs for best structures
        
        Returns:
            Dictionary with all results
        """
        logger.info("="*60)
        logger.info(f"Starting {self.metal.symbol}-{self.ligand.name} binding analysis")
        logger.info("="*60)
        
        # Step 1: Generate ligand conformers
        if self.config.n_conformers > 0:
            ligand_conformers = self.generate_ligand_conformers()
            self.results['ligand_conformers'] = ligand_conformers
            logger.info(f"Generated {len(ligand_conformers)} ligand conformers")
        
        # Step 2: Generate metal geometries
        if self.config.optimize_metal:
            metal_geometries = self.generate_metal_geometries()
            self.results['metal_geometries'] = metal_geometries
            logger.info(f"Generated {len(metal_geometries)} metal geometries")
        
        # Step 3: Detect binding sites
        self.binding_sites = self.detect_binding_sites()
        logger.info(f"Found {len(self.binding_sites)} binding sites")
        
        # Step 4: Generate complex poses
        complex_poses = self.generate_complex_poses()
        self.results['complex_poses'] = complex_poses
        logger.info(f"Generated {len(complex_poses)} complex poses")
        
        # Step 5: XTB optimizations
        if self.config.optimize_ligand and ligand_conformers:
            opt_ligands = self.optimize_structures_xtb(ligand_conformers, "ligand")
            self.results['optimized_ligands'] = opt_ligands
            
        if self.config.optimize_metal and metal_geometries:
            opt_metals = self.optimize_structures_xtb(metal_geometries, "metal")
            self.results['optimized_metals'] = opt_metals
            
        if self.config.optimize_complex and complex_poses:
            opt_complexes = self.optimize_structures_xtb(complex_poses, "complex")
            self.results['optimized_complexes'] = opt_complexes
        
        # Step 6: Calculate binding energies
        binding_energies = self.calculate_binding_energies()
        self.results['binding_energies'] = binding_energies
        
        # Step 7: Rank structures
        rankings = self.rank_structures("binding_energy")
        self.results['rankings'] = rankings
        
        # Step 8: Prepare ORCA inputs
        if self.config.prepare_orca:
            orca_results = self.prepare_orca_inputs()
            self.results['orca_inputs'] = orca_results
        
        logger.info("="*60)
        logger.info("Workflow completed successfully!")
        logger.info("="*60)
        
        return self.results
    
    def save_results(self) -> Dict[str, Path]:
        """
        Save all results to organized directory structure.
        
        This method:
        1. Creates a comprehensive results archive (JSON)
        2. Exports CSV summaries for each structure type
        3. Copies best structures to easily accessible locations
        4. Creates directory index files
        
        Returns:
            Dictionary with paths to saved files
        """
        logger.info("Saving analysis results...")
        
        # Initialize report generator
        report_gen = ReportGenerator(self.work_dir)
        
        saved_files = {}
        
        # 1. Save complete results archive
        try:
            archive_path = report_gen.save_results_archive(self.results)
            saved_files["archive"] = archive_path
            logger.info(f"Results archive saved: {self._get_relative_path(archive_path)}")
        except Exception as e:
            logger.error(f"Failed to save results archive: {e}")
        
        # 2. Save CSV summaries
        try:
            self._save_csv_summaries()
            saved_files["csv_summaries"] = self.work_dir / "04_reports"
            logger.info("CSV summaries saved")
        except Exception as e:
            logger.error(f"Failed to save CSV summaries: {e}")
        
        # 3. Copy best structures
        try:
            best_structures_dir = self._copy_best_structures()
            if best_structures_dir:
                saved_files["best_structures"] = best_structures_dir
                logger.info(f"Best structures copied to: {self._get_relative_path(best_structures_dir)}")
        except Exception as e:
            logger.error(f"Failed to copy best structures: {e}")
        
        # 4. Create directory index
        try:
            index_path = self._create_directory_index()
            saved_files["directory_index"] = index_path
            logger.info(f"Directory index created: {self._get_relative_path(index_path)}")
        except Exception as e:
            logger.error(f"Failed to create directory index: {e}")
        
        # 5. Save workflow summary
        try:
            summary_path = self._save_workflow_summary()
            saved_files["workflow_summary"] = summary_path
            logger.info(f"Workflow summary saved: {self._get_relative_path(summary_path)}")
        except Exception as e:
            logger.error(f"Failed to save workflow summary: {e}")
        
        logger.info("Results saved successfully! See README.md for directory structure and details.")
        
        return saved_files
    
    def _save_csv_summaries(self) -> None:
        """Save CSV summaries for optimization results."""
        if "optimization_results" not in self.results:
            return
        
        reports_dir = self.work_dir / "04_reports"
        reports_dir.mkdir(exist_ok=True)
        
        opt_results = self.results["optimization_results"]
        
        for struct_type in ["metals", "ligands", "complexes"]:
            if struct_type not in opt_results:
                continue
            
            # Prepare data for CSV
            data = []
            for i, result in enumerate(opt_results[struct_type]):
                row = {
                    "index": i,
                    "structure_name": result.initial_geometry.title,
                    "success": result.success,
                    "initial_energy": getattr(result.initial_geometry, 'energy', None),
                    "final_energy": result.energy if result.success else None,
                    "energy_change": (result.energy - getattr(result.initial_geometry, 'energy', result.energy)) 
                                   if (result.success and result.energy is not None and 
                                       hasattr(result.initial_geometry, 'energy') and 
                                       getattr(result.initial_geometry, 'energy', None) is not None) else None,
                    "homo_lumo_gap": result.properties.get("homo_lumo_gap") if result.properties else None,
                    "dipole_moment": result.properties.get("dipole_moment") if result.properties else None,
                    "error_message": result.error_message if not result.success else None
                }
                
                # Add relative energy in kcal/mol
                if result.success and result.energy:
                    successful_energies = [r.energy for r in opt_results[struct_type] 
                                         if r.success and r.energy]
                    if successful_energies:
                        min_energy = min(successful_energies)
                        row["relative_energy_kcal"] = (result.energy - min_energy) * 627.5094740631
                
                data.append(row)
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                
                # Sort successful results by energy
                successful_df = df[df['success'] == True]
                failed_df = df[df['success'] == False]
                
                if not successful_df.empty:
                    successful_df = successful_df.sort_values('final_energy')
                
                # Combine successful and failed
                final_df = pd.concat([successful_df, failed_df], ignore_index=True)
                
                # Save CSV
                csv_path = reports_dir / f"{struct_type}_results.csv"
                final_df.to_csv(csv_path, index=False)
                
                # Save top 10 if enough successful results
                if len(successful_df) >= 10:
                    top10_path = reports_dir / f"{struct_type}_top10.csv"
                    successful_df.head(10).to_csv(top10_path, index=False)
    
    def _copy_best_structures(self) -> Optional[Path]:
        """Copy best structures to easily accessible location."""
        if "optimization_results" not in self.results:
            return None
        
        best_dir = self.work_dir / "05_best_structures"
        best_dir.mkdir(exist_ok=True)
        
        opt_results = self.results["optimization_results"]
        copied_any = False
        
        for struct_type in ["metals", "ligands", "complexes"]:
            if struct_type not in opt_results:
                continue
            
            # Get successful results
            successful = [r for r in opt_results[struct_type] 
                         if r.success and r.optimized_geometry]
            
            if not successful:
                continue
            
            # Sort by energy and take best 5
            successful.sort(key=lambda r: r.energy)
            best_structures = successful[:5]
            
            type_dir = best_dir / struct_type
            type_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(best_structures):
                # Save optimized geometry
                filename = f"best_{i+1:02d}_{result.optimized_geometry.title}.xyz"
                output_path = type_dir / filename
                result.optimized_geometry.save_xyz(output_path)
                copied_any = True
        
        return best_dir if copied_any else None
    
    def _create_directory_index(self) -> Path:
        """Create directory index file."""
        index_content = [
            "# Metal-Ligand Binding Analysis Results",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# System: {self.metal.symbol}({self.ligand.name})",
            "",
            "## Directory Structure",
            "",
            "```",
            f"{self.work_dir.name}/",
            "├── 01_initial_structures/",
            "│   ├── metals/              # Initial metal geometries",
            "│   ├── ligands/             # Ligand conformers",
            "│   └── complexes/           # Metal-ligand poses",
            "├── 02_optimized_structures/",
            "│   ├── metals/              # XTB optimized metals",
            "│   ├── ligands/             # XTB optimized ligands",
            "│   └── complexes/           # XTB optimized complexes",
            "├── 03_orca_inputs/",
            "│   ├── metals/mult_*/       # ORCA inputs by multiplicity",
            "│   ├── ligands/mult_*/",
            "│   └── complexes/mult_*/",
            "├── 04_reports/",
            "│   ├── analysis_report.html # Comprehensive HTML report",
            "│   ├── *_results.csv       # CSV summaries",
            "│   └── *_top10.csv         # Best 10 structures",
            "├── 05_best_structures/",
            "│   ├── metals/              # 5 best metal structures",
            "│   ├── ligands/             # 5 best ligand conformers",
            "│   └── complexes/           # 5 best complexes",
            "└── 06_metadata_files/",
            "    ├── results_archive.json   # Complete results archive",
            "    ├── workflow_summary.json  # Workflow summary",
            "    └── *_metadata.json        # Various metadata files",
            "```",
            "",
            "## Key Files",
            "",
            "- `06_metadata_files/results_archive.json`: Complete results in JSON format",
            "- `06_metadata_files/workflow_summary.json`: High-level workflow summary",
            "- `06_metadata_files/*_metadata.json`: Various metadata files",
            "- `04_reports/analysis_report.html`: Comprehensive analysis report",
            "- `05_best_structures/`: Easy access to best structures",
            "",
            "## Analysis Summary",
            ""
        ]
        
        # Add summary statistics
        if "optimization_results" in self.results:
            opt_results = self.results["optimization_results"]
            for struct_type in ["metals", "ligands", "complexes"]:
                if struct_type in opt_results:
                    total = len(opt_results[struct_type])
                    successful = len([r for r in opt_results[struct_type] if r.success])
                    index_content.append(f"- {struct_type.capitalize()}: {successful}/{total} successful optimizations")
        
        if "orca_inputs" in self.results:
            orca = self.results["orca_inputs"]
            total_orca = sum(len(orca.get(f"{t}_inputs", [])) for t in ["metal", "ligand", "complex"])
            index_content.append(f"- ORCA inputs generated: {total_orca} files")
        
        index_path = self.work_dir / "README.md"
        index_path.write_text('\n'.join(index_content))
        
        return index_path
    
    def _save_workflow_summary(self) -> Path:
        """Save high-level workflow summary."""
        summary = {
            "experiment_info": {
                "name": self.config.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "metal": f"{self.metal.symbol}{self.metal.charge:+d}",
                    "ligand": f"{self.ligand.name}{self.ligand.charge:+d}",
                    "complex_charge": self.metal.charge + self.ligand.charge
                }
            },
            "workflow_steps": {
                "structure_generation": "ligand_conformers" in self.results,
                "binding_site_detection": len(self.binding_sites) > 0,
                "pose_generation": "complex_poses" in self.results,
                "xtb_optimization": "optimization_results" in self.results,
                "orca_preparation": "orca_inputs" in self.results
            },
            "results_summary": {},
            "file_locations": {
                "work_directory": str(self.work_dir),
                "reports": str(self.work_dir / "04_reports"),
                "best_structures": str(self.work_dir / "05_best_structures"),
                "orca_inputs": str(self.work_dir / "03_orca_inputs")
            }
        }
        
        # Add results counts
        if "optimization_results" in self.results:
            opt_results = self.results["optimization_results"]
            for struct_type in ["metals", "ligands", "complexes"]:
                if struct_type in opt_results:
                    successful = [r for r in opt_results[struct_type] if r.success]
                    summary["results_summary"][struct_type] = {
                        "total_generated": len(opt_results[struct_type]),
                        "successfully_optimized": len(successful),
                        "lowest_energy": min(r.energy for r in successful) if successful else None
                    }
        
        summary_path = self.work_dir / "06_metadata_files" / "workflow_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary_path
    
    def generate_report(self) -> Path:
        """
        Generate comprehensive analysis report.
        
        This method creates an HTML report with:
        1. System metadata and configuration
        2. Structure generation summary
        3. Optimization results analysis
        4. Energy analysis and rankings
        5. Best structures identification
        6. ORCA preparation summary
        
        Returns:
            Path to generated HTML report
        """
        logger.info("Generating comprehensive analysis report...")
        
        # Initialize report generator
        report_gen = ReportGenerator(self.work_dir)
        
        # Prepare configuration data for the report
        config_data = {
            "experiment_name": self.config.experiment_name,
            "max_poses": self.config.max_poses_per_conformer,
            "n_conformers": self.config.n_conformers,
            "xtb_config": {
                "method": self.config.xtb_config.method,
                "solvent": self.config.xtb_config.solvent,
                "convergence": self.config.xtb_config.convergence
            },
            "orca_config": {
                "method": self.config.orca_config.method,
                "basis_set": self.config.orca_config.basis_set,
                "dispersion": self.config.orca_config.dispersion
            }
        }
        
        # Generate the comprehensive report
        try:
            report_path = report_gen.generate_full_report(
                metal=self.metal,
                ligand=self.ligand,
                results=self.results,
                config=config_data
            )
            
            logger.info(f"Comprehensive analysis report generated: {self._get_relative_path(report_path)}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise