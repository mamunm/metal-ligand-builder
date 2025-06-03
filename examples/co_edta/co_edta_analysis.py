#!/usr/bin/env python3
"""
Complete Metal-Ligand Binding Analysis Example: Co(II)-EDTA Complex

This example demonstrates the full workflow for analyzing metal-ligand binding:
1. Structure generation (ligand conformers, metal geometries, complex poses)
2. XTB optimization of all structures
3. Binding energy calculation
4. Structure ranking and analysis
5. Report generation

EDTA (Ethylenediaminetetraacetic acid) is a multidentate chelating ligand
that can coordinate through multiple sites with metal ions.
"""

import sys
from pathlib import Path

from ml_xtb_prescreening import (
    MetalLigandComplex,
    ComplexConfig,
    XTBConfig,
    ORCAConfig
)

import os
os.environ["ML_XTB_DEBUG"] = "0"


def analyze_co_edta_complex():
    """
    Analyze Co(II)-EDTA complex formation and binding.
    
    EDTA (Ethylenediaminetetraacetic acid) in its neutral form is used for
    initial structure generation. The ligand can coordinate through:
    - Carboxylic acid oxygen atoms
    - Amine nitrogen atoms
    
    This analysis will generate structures and optimize them to find stable
    coordination geometries around Co(II).
    """
    
    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    
    # Create configuration
    config = ComplexConfig(
        # Experiment settings
        experiment_name="co_edta_binding_analysis",
        
        # Structure generation settings
        max_poses_per_conformer=10,  # Generate 10 refined poses per conformer
        n_conformers=5,              # Generate 5 EDTA conformers (total 50 poses max)
        rmsd_threshold=0.5,          # RMSD threshold for removing duplicate structures
        optimize_poses_with_ff=True, # Use force field optimization for poses
        ff_method="auto",            # Auto-select RDKit or OpenBabel
        
        # XTB optimization settings
        xtb_config=XTBConfig(
            method="gfn2",          # GFN2-xTB method
            solvent=None,           # Gas phase optimization
            accuracy=1.0,           # Standard accuracy
            electronic_temperature=300.0,  # Room temperature
            max_iterations=250,     # Maximum optimization cycles
            convergence="normal"    # Normal convergence criteria
        ),
        
        # ORCA settings for future DFT calculations
        orca_config=ORCAConfig(
            method="B3LYP",         # B3LYP functional
            basis_set="def2-SVP",   # Double-zeta basis set
            dispersion="D3BJ",      # Grimme's D3 dispersion with BJ damping
        ),
        
        # Computational resources
        n_workers=4,  # Use 4 parallel processes
        
        # Analysis settings
        energy_window=10.0,  # Consider structures within 10 kcal/mol
        keep_top_n=5        # Keep top 5 structures for ORCA
    )
    
    # =========================================================================
    # STEP 2: Initialize Metal-Ligand Complex
    # =========================================================================
    
    # EDTA SMILES: Ethylenediaminetetraacetic acid (neutral form)
    # Using the neutral/protonated form for initial structure generation
    edta_smiles = "C(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O"
    
    complex = MetalLigandComplex(
        ligand_name="edta",
        ligand_smiles=edta_smiles,
        metal_symbol="Co",
        metal_charge=2,      # Co²⁺
        ligand_charge=0,     # EDTA neutral form for structure generation
        ligand_protonation_state="neutral",
        config=config
    )
    
    # =========================================================================
    # STEP 3: Generate Initial Structures
    # =========================================================================
    
    try:
        # Generate all structures at once
        # This creates:
        # - 5 EDTA conformers using RDKit/OpenBabel
        # - 1 Co²⁺ metal ion structure
        # - Up to 10 refined poses per conformer (50 total max)
        #   with force field optimization and conformational sampling
        structures = complex.generate_all_structures()
        
    except Exception as e:
        print(f"ERROR: Structure generation failed: {e}")
        return
    
    # =========================================================================
    # STEP 4: XTB Optimization
    # =========================================================================
    
    # Check if xTB is available
    import shutil
    if not shutil.which("xtb"):
        print("WARNING: xTB not found in PATH! Skipping optimization.")
        return
    
    try:
        # Run optimization with folder creation for detailed results
        optimization_results = complex.optimize_all_structures(create_folders=True)
        
    except Exception as e:
        print(f"ERROR: XTB optimization failed: {e}")
        return
    
    # =========================================================================
    # STEP 5: Calculate Binding Energies and Rank Structures
    # =========================================================================
    
    try:
        # Calculate binding energies
        binding_energies = complex.calculate_binding_energies()
        
        # Rank structures by binding energy
        rankings = complex.rank_structures("binding_energy")
        
    except Exception as e:
        print(f"ERROR: Binding energy calculation failed: {e}")
    
    # =========================================================================
    # STEP 6: Generate ORCA Input Files
    # =========================================================================
    
    try:
        # Generate ORCA inputs for the best structures
        orca_results = complex.prepare_orca_inputs(
            n_best=5,  # Take top 5 structures
            multiplicities=None  # Auto-determine multiplicities
        )
        
    except Exception as e:
        print(f"ERROR: ORCA input generation failed: {e}")
    
    # =========================================================================
    # STEP 7: Save Results and Generate Report
    # =========================================================================
    
    try:
        # Save all results
        saved_files = complex.save_results()
        
        # Generate comprehensive HTML report
        report_path = complex.generate_report()
        
        # Analysis complete
        print(f"Analysis complete! Results saved in: {', '.join(saved_files)}")
        try:
            rel_path = str(Path(report_path).relative_to(Path.cwd()))
        except:
            rel_path = report_path
        print(f"HTML report: {rel_path}")
        
    except Exception as e:
        print(f"ERROR: Results saving failed: {e}")


def main():
    """Main function with error handling."""
    try:
        analyze_co_edta_complex()
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()