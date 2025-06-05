#!/usr/bin/env python3
"""
Complete Metal-Ligand Binding Analysis Example: Na(I)-Methionine Complex

This example demonstrates the full workflow for analyzing metal-ligand binding:
1. Structure generation (ligand conformers, metal geometries, complex poses)
2. XTB optimization of all structures
3. Binding energy calculation
4. Structure ranking and analysis
5. Report generation

Methionine is an amino acid that can coordinate with metal ions through
amino group, carboxyl group, and potentially sulfur atom.
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


def analyze_na_methionine_complex():
    """
    Analyze Na(I)-Methionine complex formation and binding.
    
    Methionine can coordinate through:
    - Amino group nitrogen
    - Carboxyl group oxygen
    - Thioether sulfur (weaker interaction)
    
    This analysis will generate structures and optimize them to find stable
    coordination geometries around Na(I).
    """
    
    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    
    # Create configuration
    config = ComplexConfig(
        # Experiment settings
        experiment_name="na_methionine_binding_analysis",
        
        # Structure generation settings
        max_poses_per_conformer=50,  # Generate 50 poses per conformer
        n_conformers=20,             # Generate 20 conformers
        rmsd_threshold=0.5,          # RMSD threshold for removing duplicate structures
        optimize_poses_with_ff=True, # Use force field optimization for poses
        ff_method="auto",            # Auto-select RDKit or OpenBabel
        
        # XTB optimization settings
        xtb_config=XTBConfig(
            method="gfn2",          # GFN2-xTB method
            solvent=None,           # Vacuum
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
            solvent="vacuum",       # Vacuum calculations
            calculate_frequencies=True,  # Calculate vibrational frequencies
            multiplicity="auto",    # Auto-determine multiplicity
            nprocs=32,             # Use 32 processors for ORCA
            additional_keywords="RIJCOSX def2/J UHF"  # Additional ORCA keywords
        ),
        
        # Computational resources
        n_workers=4,  # Use 4 parallel processes
        
        # Analysis settings
        energy_window=25.0,  # Consider structures within 25 kcal/mol
        keep_top_n=25       # Keep top 25 structures for ORCA
    )
    
    # =========================================================================
    # STEP 2: Initialize Metal-Ligand Complex
    # =========================================================================
    
    # Methionine SMILES (L-methionine)
    methionine_smiles = "CSCCC(C(=O)O)N"
    
    complex = MetalLigandComplex(
        ligand_name="methionine",
        ligand_smiles=methionine_smiles,
        metal_symbol="Na",
        metal_charge=1,      # Na⁺
        ligand_charge=0,     # Methionine neutral form for structure generation
        ligand_protonation_state="neutral",
        config=config
    )
    
    # =========================================================================
    # STEP 3: Generate Initial Structures
    # =========================================================================
    
    try:
        structures = complex.generate_all_structures()
        
    except Exception as e:
        print(f"ERROR: Structure generation failed: {e}")
        return
    
    # =========================================================================
    # STEP 4: XTB Optimization with Automatic Validation
    # =========================================================================
    
    # Check if xTB is available
    import shutil
    if not shutil.which("xtb"):
        print("WARNING: xTB not found in PATH! Skipping optimization.")
        return
    
    try:
        optimization_results = complex.optimize_all_structures(create_folders=True)
        
        if 'complexes' in optimization_results:
            valid_count = sum(1 for r in optimization_results['complexes'] 
                            if r.success and r.validation_passed)
            invalid_count = sum(1 for r in optimization_results['complexes'] 
                              if r.success and not r.validation_passed)
            print(f"\nValidation Results: {valid_count} valid, {invalid_count} invalid complexes")
        
    except Exception as e:
        print(f"ERROR: XTB optimization failed: {e}")
        return
    
    # =========================================================================
    # STEP 5: Calculate Binding Energies and Rank Structures
    # =========================================================================
    
    try:
        binding_energies = complex.calculate_binding_energies()
        rankings = complex.rank_structures("binding_energy")
        
        print(f"\nBinding energy calculations completed for {len(binding_energies)} structures")
        
    except Exception as e:
        print(f"ERROR: Binding energy calculation failed: {e}")
    
    # =========================================================================
    # STEP 6: Generate ORCA Input Files
    # =========================================================================
    
    try:
        orca_results = complex.prepare_orca_inputs(
            n_best=25,  # Take top 25 structures
            multiplicities=None  # Auto-determine multiplicities
        )
        
        print(f"\nORCA inputs generated for top {len(orca_results.get('complex_inputs', []))} structures")
        
    except Exception as e:
        print(f"ERROR: ORCA input generation failed: {e}")
    
    # =========================================================================
    # STEP 7: Save Results and Generate Report
    # =========================================================================
    
    try:
        saved_files = complex.save_results()
        report_path = complex.generate_report()
        
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
        analyze_na_methionine_complex()
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()