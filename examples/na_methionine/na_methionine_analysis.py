#!/usr/bin/env python3
"""
Metal-Ligand Binding Analysis Example: Na(I)-Methionine Complex

This example demonstrates the analysis of sodium ion binding to methionine.
Methionine is an amino acid with a thioether sulfur that can coordinate to metals,
along with the carboxylate group.

Na+ is a hard metal ion that typically prefers oxygen coordination over sulfur.
This analysis will explore the preferred binding modes.
"""

import sys
from pathlib import Path

from ml_xtb_prescreening import (
    MetalLigandComplex,
    ComplexConfig,
    XTBConfig,
    ORCAConfig
)
from ml_xtb_prescreening.core.logger import logger


def analyze_na_methionine_complex():
    """
    Analyze Na(I)-methionine complex formation and binding.
    
    Methionine has multiple potential coordination sites:
    - Carboxylate oxygen atoms (likely preferred for Na+)
    - Amino nitrogen
    - Thioether sulfur (less likely for hard Na+)
    
    This analysis will identify the preferred coordination mode.
    """
    
    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    
    # Create configuration
    config = ComplexConfig(
        # Experiment settings
        experiment_name="na_methionine_binding_analysis",
        
        # Structure generation settings
        max_poses=200,          # Generate 30 different metal-ligand poses
        n_conformers=200,       # Generate 15 methionine conformers
        rmsd_threshold=0.5,    # RMSD threshold for removing duplicate structures
        
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
            method="PBE0",          # PBE0 functional (good for alkali metals)
            basis_set="def2-TZVP",  # Triple-zeta basis set
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
    
    # Methionine SMILES: 2-amino-4-(methylthio)butanoic acid
    # Using zwitterionic form which is common at physiological pH
    methionine_smiles = "CSCCC(N)C(=O)O"
    
    complex = MetalLigandComplex(
        ligand_name="methionine",
        ligand_smiles=methionine_smiles,
        metal_symbol="Na",
        metal_charge=1,      # Na+
        ligand_charge=0,     # Neutral methionine (zwitterionic)
        ligand_protonation_state="neutral",
        config=config
    )
    
    # =========================================================================
    # STEP 3: Generate Initial Structures
    # =========================================================================
    
    try:
        # Generate all structures at once
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
        print("Please install xTB to continue with optimization.")
        return
    
    try:
        # Run optimization with folder creation for detailed results
        print("\nStarting XTB optimization...")
        optimization_results = complex.optimize_all_structures(create_folders=True)
        
    except Exception as e:
        print(f"ERROR: XTB optimization failed: {e}")
        return
    
    # =========================================================================
    # STEP 5: Calculate Binding Energies and Rank Structures
    # =========================================================================
    
    try:
        # Calculate binding energies
        print("\nCalculating binding energies...")
        binding_energies = complex.calculate_binding_energies()
        
        if binding_energies:
            print(f"Calculated {len(binding_energies)} binding energies")
            
            # Show top 5 binding energies
            print("\nTop 5 binding energies (kcal/mol):")
            for i, be in enumerate(binding_energies[:5]):
                print(f"  {i+1}. {be.complex_geometry.title}: {be.binding_energy:.2f} kcal/mol")
        
        # Rank structures by binding energy
        rankings = complex.rank_structures("binding_energy")
        
    except Exception as e:
        print(f"ERROR: Binding energy calculation failed: {e}")
    
    # =========================================================================
    # STEP 6: Generate ORCA Input Files
    # =========================================================================
    
    try:
        # Generate ORCA inputs for the best structures
        print("\nGenerating ORCA input files...")
        orca_results = complex.prepare_orca_inputs(
            n_best=5,  # Take top 5 structures
            multiplicities=[1]  # Na-methionine complex should be singlet
        )
        
        print(f"ORCA inputs generated in: {complex.work_dir / '04_orca_inputs'}")
        
    except Exception as e:
        print(f"ERROR: ORCA input generation failed: {e}")
    
    # =========================================================================
    # STEP 7: Save Results and Generate Report
    # =========================================================================
    
    try:
        # Save all results
        print("\nSaving results...")
        saved_files = complex.save_results()
        
        # Generate comprehensive HTML report
        report_path = complex.generate_report()
        
        # Analysis complete
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"{'='*60}")
        print(f"Results saved in: {complex.work_dir}")
        print(f"HTML report: {report_path}")
        print(f"\nKey findings:")
        print(f"  - Generated {len(structures['complexes'])} initial complex poses")
        print(f"  - Successfully optimized {len([r for r in optimization_results.get('complexes', []) if r.success])} complexes")
        if binding_energies:
            print(f"  - Best binding energy: {binding_energies[0].binding_energy:.2f} kcal/mol")
            print(f"  - Binding site: Check the HTML report for coordination details")
        
    except Exception as e:
        print(f"ERROR: Results saving failed: {e}")


def main():
    """Main function with error handling."""
    print("="*60)
    print("Na(I)-Methionine Binding Analysis")
    print("="*60)
    
    # Example: Disable debug logging for cleaner output
    # Uncomment the following line to disable debug messages:
    # logger.disable_debug()
    
    # Or set via environment variable before running:
    # export ML_XTB_DEBUG=0
    
    print("\nThis analysis will:")
    print("1. Generate methionine conformers")
    print("2. Create Na+ ion structure")  
    print("3. Generate Na-methionine complex poses")
    print("4. Optimize all structures with xTB")
    print("5. Calculate binding energies")
    print("6. Prepare ORCA input files for DFT")
    print("="*60)
    
    try:
        analyze_na_methionine_complex()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()