"""
Tests for ORCA input generation functionality.

Tests cover:
- ORCA input file generation
- Multiplicity handling
- Directory organization
- Submission script creation
"""

import pytest
import json
from pathlib import Path
import numpy as np

from ml_xtb_prescreening.core.data_models import (
    Geometry, OptimizationResult, Metal, Ligand
)
from ml_xtb_prescreening.core.config import ORCAConfig
from ml_xtb_prescreening.io.orca_generator import ORCAGenerator
from ml_xtb_prescreening import MetalLigandComplex, ComplexConfig


class TestORCAGenerator:
    """Test ORCA generator functionality."""
    
    @pytest.fixture
    def orca_generator(self, temp_dir):
        """Create ORCA generator instance."""
        config = ORCAConfig(
            method="B3LYP",
            basis_set="def2-SVP",
            dispersion="D3BJ",
            n_cores=4
        )
        return ORCAGenerator(temp_dir, config)
    
    @pytest.fixture
    def sample_optimization_results(self):
        """Create sample optimization results for testing."""
        # Create metal result
        metal_geom = Geometry(
            atoms=["Co"],
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            title="Co_ion"
        )
        metal_opt = OptimizationResult(
            success=True,
            initial_geometry=metal_geom,
            optimized_geometry=metal_geom,
            energy=-1382.5,
            properties={"homo_lumo_gap": 5.0}
        )
        
        # Create ligand results
        ligand_geom = Geometry(
            atoms=["N", "H", "H", "H"],
            coordinates=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-0.5, 0.866, 0.0],
                [-0.5, -0.866, 0.0]
            ]),
            title="ammonia"
        )
        ligand_opt = OptimizationResult(
            success=True,
            initial_geometry=ligand_geom,
            optimized_geometry=ligand_geom,
            energy=-56.5,
            properties={"homo_lumo_gap": 8.0}
        )
        
        # Create complex result
        complex_geom = Geometry(
            atoms=["Co", "N", "H", "H", "H"],
            coordinates=np.array([
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.5, 0.5, 0.5],
                [2.5, 0.5, -0.5],
                [2.5, -1.0, 0.0]
            ]),
            title="Co_NH3_complex"
        )
        complex_opt = OptimizationResult(
            success=True,
            initial_geometry=complex_geom,
            optimized_geometry=complex_geom,
            energy=-1439.2,
            properties={"homo_lumo_gap": 4.5}
        )
        
        return {
            "metals": [metal_opt],
            "ligands": [ligand_opt],
            "complexes": [complex_opt]
        }
    
    def test_d_electron_counting(self, orca_generator):
        """Test d-electron counting for transition metals."""
        # Test various metals
        co2 = Metal(symbol="Co", charge=2)  # Co2+ is d7
        assert orca_generator._get_d_electron_count(co2) == 7
        
        ni2 = Metal(symbol="Ni", charge=2)  # Ni2+ is d8
        assert orca_generator._get_d_electron_count(ni2) == 8
        
        fe3 = Metal(symbol="Fe", charge=3)  # Fe3+ is d5
        assert orca_generator._get_d_electron_count(fe3) == 5
        
        zn2 = Metal(symbol="Zn", charge=2)  # Zn2+ is d10
        assert orca_generator._get_d_electron_count(zn2) == 10
        
        # Test main group metal
        ca2 = Metal(symbol="Ca", charge=2)
        assert orca_generator._get_d_electron_count(ca2) == 0
    
    def test_multiplicity_determination(self, orca_generator):
        """Test automatic multiplicity determination."""
        # d7 system (Co2+)
        co2 = Metal(symbol="Co", charge=2)
        ligand = Ligand(name="test", smiles="", charge=0)
        mults = orca_generator._determine_multiplicities(co2, ligand)
        assert 2 in mults  # Low spin doublet
        assert 4 in mults  # High spin quartet
        
        # d10 system (Zn2+)
        zn2 = Metal(symbol="Zn", charge=2)
        mults = orca_generator._determine_multiplicities(zn2, ligand)
        assert mults == [1]  # Always singlet
        
        # d5 system (Fe3+)
        fe3 = Metal(symbol="Fe", charge=3)
        mults = orca_generator._determine_multiplicities(fe3, ligand)
        assert 2 in mults  # Low spin
        assert 6 in mults  # High spin sextet
    
    def test_orca_input_generation(
        self, 
        orca_generator, 
        sample_optimization_results
    ):
        """Test ORCA input file generation."""
        metal = Metal(symbol="Co", charge=2)
        ligand = Ligand(name="ammonia", smiles="N", charge=0)
        
        results = orca_generator.generate_from_optimization_results(
            optimization_results=sample_optimization_results,
            metal=metal,
            ligand=ligand,
            n_lowest=1,
            multiplicities=[2, 4]
        )
        
        # Check results structure
        assert "metal_inputs" in results
        assert "ligand_inputs" in results
        assert "complex_inputs" in results
        
        # Check that files were created
        orca_dir = orca_generator.orca_dir
        assert orca_dir.exists()
        
        # Check complex inputs
        complex_dir = orca_dir / "complexes"
        assert complex_dir.exists()
        
        # Check multiplicity directories
        mult_2_dir = complex_dir / "mult_2"
        mult_4_dir = complex_dir / "mult_4"
        assert mult_2_dir.exists()
        assert mult_4_dir.exists()
        
        # Check input files
        inp_files_m2 = list(mult_2_dir.glob("*.inp"))
        inp_files_m4 = list(mult_4_dir.glob("*.inp"))
        assert len(inp_files_m2) > 0
        assert len(inp_files_m4) > 0
        
        # Check file content
        inp_file = inp_files_m2[0]
        content = inp_file.read_text()
        
        assert "! B3LYP def2-SVP" in content
        assert "! D3BJ" in content
        assert "! Opt" in content
        assert "* xyz 2 2" in content  # Charge 2, multiplicity 2
        assert "%maxcore" in content
        assert "%pal" in content
        assert "UHF" in content  # Should use UHF for multiplicity > 1
    
    def test_submission_script_generation(
        self,
        orca_generator,
        sample_optimization_results
    ):
        """Test submission script generation."""
        metal = Metal(symbol="Co", charge=2)
        ligand = Ligand(name="test", smiles="", charge=0)
        
        orca_generator.generate_from_optimization_results(
            optimization_results=sample_optimization_results,
            metal=metal,
            ligand=ligand,
            n_lowest=1,
            multiplicities=[2]
        )
        
        # Check submission scripts
        complex_dir = orca_generator.orca_dir / "complexes"
        
        # Master script
        master_script = complex_dir / "submit_all.sh"
        assert master_script.exists()
        assert master_script.stat().st_mode & 0o111  # Is executable
        
        content = master_script.read_text()
        assert "#!/bin/bash" in content
        assert "module load orca" in content
        assert "mult_2" in content
        
        # Single job script
        single_script = complex_dir / "run_single.sh"
        assert single_script.exists()
        assert single_script.stat().st_mode & 0o111
    
    def test_summary_generation(
        self,
        orca_generator,
        sample_optimization_results
    ):
        """Test summary JSON generation."""
        metal = Metal(symbol="Ni", charge=2)
        ligand = Ligand(name="water", smiles="O", charge=0)
        
        orca_generator.generate_from_optimization_results(
            optimization_results=sample_optimization_results,
            metal=metal,
            ligand=ligand,
            n_lowest=1,
            multiplicities=[1, 3]
        )
        
        # Check summary file
        summary_file = orca_generator.orca_dir / "orca_generation_summary.json"
        assert summary_file.exists()
        
        # Load and check content
        with open(summary_file) as f:
            summary = json.load(f)
        
        assert "generation_info" in summary
        assert "generated_files" in summary
        assert "file_details" in summary
        
        # Check configuration was saved
        assert summary["generation_info"]["orca_config"]["method"] == "B3LYP"
        assert summary["generation_info"]["orca_config"]["basis_set"] == "def2-SVP"


class TestMetalLigandComplexORCA:
    """Test ORCA functionality in MetalLigandComplex."""
    
    @pytest.fixture
    def complex_with_results(self, temp_dir):
        """Create complex with mock optimization results."""
        config = ComplexConfig(
            experiment_name="test_orca",
            output_dir=temp_dir,
            keep_top_n=3,
            orca_config=ORCAConfig(
                method="PBE0",
                basis_set="def2-TZVP",
                n_cores=8
            )
        )
        
        complex = MetalLigandComplex(
            ligand_name="water",
            ligand_smiles="O",
            metal_symbol="Cu",
            metal_charge=2,
            ligand_charge=0,
            config=config
        )
        
        # Mock optimization results
        opt_results = {
            "complexes": [
                OptimizationResult(
                    success=True,
                    initial_geometry=Geometry(
                        atoms=["Cu", "O", "H", "H"],
                        coordinates=np.array([
                            [0.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [2.5, 0.5, 0.0],
                            [2.5, -0.5, 0.0]
                        ]),
                        title=f"Cu_H2O_{i}"
                    ),
                    optimized_geometry=Geometry(
                        atoms=["Cu", "O", "H", "H"],
                        coordinates=np.array([
                            [0.0, 0.0, 0.0],
                            [1.95, 0.0, 0.0],
                            [2.45, 0.48, 0.0],
                            [2.45, -0.48, 0.0]
                        ]),
                        title=f"Cu_H2O_{i}_opt"
                    ),
                    energy=-1640.5 - i * 0.01  # Different energies
                )
                for i in range(5)
            ]
        }
        
        complex.results["optimization_results"] = opt_results
        
        return complex
    
    def test_prepare_orca_inputs(self, complex_with_results):
        """Test ORCA input preparation from complex."""
        results = complex_with_results.prepare_orca_inputs(
            n_best=3,
            multiplicities=[2]  # Cu2+ is d9, doublet
        )
        
        assert "complex_inputs" in results
        assert len(results["complex_inputs"]) == 3  # 3 structures * 1 multiplicity
        
        # Check directory structure
        orca_dir = complex_with_results.work_dir / "04_orca_inputs"
        assert orca_dir.exists()
        
        complex_dir = orca_dir / "complexes" / "mult_2"
        assert complex_dir.exists()
        
        inp_files = list(complex_dir.glob("*.inp"))
        assert len(inp_files) == 3
    
    def test_prepare_orca_complexes_only(self, complex_with_results):
        """Test ORCA preparation for complexes only."""
        results = complex_with_results.prepare_orca_for_complexes_only(
            n_best=2,
            multiplicities=[2]
        )
        
        # Should only have complex inputs
        assert "complex_inputs" in results
        assert len(results["complex_inputs"]) == 2
        
        # Should not have metal or ligand inputs
        assert len(results.get("metal_inputs", [])) == 0
        assert len(results.get("ligand_inputs", [])) == 0