"""
Tests for file handling and I/O operations.

Tests cover:
- FileHandler functionality
- Directory organization
- Metadata saving/loading
- ORCA file writing
"""

import pytest
import json
from pathlib import Path
import numpy as np

from ml_xtb_prescreening.io.file_handler import FileHandler
from ml_xtb_prescreening.io.orca_writer import ORCAWriter
from ml_xtb_prescreening.core.data_models import Geometry, Metal, Ligand
from ml_xtb_prescreening.core.config import ORCAConfig


class TestFileHandler:
    """Test FileHandler functionality."""
    
    @pytest.fixture
    def file_handler(self, temp_dir):
        """Create FileHandler instance."""
        return FileHandler(temp_dir, "test_experiment")
    
    def test_directory_creation(self, file_handler):
        """Test that all required directories are created."""
        # Check main directories
        assert file_handler.experiment_dir.exists()
        assert file_handler.dirs["initial"].exists()
        assert file_handler.dirs["optimized"].exists()
        assert file_handler.dirs["reports"].exists()
        assert file_handler.dirs["orca"].exists()
        
        # Check subdirectories
        assert file_handler.dirs["initial_metals"].exists()
        assert file_handler.dirs["initial_ligands"].exists()
        assert file_handler.dirs["initial_complexes"].exists()
        assert file_handler.dirs["optimized_metals"].exists()
        assert file_handler.dirs["optimized_ligands"].exists()
        assert file_handler.dirs["optimized_complexes"].exists()
    
    def test_filename_generation(self, file_handler):
        """Test filename generation methods."""
        # Test metal filenames
        assert file_handler.get_metal_filename("Co", 0) == "Co_ion.xyz"
        assert file_handler.get_metal_filename("Co", 1) == "Co_ion_001.xyz"
        assert file_handler.get_metal_filename("Fe", 10) == "Fe_ion_010.xyz"
        
        # Test ligand filenames
        assert file_handler.get_ligand_filename("edta", 0) == "edta_conf_000.xyz"
        assert file_handler.get_ligand_filename("water", 99) == "water_conf_099.xyz"
        
        # Test complex filenames
        assert file_handler.get_complex_filename("edta", "Co", 0) == "edta_Co_pose_000.xyz"
        assert file_handler.get_complex_filename("edta", "Co", 5, "octahedral") == "edta_Co_pose_005_octahedral.xyz"
    
    def test_metadata_saving(self, file_handler):
        """Test metadata saving functionality."""
        metadata = {
            "experiment": "test",
            "version": 1,
            "parameters": {
                "method": "gfn2",
                "solvent": "water"
            },
            "results": [1.0, 2.0, 3.0]
        }
        
        # Save metadata
        saved_path = file_handler.save_metadata(metadata, "test_metadata.json")
        
        assert saved_path.exists()
        assert saved_path.name == "test_metadata.json"
        
        # Load and verify
        with open(saved_path) as f:
            loaded = json.load(f)
        
        assert loaded["experiment"] == "test"
        assert loaded["version"] == 1
        assert loaded["parameters"]["method"] == "gfn2"
        assert loaded["results"] == [1.0, 2.0, 3.0]
    
    def test_directory_cleaning(self, file_handler):
        """Test directory cleaning functionality."""
        # Create a test file
        test_file = file_handler.dirs["initial_metals"] / "test.xyz"
        test_file.write_text("test content")
        assert test_file.exists()
        
        # Clean directory
        file_handler.clean_directory("initial_metals")
        
        # Directory should exist but be empty
        assert file_handler.dirs["initial_metals"].exists()
        assert not test_file.exists()
        assert len(list(file_handler.dirs["initial_metals"].iterdir())) == 0
    
    def test_file_copying(self, file_handler, temp_dir):
        """Test file copying functionality."""
        # Create source file
        source = temp_dir / "source.xyz"
        source.write_text("xyz content")
        
        # Copy to destination
        dest = file_handler.copy_file(source, "initial_metals")
        
        assert dest.exists()
        assert dest.parent == file_handler.dirs["initial_metals"]
        assert dest.read_text() == "xyz content"
        
        # Copy with rename
        dest2 = file_handler.copy_file(source, "initial_metals", "renamed.xyz")
        assert dest2.name == "renamed.xyz"


class TestORCAWriter:
    """Test ORCA input file writing."""
    
    @pytest.fixture
    def orca_writer(self):
        """Create ORCA writer instance."""
        config = ORCAConfig(
            method="B3LYP",
            basis_set="def2-SVP",
            dispersion="D3BJ",
            solvent_model="CPCM",
            solvent="water"
        )
        return ORCAWriter(config)
    
    @pytest.fixture
    def sample_geometry(self):
        """Create sample geometry for testing."""
        atoms = ["C", "O", "O", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-1.2, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        return Geometry(atoms=atoms, coordinates=coords, title="formate")
    
    def test_orca_input_generation(self, orca_writer, sample_geometry, temp_dir):
        """Test ORCA input file generation."""
        output_file = temp_dir / "test.inp"
        
        result = orca_writer.write_input(
            geometry=sample_geometry,
            charge=-1,
            multiplicity=1,
            output_path=output_file,
            title="Test calculation"
        )
        
        assert result.exists()
        content = result.read_text()
        
        # Check content
        assert "# Test calculation" in content
        assert "! B3LYP def2-SVP" in content
        assert "D3BJ" in content
        assert "CPCM(water)" in content
        assert "* xyz -1 1" in content
        assert "C" in content
        assert "O" in content
        assert "H" in content
        assert "*" in content
    
    def test_orca_complex_input(self, orca_writer, temp_dir):
        """Test ORCA input for metal-ligand complex."""
        # Create Co-NH3 complex
        atoms = ["Co", "N", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],    # Co
            [2.0, 0.0, 0.0],    # N
            [2.5, 0.5, 0.5],    # H
            [2.5, 0.5, -0.5],   # H
            [2.5, -1.0, 0.0]    # H
        ])
        geometry = Geometry(atoms=atoms, coordinates=coords)
        
        metal = Metal(symbol="Co", charge=2)
        ligand = Ligand(name="ammonia", smiles="N", charge=0)
        
        output_file = temp_dir / "complex.inp"
        result = orca_writer.write_complex_input(
            geometry=geometry,
            metal=metal,
            ligand=ligand,
            output_path=output_file
        )
        
        assert result.exists()
        content = result.read_text()
        
        # Check charge (Co2+ + NH3 = +2)
        assert "* xyz 2" in content
        assert "Co-ammonia complex" in content
    
    def test_orca_configuration(self):
        """Test ORCA configuration options."""
        config = ORCAConfig(
            method="PBE0",
            basis_set="def2-TZVP",
            auxiliary_basis="def2/J",
            dispersion=None,
            solvent_model=None,
            n_cores=8,
            additional_keywords=["TightSCF", "Grid6", "FinalGrid7"]
        )
        
        writer = ORCAWriter(config)
        keywords = writer._generate_keywords()
        
        assert "PBE0" in keywords
        assert "def2-TZVP" in keywords
        assert "RIJCOSX def2/J" in keywords
        assert "TightSCF" in keywords
        assert "Grid6" in keywords
        assert "FinalGrid7" in keywords
        
        # Should not include dispersion or solvent
        assert "D3" not in keywords
        assert "CPCM" not in keywords
    
    def test_multiplicity_determination(self, orca_writer):
        """Test automatic multiplicity determination for metals."""
        # Test common metals
        co2 = Metal(symbol="Co", charge=2)  # d7
        mult = orca_writer._determine_multiplicity(co2, Ligand("", "", 0))
        assert mult in [2, 4]  # Could be low or high spin
        
        zn2 = Metal(symbol="Zn", charge=2)  # d10
        mult = orca_writer._determine_multiplicity(zn2, Ligand("", "", 0))
        assert mult == 1  # Always singlet
        
        fe3 = Metal(symbol="Fe", charge=3)  # d5
        mult = orca_writer._determine_multiplicity(fe3, Ligand("", "", 0))
        assert mult == 6  # High spin d5 is sextet
    
    def test_batch_input_writing(self, orca_writer, temp_dir):
        """Test writing multiple ORCA input files."""
        # Create multiple geometries
        geometries = []
        for i in range(3):
            atoms = ["O", "H", "H"]
            coords = np.array([
                [0.0, 0.0, 0.0],
                [0.96 + i*0.01, 0.0, 0.0],
                [-0.24, 0.93, 0.0]
            ])
            geom = Geometry(atoms=atoms, coordinates=coords, title=f"water_{i}")
            geometries.append(geom)
        
        charges = [0, 0, 0]
        multiplicities = [1, 1, 1]
        
        results = orca_writer.write_batch_inputs(
            geometries=geometries,
            charges=charges,
            multiplicities=multiplicities,
            output_dir=temp_dir,
            prefix="water"
        )
        
        assert len(results) == 3
        for i, path in enumerate(results):
            assert path.exists()
            assert path.name == f"water_{i:04d}.inp"
        
        # Check that all files contain valid ORCA input format
        for path in results:
            content = path.read_text()
            assert "!" in content  # Keyword line
            assert "* xyz" in content  # Coordinate block