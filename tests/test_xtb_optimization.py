"""
Tests for XTB optimization components.

Tests cover:
- XTB optimizer functionality
- XTB workflow manager
- Parallel optimization
- Result parsing
"""

import pytest
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np

from ml_xtb_prescreening.core.data_models import Geometry, OptimizationResult
from ml_xtb_prescreening.core.config import XTBConfig
from ml_xtb_prescreening.optimizers.xtb_optimizer import XTBOptimizer
from ml_xtb_prescreening.optimizers.xtb_workflow import XTBWorkflowManager


class TestXTBOptimizer:
    """Test XTB optimizer functionality."""
    
    @pytest.fixture
    def xtb_optimizer(self):
        """Create XTB optimizer instance."""
        config = XTBConfig(
            method="gfn2",
            convergence="loose",  # Faster for tests
            max_iterations=50
        )
        return XTBOptimizer(config)
    
    @pytest.fixture
    def water_geometry(self):
        """Create water molecule geometry."""
        atoms = ["O", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0]
        ])
        return Geometry(atoms=atoms, coordinates=coords, title="water")
    
    def test_xtb_availability(self, xtb_optimizer):
        """Test XTB availability check."""
        available = xtb_optimizer.check_available()
        
        if shutil.which("xtb"):
            assert available is True
        else:
            assert available is False
    
    @pytest.mark.skipif(not shutil.which("xtb"), reason="xTB not available")
    def test_optimize_water(self, xtb_optimizer, water_geometry, tmp_path):
        """Test optimization of water molecule."""
        result = xtb_optimizer.optimize(
            geometry=water_geometry,
            charge=0,
            multiplicity=1,
            work_dir=tmp_path / "water_opt"
        )
        
        assert result.success is True
        assert result.optimized_geometry is not None
        assert result.energy is not None
        assert result.energy < 0  # Should be negative
        
        # Check that coordinates changed (optimization occurred)
        initial_coords = water_geometry.coordinates
        final_coords = result.optimized_geometry.coordinates
        assert not np.allclose(initial_coords, final_coords, atol=1e-4)
        
        # Check properties
        assert "homo_lumo_gap" in result.properties
        assert result.properties["homo_lumo_gap"] > 0
    
    @pytest.mark.skipif(not shutil.which("xtb"), reason="xTB not available")
    @pytest.mark.requires_xtb
    def test_optimize_charged_system(self, xtb_optimizer, tmp_path):
        """Test optimization of charged system."""
        # Create Na+ ion
        atoms = ["Na"]
        coords = np.array([[0.0, 0.0, 0.0]])
        sodium = Geometry(atoms=atoms, coordinates=coords, title="sodium_cation")
        
        result = xtb_optimizer.optimize(
            geometry=sodium,
            charge=1,
            multiplicity=1,
            work_dir=tmp_path / "na_opt"
        )
        
        assert result.success is True
        assert result.energy is not None
    
    @pytest.mark.skipif(not shutil.which("xtb"), reason="xTB not available")
    @pytest.mark.requires_xtb
    def test_optimization_failure(self, xtb_optimizer, tmp_path):
        """Test handling of optimization failure."""
        # Create geometry with atoms too close (will fail)
        atoms = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])  # Too close
        bad_geom = Geometry(atoms=atoms, coordinates=coords, title="bad_h2")
        
        result = xtb_optimizer.optimize(
            geometry=bad_geom,
            charge=0,
            multiplicity=1,
            work_dir=tmp_path / "bad_opt"
        )
        
        # Should handle failure gracefully
        assert result.success is False
        assert result.error_message is not None
        assert result.optimized_geometry is None
    
    def test_xtb_config_to_args(self):
        """Test XTB configuration to command line arguments conversion."""
        config = XTBConfig(
            method="gfn1",
            solvent="dmso",
            accuracy=0.1,
            electronic_temperature=500.0,
            convergence="tight"
        )
        
        args = config.to_cmd_args()
        
        assert "--gfn1" in args
        assert "--alpb" in args
        assert "dmso" in args
        assert "--acc" in args
        assert "0.1" in args
        assert "--etemp" in args
        assert "500.0" in args
        assert "--tight" in args


class TestXTBWorkflowManager:
    """Test XTB workflow manager functionality."""
    
    @pytest.fixture
    def setup_test_structures(self, tmp_path):
        """Set up test directory with initial structures."""
        # Create directory structure
        base_dir = tmp_path / "test_experiment"
        initial_dir = base_dir / "01_initial_structures"
        
        # Create subdirectories
        (initial_dir / "metals").mkdir(parents=True)
        (initial_dir / "ligands").mkdir(parents=True)
        (initial_dir / "complexes").mkdir(parents=True)
        
        # Create test structures
        # Metal: Li+
        metal_geom = Geometry(
            atoms=["Li"],
            coordinates=np.array([[0.0, 0.0, 0.0]]),
            title="lithium_ion"
        )
        metal_geom.save_xyz(initial_dir / "metals" / "Li_ion.xyz")
        
        # Ligand: Water
        water_geom = Geometry(
            atoms=["O", "H", "H"],
            coordinates=np.array([
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0]
            ]),
            title="water"
        )
        water_geom.save_xyz(initial_dir / "ligands" / "water_conf_000.xyz")
        
        # Complex: Li-H2O
        complex_geom = Geometry(
            atoms=["O", "H", "H", "Li"],
            coordinates=np.array([
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
                [0.0, 0.0, 2.0]
            ]),
            title="water_Li_complex"
        )
        complex_geom.save_xyz(initial_dir / "complexes" / "water_Li_pose_000.xyz")
        
        return base_dir
    
    @pytest.fixture
    def workflow_manager(self, setup_test_structures):
        """Create workflow manager with test structures."""
        config = XTBConfig(
            method="gfn2",
            convergence="loose",
            max_iterations=30
        )
        
        return XTBWorkflowManager(
            base_dir=setup_test_structures,
            config=config,
            n_workers=1,
            create_folders=True
        )
    
    def test_workflow_initialization(self, workflow_manager, setup_test_structures):
        """Test workflow manager initialization."""
        assert workflow_manager.base_dir == setup_test_structures
        assert workflow_manager.input_dir == setup_test_structures / "01_initial_structures"
        assert workflow_manager.output_dir == setup_test_structures / "02_optimized_structures"
        assert workflow_manager.n_workers == 1
        assert workflow_manager.create_folders is True
    
    @pytest.mark.skipif(not shutil.which("xtb"), reason="xTB not available")
    def test_optimize_all_structures(self, workflow_manager):
        """Test optimization of all structures."""
        charge_map = {
            "metals": 1,     # Li+
            "ligands": 0,    # H2O
            "complexes": 1   # Li+ + H2O
        }
        
        results = workflow_manager.optimize_all_structures(charge_map=charge_map)
        
        # Check results structure
        assert isinstance(results, dict)
        assert "metals" in results
        assert "ligands" in results
        assert "complexes" in results
        
        # Check that optimizations were attempted
        assert len(results["metals"]) == 1
        assert len(results["ligands"]) == 1
        assert len(results["complexes"]) == 1
        
        # Check output directory was created
        assert workflow_manager.output_dir.exists()
        assert (workflow_manager.output_dir / "metals").exists()
        assert (workflow_manager.output_dir / "ligands").exists()
        assert (workflow_manager.output_dir / "complexes").exists()
        
        # Check summary files
        metal_summary = workflow_manager.output_dir / "metals" / "metals_optimization_summary.json"
        assert metal_summary.exists()
        
        # Load and check summary content
        with open(metal_summary) as f:
            summary_data = json.load(f)
            assert summary_data["structure_type"] == "metals"
            assert summary_data["n_structures"] == 1
    
    @pytest.mark.skipif(not shutil.which("xtb"), reason="xTB not available")
    @pytest.mark.requires_xtb
    @pytest.mark.slow
    def test_optimize_without_folders(self, setup_test_structures):
        """Test optimization without creating folders."""
        workflow_manager = XTBWorkflowManager(
            base_dir=setup_test_structures,
            config=XTBConfig(convergence="loose"),
            n_workers=1,
            create_folders=False  # Don't create folders
        )
        
        results = workflow_manager.optimize_all_structures(
            charge_map={"metals": 1, "ligands": 0, "complexes": 1}
        )
        
        # Check that only XYZ files were created
        metals_dir = workflow_manager.output_dir / "metals"
        if metals_dir.exists():
            # Should have optimized XYZ files
            xyz_files = list(metals_dir.glob("*_opt.xyz"))
            assert len(xyz_files) > 0
            
            # Should NOT have subdirectories
            subdirs = [d for d in metals_dir.iterdir() if d.is_dir()]
            assert len(subdirs) == 0
    
    def test_parallel_optimization(self, setup_test_structures):
        """Test parallel optimization with multiple workers."""
        # Create additional structures for parallel testing
        ligands_dir = setup_test_structures / "01_initial_structures" / "ligands"
        
        # Add more water conformers
        for i in range(1, 4):
            water_geom = Geometry(
                atoms=["O", "H", "H"],
                coordinates=np.array([
                    [0.0, 0.0, 0.0],
                    [0.96 + i*0.01, 0.0, 0.0],  # Slightly different
                    [-0.24, 0.93, 0.0]
                ]),
                title=f"water_conf_{i}"
            )
            water_geom.save_xyz(ligands_dir / f"water_conf_{i:03d}.xyz")
        
        # Create workflow with multiple workers
        workflow_manager = XTBWorkflowManager(
            base_dir=setup_test_structures,
            config=XTBConfig(convergence="loose"),
            n_workers=2,  # Use 2 workers
            create_folders=False
        )
        
        if shutil.which("xtb"):
            results = workflow_manager.optimize_all_structures(
                charge_map={"metals": 1, "ligands": 0, "complexes": 1}
            )
            
            # Should have optimized all ligands
            assert len(results["ligands"]) == 4


class TestResultParsing:
    """Test parsing of XTB output files."""
    
    @pytest.fixture
    def sample_xtb_output(self, tmp_path):
        """Create sample xTB output files for testing."""
        work_dir = tmp_path / "xtb_test"
        work_dir.mkdir()
        
        # Create sample JSON output (newer xTB format)
        json_data = {
            "total energy": -5.070874,
            "HOMO-LUMO gap/eV": 13.947,
            "dipole": 1.827,
            "dipole moment/D": 1.827
        }
        
        with open(work_dir / "xtbout.json", 'w') as f:
            json.dump(json_data, f)
        
        # Create sample text output
        output_text = """
          -------------------------------------------------
          |                     SUMMARY                   |
          -------------------------------------------------
          
          :: total energy              -5.070874369574 Eh
          :: gradient norm              0.000254783457 Eh/α
          :: HOMO-LUMO gap             13.947382913015 eV
          
          molecular dipole:
          x           y           z       tot (Debye)
          0.719      -0.000       0.000       1.827
        """
        
        with open(work_dir / "xtb.out", 'w') as f:
            f.write(output_text)
        
        return work_dir
    
    def test_parse_json_output(self, sample_xtb_output):
        """Test parsing of JSON output from xTB."""
        optimizer = XTBOptimizer()
        properties = optimizer._parse_xtb_output(sample_xtb_output)
        
        assert properties["total_energy"] == pytest.approx(-5.070874)
        assert properties["homo_lumo_gap"] == pytest.approx(13.947)
        assert properties["dipole_moment"] == pytest.approx(1.827)
    
    @pytest.mark.requires_xtb
    def test_parse_text_output_fallback(self, sample_xtb_output):
        """Test parsing of text output when JSON is not available."""
        # Remove JSON file to test fallback
        (sample_xtb_output / "xtbout.json").unlink()
        
        optimizer = XTBOptimizer()
        properties = optimizer._parse_xtb_output(sample_xtb_output)
        
        # Should parse from text file
        assert properties["total_energy"] == pytest.approx(-5.070874, rel=1e-5)
        assert properties["dipole_moment"] == pytest.approx(1.827, rel=1e-3)