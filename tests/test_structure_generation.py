"""
Tests for structure generation components.

Tests cover:
- Conformer generation
- Binding site detection
- Pose generation
- Metal geometry generation
"""

import pytest
import numpy as np
import shutil
from pathlib import Path

from ml_xtb_prescreening.core.data_models import (
    Geometry, Metal, BindingSite, BindingSiteType, CoordinationGeometry
)
from ml_xtb_prescreening.generators.binding_site_detector import BindingSiteDetector
from ml_xtb_prescreening.generators.pose_generator import PoseGenerator
from ml_xtb_prescreening.generators.metal_generator import MetalGenerator
from ml_xtb_prescreening.optimizers.conformer_generator import ConformerGenerator


class TestConformerGenerator:
    """Test conformer generation functionality."""
    
    @pytest.fixture
    def conformer_gen(self):
        """Create conformer generator instance."""
        return ConformerGenerator()
    
    @pytest.mark.skipif(not shutil.which("obabel"), reason="Open Babel not available")
    def test_conformer_generation_simple(self, conformer_gen, tmp_path):
        """Test conformer generation for simple molecule."""
        # Generate conformers for ethanol
        conformers = conformer_gen.from_smiles(
            smiles="CCO",
            n_conformers=5,
            work_dir=tmp_path
        )
        
        assert len(conformers) > 0
        assert len(conformers) <= 5
        
        # Check structure
        for conf in conformers:
            assert len(conf.atoms) == 9  # C2H6O
            assert conf.coordinates.shape == (9, 3)
            assert "C" in conf.atoms
            assert "O" in conf.atoms
            assert "H" in conf.atoms
    
    @pytest.mark.skipif(not shutil.which("obabel"), reason="Open Babel not available")
    def test_conformer_generation_complex(self, conformer_gen, tmp_path):
        """Test conformer generation for complex molecule."""
        # Generate conformers for a more complex molecule (citric acid)
        conformers = conformer_gen.from_smiles(
            smiles="C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
            n_conformers=10,
            energy_window=50.0,
            rmsd_threshold=0.5,
            work_dir=tmp_path
        )
        
        assert len(conformers) > 0
        
        # Check that conformers are different
        if len(conformers) > 1:
            # Compare first two conformers
            rmsd = np.sqrt(np.mean((conformers[0].coordinates - conformers[1].coordinates)**2))
            assert rmsd > 0.1  # Should be different
    
    def test_conformer_rmsd_calculation(self, conformer_gen):
        """Test RMSD calculation between conformers."""
        # Create two identical geometries
        atoms = ["C", "O"]
        coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        geom1 = Geometry(atoms=atoms, coordinates=coords1)
        geom2 = Geometry(atoms=atoms, coordinates=coords2)
        
        rmsd = conformer_gen._calculate_rmsd(geom1, geom2)
        assert rmsd < 1e-6  # Should be essentially zero
        
        # Create different geometry
        coords3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        geom3 = Geometry(atoms=atoms, coordinates=coords3)
        
        rmsd2 = conformer_gen._calculate_rmsd(geom1, geom3)
        assert rmsd2 > 0.9  # Should be about 1.0


class TestBindingSiteDetector:
    """Test binding site detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create binding site detector instance."""
        return BindingSiteDetector()
    
    def test_carboxylate_detection(self, detector):
        """Test detection of carboxylate groups."""
        # Create acetate geometry (CH3COO-)
        atoms = ["C", "C", "O", "O", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],    # C (methyl)
            [1.5, 0.0, 0.0],    # C (carboxyl)
            [2.0, 1.2, 0.0],    # O
            [2.0, -1.2, 0.0],   # O
            [-0.5, 1.0, 0.0],   # H
            [-0.5, -0.5, 0.9],  # H
            [-0.5, -0.5, -0.9]  # H
        ])
        
        geom = Geometry(atoms=atoms, coordinates=coords)
        sites = detector.detect_sites(geom)
        
        # Should find carboxylate site
        assert len(sites) > 0
        carboxylate_sites = [s for s in sites if s.site_type == BindingSiteType.CARBOXYLATE]
        assert len(carboxylate_sites) > 0
        
        # Check that it identified both oxygens
        site = carboxylate_sites[0]
        assert 2 in site.atom_indices  # First oxygen
        assert 3 in site.atom_indices  # Second oxygen
    
    def test_amine_detection(self, detector):
        """Test detection of amine groups."""
        # Create methylamine geometry (CH3NH2)
        atoms = ["C", "N", "H", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],    # C
            [1.5, 0.0, 0.0],    # N
            [-0.5, 1.0, 0.0],   # H (on C)
            [-0.5, -0.5, 0.9],  # H (on C)
            [-0.5, -0.5, -0.9], # H (on C)
            [2.0, 0.5, 0.5],    # H (on N)
            [2.0, 0.5, -0.5]    # H (on N)
        ])
        
        geom = Geometry(atoms=atoms, coordinates=coords)
        sites = detector.detect_sites(geom)
        
        # Should find amine site
        amine_sites = [s for s in sites if s.site_type == BindingSiteType.AMINE]
        assert len(amine_sites) > 0
        assert 1 in amine_sites[0].atom_indices  # Nitrogen
    
    def test_hydroxyl_detection(self, detector):
        """Test detection of hydroxyl groups."""
        # Create methanol geometry (CH3OH)
        atoms = ["C", "O", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],    # C
            [1.5, 0.0, 0.0],    # O
            [-0.5, 1.0, 0.0],   # H (on C)
            [-0.5, -0.5, 0.9],  # H (on C)
            [-0.5, -0.5, -0.9], # H (on C)
            [2.0, 0.5, 0.0]     # H (on O)
        ])
        
        geom = Geometry(atoms=atoms, coordinates=coords)
        sites = detector.detect_sites(geom)
        
        # Should find hydroxyl site
        hydroxyl_sites = [s for s in sites if s.site_type == BindingSiteType.HYDROXYL]
        assert len(hydroxyl_sites) > 0
        assert 1 in hydroxyl_sites[0].atom_indices  # Oxygen
    
    def test_site_clustering(self, detector):
        """Test clustering of nearby binding sites."""
        # Create a molecule with two close carboxylate groups
        atoms = ["C", "C", "O", "O", "C", "O", "O"]
        coords = np.array([
            [0.0, 0.0, 0.0],    # C
            [1.5, 0.0, 0.0],    # C (carboxyl 1)
            [2.0, 1.2, 0.0],    # O
            [2.0, -1.2, 0.0],   # O
            [3.0, 0.0, 0.0],    # C (carboxyl 2)
            [3.5, 1.2, 0.0],    # O
            [3.5, -1.2, 0.0]    # O
        ])
        
        geom = Geometry(atoms=atoms, coordinates=coords)
        
        # Detect without clustering
        detector._cluster_sites = lambda sites, threshold: sites
        sites_unclustered = detector.detect_sites(geom)
        
        # Detect with clustering (normal behavior)
        detector_clustered = BindingSiteDetector()
        sites_clustered = detector_clustered.detect_sites(geom)
        
        # Clustering should reduce the number of sites
        assert len(sites_clustered) <= len(sites_unclustered)


class TestPoseGenerator:
    """Test pose generation functionality."""
    
    @pytest.fixture
    def pose_gen(self):
        """Create pose generator instance."""
        return PoseGenerator()
    
    def test_tetrahedral_geometry(self, pose_gen):
        """Test tetrahedral coordination geometry generation."""
        # Create simple ligand with one binding site
        ligand_atoms = ["O", "H", "H"]
        ligand_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        ligand = Geometry(atoms=ligand_atoms, coordinates=ligand_coords)
        
        # Create binding site at oxygen
        binding_site = BindingSite(
            atom_indices=[0],
            site_type=BindingSiteType.HYDROXYL,
            score=0.5,
            position=ligand_coords[0]
        )
        
        # Create metal
        metal = Metal(symbol="Zn", charge=2)
        
        # Generate poses
        poses = pose_gen.generate_poses(
            ligand_geometry=ligand,
            metal=metal,
            binding_sites=[binding_site],
            max_poses=1
        )
        
        assert len(poses) > 0
        pose = poses[0]
        
        # Check that metal was added
        assert "Zn" in pose.atoms
        assert len(pose.atoms) == len(ligand_atoms) + 1
        assert pose.coordinates.shape[0] == len(pose.atoms)
    
    def test_coordination_number_calculation(self, pose_gen):
        """Test coordination number calculation for different geometries."""
        assert pose_gen._get_coordination_number(CoordinationGeometry.TETRAHEDRAL) == 4
        assert pose_gen._get_coordination_number(CoordinationGeometry.SQUARE_PLANAR) == 4
        assert pose_gen._get_coordination_number(CoordinationGeometry.TRIGONAL_BIPYRAMIDAL) == 5
        assert pose_gen._get_coordination_number(CoordinationGeometry.SQUARE_PYRAMIDAL) == 5
        assert pose_gen._get_coordination_number(CoordinationGeometry.OCTAHEDRAL) == 6
    
    def test_ideal_geometry_vectors(self, pose_gen):
        """Test ideal coordination geometry vectors."""
        # Test tetrahedral vectors
        tet_vectors = pose_gen._tetrahedral_vectors()
        assert tet_vectors.shape == (4, 3)
        # Vectors should be normalized
        norms = np.linalg.norm(tet_vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        
        # Test octahedral vectors
        oct_vectors = pose_gen._octahedral_vectors()
        assert oct_vectors.shape == (6, 3)
        # Should have pairs of opposite vectors
        assert np.allclose(oct_vectors[0], -oct_vectors[1])
        assert np.allclose(oct_vectors[2], -oct_vectors[3])
        assert np.allclose(oct_vectors[4], -oct_vectors[5])


class TestMetalGenerator:
    """Test metal geometry generation."""
    
    @pytest.fixture
    def metal_gen(self):
        """Create metal generator instance."""
        return MetalGenerator()
    
    def test_bare_metal_generation(self, metal_gen):
        """Test generation of bare metal ion."""
        metal = Metal(symbol="Co", charge=2)
        geom = metal_gen.generate_metal_geometry(metal)
        
        assert len(geom.atoms) == 1
        assert geom.atoms[0] == "Co"
        assert geom.coordinates.shape == (1, 3)
        assert np.allclose(geom.coordinates[0], [0.0, 0.0, 0.0])
        assert "Co2+" in geom.title
    
    def test_hydrated_metal_generation(self, metal_gen):
        """Test generation of hydrated metal complex."""
        metal = Metal(symbol="Ni", charge=2)
        metal.typical_bond_lengths = {"O": 2.05}
        
        # Generate with 6 water molecules (octahedral)
        geom = metal_gen.generate_hydrated_metal(metal, n_water=6)
        
        # Should have 1 metal + 6*3 atoms (O + 2H per water)
        assert len(geom.atoms) == 19  # 1 + 6*3
        assert geom.atoms[0] == "Ni"
        assert geom.atoms.count("O") == 6
        assert geom.atoms.count("H") == 12
        
        # Check metal-oxygen distances
        metal_pos = geom.coordinates[0]
        for i in range(1, 19, 3):  # Oxygen positions
            if geom.atoms[i] == "O":
                distance = np.linalg.norm(geom.coordinates[i] - metal_pos)
                assert abs(distance - 2.05) < 0.1  # Should be close to typical distance
    
    def test_tetrahedral_hydration(self, metal_gen):
        """Test tetrahedral hydration geometry."""
        metal = Metal(symbol="Zn", charge=2)
        metal.typical_bond_lengths = {"O": 2.1}
        
        geom = metal_gen.generate_hydrated_metal(metal, n_water=4)
        
        # Should have 1 metal + 4*3 atoms
        assert len(geom.atoms) == 13
        assert geom.atoms.count("O") == 4
        
        # Check tetrahedral arrangement
        metal_pos = geom.coordinates[0]
        oxygen_positions = []
        for i, atom in enumerate(geom.atoms):
            if atom == "O" and i > 0:
                oxygen_positions.append(geom.coordinates[i])
        
        assert len(oxygen_positions) == 4
        
        # Check angles between oxygen atoms (should be ~109.5° for tetrahedral)
        # This is a simplified check
        vectors = [pos - metal_pos for pos in oxygen_positions]
        angles = []
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                cos_angle = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                angles.append(angle)
        
        # Tetrahedral angles should be around 109.5°
        assert all(90 < angle < 120 for angle in angles)