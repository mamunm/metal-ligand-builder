"""
Pytest configuration and shared fixtures for ml-xtb-prescreening tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after test."""
    temp_dir = tempfile.mkdtemp(prefix="mlxtb_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def check_dependencies():
    """Check for required external dependencies."""
    dependencies = {
        "obabel": shutil.which("obabel"),
        "xtb": shutil.which("xtb")
    }
    return dependencies


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_obabel: marks tests that require Open Babel"
    )
    config.addinivalue_line(
        "markers", "requires_xtb: marks tests that require xTB"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test requirements."""
    for item in items:
        # Add markers based on test function names or docstrings
        if "obabel" in item.nodeid.lower() or "conformer" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_obabel)
        
        if "xtb" in item.nodeid.lower() or "optimize" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_xtb)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "workflow" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)


# Pytest command line options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--skip-external",
        action="store_true",
        default=False,
        help="skip tests requiring external programs (obabel, xtb)"
    )