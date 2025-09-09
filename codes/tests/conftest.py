"""
Test configuration and fixtures for the Cataract-LMM project.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Provide the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "batch_size": 2,
        "num_epochs": 1,
        "learning_rate": 0.001,
        "test_data_size": 10,
        "timeout": 30,
        "gpu_available": False,  # Set to False for CI
        "debug_mode": True,
    }


@pytest.fixture(scope="function")
def mock_environment() -> Generator[Dict[str, str], None, None]:
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": str(PROJECT_ROOT),
    }

    # Set test environment
    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    """Provide a configured logger for tests."""
    logger = logging.getLogger("cataract_lmm_tests")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@pytest.fixture(scope="function")
def sample_video_data(temp_data_dir: Path) -> Dict[str, Path]:
    """Create sample video data for testing."""
    video_dir = temp_data_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    # Create mock video files (empty for testing)
    sample_files = {
        "video1": video_dir / "sample_video_1.mp4",
        "video2": video_dir / "sample_video_2.mp4",
        "annotation": video_dir / "annotations.json",
    }

    for file_path in sample_files.values():
        file_path.touch()

    return sample_files


@pytest.fixture(scope="function")
def sample_image_data(temp_data_dir: Path) -> Dict[str, Path]:
    """Create sample image data for testing."""
    image_dir = temp_data_dir / "images"
    image_dir.mkdir(exist_ok=True)

    sample_files = {
        "image1": image_dir / "sample_image_1.jpg",
        "image2": image_dir / "sample_image_2.jpg",
        "mask1": image_dir / "sample_mask_1.png",
        "mask2": image_dir / "sample_mask_2.png",
    }

    for file_path in sample_files.values():
        file_path.touch()

    return sample_files


@pytest.fixture(scope="function")
def mock_model_config() -> Dict[str, Any]:
    """Provide mock model configuration."""
    return {
        "model_name": "test_model",
        "input_size": [224, 224],
        "num_classes": 10,
        "batch_size": 2,
        "learning_rate": 0.001,
        "num_epochs": 1,
        "device": "cpu",
        "checkpoint_path": None,
        "pretrained": False,
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual functions/classes"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmarking tests"
    )
    config.addinivalue_line("markers", "gpu: Tests requiring GPU/CUDA support")
    config.addinivalue_line("markers", "slow: Tests that take more than 10 seconds")
    config.addinivalue_line("markers", "network: Tests requiring network connectivity")
    config.addinivalue_line("markers", "docker: Tests requiring Docker environment")
    config.addinivalue_line("markers", "e2e: End-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test name patterns
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)

        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)

        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)

        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)

        if "e2e" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.e2e)

        # Add unit marker to all other tests
        if not any(
            marker.name in ["integration", "performance", "gpu", "slow", "e2e"]
            for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)
