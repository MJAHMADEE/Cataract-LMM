"""
Test Configuration and Utilities for pytest
==========================================

This module contains additional pytest configuration, custom fixtures,
and utility functions for the test suite.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")

    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Auto-mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Auto-mark security tests
        if "security" in item.nodeid:
            item.add_marker(pytest.mark.security)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Provide the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root) -> Path:
    """Provide the test data directory."""
    test_data = project_root / "tests" / "data"
    test_data.mkdir(exist_ok=True)
    return test_data


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Provide a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        yield workspace


@pytest.fixture
def clean_environment():
    """Provide a clean environment for tests."""
    # Store original environment
    original_env = os.environ.copy()
    original_path = sys.path.copy()

    yield

    # Restore environment
    os.environ.clear()
    os.environ.update(original_env)
    sys.path.clear()
    sys.path.extend(original_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration for testing."""
    return {
        "model": {
            "name": "test_model",
            "version": "1.0.0",
            "parameters": {"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
        },
        "data": {
            "input_dir": "/tmp/test_input",
            "output_dir": "/tmp/test_output",
            "formats": ["jpg", "png", "mp4"],
        },
        "training": {"device": "cpu", "num_workers": 2, "pin_memory": False},
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "save_predictions": True,
        },
    }


@pytest.fixture
def sample_project_structure(temp_workspace) -> Path:
    """Create a sample project structure for testing."""
    # Create directory structure
    directories = [
        "src",
        "src/models",
        "src/data",
        "src/utils",
        "tests",
        "configs",
        "docs",
        "scripts",
    ]

    for directory in directories:
        (temp_workspace / directory).mkdir(parents=True, exist_ok=True)

    # Create sample files
    files = {
        "README.md": "# Test Project\n\nThis is a test project.",
        "requirements.txt": "pytest>=6.0.0\nnumpy>=1.19.0\n",
        "setup.py": 'from setuptools import setup\nsetup(name="test-project")',
        "src/__init__.py": "",
        "src/models/__init__.py": "",
        "src/data/__init__.py": "",
        "src/utils/__init__.py": "",
        "tests/__init__.py": "",
        "configs/config.yaml": "model:\n  name: test\n",
        ".gitignore": "__pycache__/\n*.pyc\n",
    }

    for file_path, content in files.items():
        full_path = temp_workspace / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    return temp_workspace


@pytest.fixture
def captured_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(handler)

    yield log_capture

    logger.removeHandler(handler)


class TestFileManager:
    """Utility class for managing test files."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.created_files = []
        self.created_dirs = []

    def create_file(self, relative_path: str, content: str = "") -> Path:
        """Create a test file with content."""
        file_path = self.base_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        self.created_files.append(file_path)
        return file_path

    def create_directory(self, relative_path: str) -> Path:
        """Create a test directory."""
        dir_path = self.base_dir / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        self.created_dirs.append(dir_path)
        return dir_path

    def cleanup(self):
        """Clean up created files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()

        for dir_path in reversed(self.created_dirs):
            if dir_path.exists() and not any(dir_path.iterdir()):
                dir_path.rmdir()


@pytest.fixture
def file_manager(temp_workspace) -> Generator[TestFileManager, None, None]:
    """Provide a file manager for tests."""
    manager = TestFileManager(temp_workspace)
    yield manager
    manager.cleanup()


class MockModule:
    """Mock module for testing imports."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.__version__ = version

    def mock_function(self, *args, **kwargs):
        """Mock function."""
        return f"mock_result_{self.name}"


@pytest.fixture
def mock_modules() -> Dict[str, MockModule]:
    """Provide mock modules for testing."""
    return {
        "surgical_video_processing": MockModule("surgical_video_processing"),
        "surgical_instance_segmentation": MockModule("surgical_instance_segmentation"),
        "surgical_phase_recognition": MockModule("surgical_phase_recognition"),
        "surgical_skill_assessment": MockModule("surgical_skill_assessment"),
    }


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip tests based on markers and conditions
    if "slow" in item.keywords and item.config.getoption("--skip-slow", default=False):
        pytest.skip("Skipping slow test")

    if "integration" in item.keywords and item.config.getoption(
        "--skip-integration", default=False
    ):
        pytest.skip("Skipping integration test")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="Skip slow tests"
    )
    parser.addoption(
        "--skip-integration",
        action="store_true",
        default=False,
        help="Skip integration tests",
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run performance tests",
    )


@pytest.fixture(autouse=True)
def test_environment_setup():
    """Automatically set up test environment for all tests."""
    # Ensure test-specific environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"

    yield

    # Clean up test environment
    os.environ.pop("TESTING", None)


class TestDataGenerator:
    """Generate test data for various test scenarios."""

    @staticmethod
    def generate_config_data(config_type: str = "basic") -> Dict[str, Any]:
        """Generate configuration data."""
        if config_type == "basic":
            return {
                "name": "test_config",
                "version": "1.0.0",
                "settings": {"debug": True, "verbose": False},
            }
        elif config_type == "model":
            return {
                "model": {
                    "architecture": "resnet50",
                    "pretrained": True,
                    "num_classes": 10,
                },
                "training": {"epochs": 50, "learning_rate": 0.001, "batch_size": 32},
            }
        else:
            return {}

    @staticmethod
    def generate_test_files(base_dir: Path, file_count: int = 5) -> List[Path]:
        """Generate test files."""
        files = []
        for i in range(file_count):
            file_path = base_dir / f"test_file_{i}.txt"
            file_path.write_text(f"Test content {i}")
            files.append(file_path)
        return files


@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Provide test data generator."""
    return TestDataGenerator()


# Performance testing utilities
class PerformanceMonitor:
    """Monitor performance during tests."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def start(self):
        """Start performance monitoring."""
        import time

        self.start_time = time.time()

    def stop(self):
        """Stop performance monitoring."""
        import time

        self.end_time = time.time()
        return self.end_time - self.start_time if self.start_time else 0

    def record_metric(self, name: str, value: Any):
        """Record a performance metric."""
        self.metrics[name] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get recorded metrics."""
        return self.metrics.copy()


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Provide performance monitor."""
    return PerformanceMonitor()


# Security testing utilities
class SecurityValidator:
    """Validate security aspects during tests."""

    @staticmethod
    def check_file_permissions(file_path: Path) -> bool:
        """Check if file has secure permissions."""
        try:
            stat_info = file_path.stat()
            # Check that file is not world-writable
            return not (stat_info.st_mode & 0o002)
        except OSError:
            return False

    @staticmethod
    def check_for_secrets(content: str) -> List[str]:
        """Check content for potential secrets."""
        import re

        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]

        found_secrets = []
        for pattern in secret_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_secrets.extend(matches)

        return found_secrets


@pytest.fixture
def security_validator() -> SecurityValidator:
    """Provide security validator."""
    return SecurityValidator()


# Test result collection
class TestResultCollector:
    """Collect and analyze test results."""

    def __init__(self):
        self.results = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "warnings": [],
        }

    def record_result(self, test_name: str, status: str, message: str = ""):
        """Record a test result."""
        if status in self.results:
            if isinstance(self.results[status], int):
                self.results[status] += 1

        if status == "failed" and message:
            self.results["errors"].append(f"{test_name}: {message}")

    def get_summary(self) -> Dict[str, Any]:
        """Get test results summary."""
        total = (
            self.results["passed"] + self.results["failed"] + self.results["skipped"]
        )
        success_rate = (self.results["passed"] / total * 100) if total > 0 else 0

        return {
            "total_tests": total,
            "passed": self.results["passed"],
            "failed": self.results["failed"],
            "skipped": self.results["skipped"],
            "success_rate": round(success_rate, 2),
            "errors": self.results["errors"],
        }


@pytest.fixture(scope="session")
def test_result_collector() -> TestResultCollector:
    """Provide test result collector."""
    return TestResultCollector()
