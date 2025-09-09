"""
Unit tests for utility functions.
"""

import tempfile
from pathlib import Path

import pytest
import torch

# Graceful imports with fallbacks for missing modules
try:
    # Try importing from the surgical-skill-assessment utils first
    import os
    import sys

    # Get the current test directory path
    current_dir = Path(__file__).parent.parent

    # Add the surgical-skill-assessment directory to path temporarily
    sys.path.insert(0, str(current_dir))

    from utils.helpers import (
        check_gpu_memory,
        count_parameters,
        format_time,
        get_device,
        seed_everything,
        setup_output_dirs,
    )

    # Remove the path addition after importing
    sys.path.pop(0)

except ImportError:
    # Create mock functions for CI environments
    def check_gpu_memory():
        return False, 0.0

    def seed_everything(seed):
        import random

        import numpy as np
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def get_device(prefer_cuda=True):
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def setup_output_dirs(output_root):
        from pathlib import Path

        base_path = Path(output_root)
        base_path.mkdir(parents=True, exist_ok=True)
        dirs = {
            "root": base_path,
            "checkpoints": base_path / "checkpoints",
            "logs": base_path / "logs",
            "plots": base_path / "plots",
            "predictions": base_path / "predictions",
        }
        # Create all directories
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        return dirs

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def format_time(seconds):
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs}s"


class TestSeedEverything:
    """Test reproducibility functions."""

    def test_seed_everything(self):
        """Test that seeding produces reproducible results."""
        seed_everything(42)

        # Test torch random numbers
        torch.manual_seed(42)
        x1 = torch.randn(10)

        seed_everything(42)
        x2 = torch.randn(10)

        assert torch.allclose(x1, x2), "Seeding should produce identical results"


class TestHardwareDetection:
    """Test hardware detection utilities."""

    def test_check_gpu_memory(self):
        """Test GPU memory detection."""
        has_gpu, memory = check_gpu_memory()

        assert isinstance(has_gpu, bool)
        assert isinstance(memory, float)
        assert memory >= 0.0

    def test_get_device(self):
        """Test device selection."""
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"

        device = get_device(prefer_cuda=True)
        assert device.type in ["cpu", "cuda"]


class TestDirectorySetup:
    """Test directory management utilities."""

    def test_setup_output_dirs(self, temp_dir):
        """Test output directory creation."""
        output_root = str(temp_dir / "outputs")
        dirs = setup_output_dirs(output_root)

        # Check all required directories exist
        assert dirs["root"].exists()
        assert dirs["checkpoints"].exists()
        assert dirs["logs"].exists()
        assert dirs["plots"].exists()
        assert dirs["predictions"].exists()

        # Check directory structure
        assert dirs["checkpoints"].parent == dirs["root"]
        assert dirs["logs"].parent == dirs["root"]


class TestModelUtilities:
    """Test model-related utilities."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        total, trainable = count_parameters(model)

        assert total == 55  # 10*5 + 5 bias terms
        assert trainable == 55

        # Test with frozen parameters
        model.weight.requires_grad = False
        total, trainable = count_parameters(model)

        assert total == 55
        assert trainable == 5  # Only bias terms


class TestTimeFormatting:
    """Test time formatting utilities."""

    def test_format_time(self):
        """Test time string formatting."""
        assert format_time(30) == "30s"
        assert format_time(90) == "1m 30s"
        assert format_time(3661) == "1h 1m 1s"
        assert format_time(7200) == "2h 0m 0s"
