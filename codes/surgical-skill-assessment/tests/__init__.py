"""
Surgical Skill Assessment Test Suite

Comprehensive test suite for validating surgical skill assessment functionality,
including video classification models and evaluation metrics.

Test Coverage:
- Model architecture validation
- Data loading and preprocessing
- Training pipeline functionality
- Evaluation metrics accuracy
- Configuration handling
- Cross-validation procedures

Author: Cataract-LMM Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Cataract-LMM Team"

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

__all__ = ["run_all_tests"]


def run_all_tests():
    """Run all tests in the test suite."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)
