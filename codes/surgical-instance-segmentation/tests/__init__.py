"""
Test Suite for Surgical Instance Segmentation Framework

This module provides comprehensive testing utilities for all framework components.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: End-to-end workflow testing
- Performance Tests: Speed and memory benchmarking
- Regression Tests: Model accuracy validation
- Data Tests: Dataset integrity validation

Example Usage:
    >>> from surgical_instance_segmentation.tests import run_all_tests
    >>> from surgical_instance_segmentation.tests import TestRunner

    # Run all tests
    >>> run_all_tests()

    # Run specific test category
    >>> runner = TestRunner()
    >>> runner.run_model_tests()
    >>> runner.run_training_tests()

Test Structure:
- test_models/: Model implementation tests
- test_training/: Training pipeline tests
- test_inference/: Inference engine tests
- test_evaluation/: Metrics and evaluation tests
- test_data/: Dataset and preprocessing tests
- test_integration/: End-to-end workflow tests
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunner:
    """Comprehensive test runner for the segmentation framework."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_dir = Path(__file__).parent

    def run_all_tests(self):
        """Run all test suites."""
        print("🧪 Running Surgical Instance Segmentation Framework Tests")
        print("=" * 60)

        # Discover and run all tests
        loader = unittest.TestLoader()
        start_dir = str(self.test_dir)
        suite = loader.discover(start_dir, pattern="test_*.py")

        runner = unittest.TextTestRunner(verbosity=2 if self.verbose else 1)
        result = runner.run(suite)

        # Print summary
        self._print_test_summary(result)
        return result

    def run_model_tests(self):
        """Run model-specific tests."""
        print("🤖 Running Model Tests...")
        return self._run_test_module("test_models")

    def run_training_tests(self):
        """Run training pipeline tests."""
        print("🏋️ Running Training Tests...")
        return self._run_test_module("test_training")

    def run_inference_tests(self):
        """Run inference engine tests."""
        print("🚀 Running Inference Tests...")
        return self._run_test_module("test_inference")

    def run_evaluation_tests(self):
        """Run evaluation and metrics tests."""
        print("📊 Running Evaluation Tests...")
        return self._run_test_module("test_evaluation")

    def run_data_tests(self):
        """Run data handling tests."""
        print("📁 Running Data Tests...")
        return self._run_test_module("test_data")

    def run_integration_tests(self):
        """Run end-to-end integration tests."""
        print("🔗 Running Integration Tests...")
        return self._run_test_module("test_integration")

    def _run_test_module(self, module_name):
        """Run tests from a specific module."""
        try:
            loader = unittest.TestLoader()
            suite = loader.discover(str(self.test_dir), pattern=f"{module_name}.py")
            runner = unittest.TextTestRunner(verbosity=2 if self.verbose else 1)
            return runner.run(suite)
        except Exception as e:
            print(f"❌ Error running {module_name}: {e}")
            return None

    def _print_test_summary(self, result):
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("📋 Test Summary")
        print("=" * 60)
        print(f"Tests Run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")

        if result.failures:
            print("\n❌ Failures:")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print("\n⚠️ Errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

        if result.wasSuccessful():
            print("\n✅ All tests passed successfully!")
        else:
            print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")


def run_all_tests(verbose=True):
    """Convenience function to run all tests."""
    runner = TestRunner(verbose=verbose)
    return runner.run_all_tests()


def create_test_data():
    """Create synthetic test data for testing purposes."""
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    print("📊 Creating synthetic test data...")

    # Create dummy images and annotations for testing
    import json

    import numpy as np
    from PIL import Image

    # Create test images
    for i in range(5):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(test_data_dir / f"test_image_{i:03d}.jpg")

    # Create test COCO annotations
    coco_data = {
        "images": [
            {
                "id": i,
                "file_name": f"test_image_{i:03d}.jpg",
                "width": 640,
                "height": 480,
            }
            for i in range(5)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i % 5,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "area": 2500,
                "iscrowd": 0,
                "segmentation": [[100, 100, 150, 100, 150, 150, 100, 150]],
            }
            for i in range(10)
        ],
        "categories": [
            {"id": 1, "name": "surgical_instrument", "supercategory": "instrument"}
        ],
    }

    with open(test_data_dir / "test_annotations.json", "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ Test data created in {test_data_dir}")


# Framework test utilities
__all__ = ["TestRunner", "run_all_tests", "create_test_data"]

if __name__ == "__main__":
    # Run tests when module is executed directly
    run_all_tests()
