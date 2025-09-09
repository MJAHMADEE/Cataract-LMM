#!/usr/bin/env python3
"""
Tests package for surgical phase recognition system.

This package provides comprehensive unit tests for all components of the
surgical phase recognition framework.

Modules:
    test_all_components: Comprehensive test suite for all system components
"""


# Test runner convenience function
def run_all_tests():
    """
    Run all tests in the test suite.

    This function provides a convenient way to run all tests programmatically.
    For more control, use pytest directly from the command line.
    """
    import sys
    from pathlib import Path

    import pytest

    # Get the directory containing this file
    test_dir = Path(__file__).parent

    # Run pytest on the test directory
    exit_code = pytest.main([str(test_dir), "-v", "--tb=short", "--disable-warnings"])

    return exit_code


# Test configuration
TEST_CONFIG = {
    "test_data_size": 100,
    "test_batch_size": 4,
    "test_sequence_length": 8,
    "test_num_classes": 11,
    "test_image_size": (224, 224),
    "test_video_frames": 16,
}

__all__ = ["run_all_tests", "TEST_CONFIG"]

__version__ = "1.0.0"
__author__ = "Surgical Phase Recognition Team"
