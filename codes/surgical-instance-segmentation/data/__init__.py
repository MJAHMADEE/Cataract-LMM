"""
Data Processing Module for Surgical Instance Segmentation

This module provides comprehensive data handling, preprocessing, and augmentation
capabilities for surgical instance segmentation datasets.

Components:
- dataset_utils: COCO dataset loading and manipulation
- preprocessing: Image and mask preprocessing pipelines
- augmentation: Medical-safe data augmentation
- validation: Data quality validation and integrity checks

Author: Research Team
Date: August 2025
"""

# Graceful imports with fallbacks for CI/testing environments
try:
    from .augmentation import SurgicalAugmentator
except (ImportError, ValueError, Exception):
    SurgicalAugmentator = None

try:
    from .dataset_utils import SurgicalCocoDataset, create_data_splits
except (ImportError, ValueError, Exception):
    SurgicalCocoDataset = None
    create_data_splits = None

try:
    from .preprocessing import SurgicalImagePreprocessor, get_surgical_transforms
except (ImportError, ValueError, Exception):
    SurgicalImagePreprocessor = None
    get_surgical_transforms = None

try:
    from .validation import DatasetValidator
except (ImportError, ValueError, Exception):
    DatasetValidator = None

__all__ = [
    "SurgicalCocoDataset",
    "create_data_splits",
    "SurgicalImagePreprocessor",
    "get_surgical_transforms",
    "SurgicalAugmentator",
    "DatasetValidator",
]
