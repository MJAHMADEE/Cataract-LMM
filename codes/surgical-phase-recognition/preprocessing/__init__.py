#!/usr/bin/env python3
"""
Preprocessing Module for Surgical Phase Recognition

This module provides video preprocessing functionality including frame extraction,
augmentation, temporal sampling, format conversion, and data transforms.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

from .video_preprocessing import (
    TemporalSampler,
    VideoAugmentation,
    VideoFrameExtractor,
    VideoPreprocessor,
    save_frames_to_disk,
)

try:
    from .advanced_transforms import SequenceTransform as AdvancedTransforms
    from .advanced_transforms import (
        SurgicalVideoTransform as CataractSpecificTransforms,
    )
except ImportError:
    # Fallback for missing advanced transforms
    AdvancedTransforms = None
    CataractSpecificTransforms = None

# Create aliases for compatibility
SurgicalDataAugmentation = AdvancedTransforms
DataAugmentationPipeline = CataractSpecificTransforms

import os

# Import transforms from the root transform module for compatibility
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from transform import SequenceTransform as SurgicalVideoTransform


def get_training_transforms(**kwargs):
    """Get training transforms."""
    return SurgicalVideoTransform(is_training=True, **kwargs)


def get_validation_transforms(**kwargs):
    """Get validation transforms."""
    return SurgicalVideoTransform(is_training=False, **kwargs)


def get_surgical_transforms(is_training=True, **kwargs):
    """Get surgical-specific transforms."""
    return SurgicalVideoTransform(is_training=is_training, **kwargs)


__all__ = [
    # Video preprocessing
    "VideoFrameExtractor",
    "TemporalSampler",
    "VideoPreprocessor",
    "save_frames_to_disk",
    # Transforms
    "SurgicalVideoTransform",
    "get_training_transforms",
    "get_validation_transforms",
    "get_surgical_transforms",
    # Advanced transforms
    "AdvancedTransforms",
    "SurgicalDataAugmentation",
    "DataAugmentationPipeline",
    "CataractSpecificTransforms",
]
