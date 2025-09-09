#!/usr/bin/env python3
"""
Data Module for Surgical Phase Recognition

This module provides a unified interface for data loading, preprocessing,
and dataset management for surgical phase recognition.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

from .data_utils import DataManager, create_data_splits_from_directory
from .datasets import (
    FrameLevelDataset,
    SurgicalPhaseDataset,
    collate_fn_surgical_phase,
    create_surgical_phase_dataloaders,
)
from .sequential_dataset import SequentialSurgicalPhaseDatasetAugOverlap

__all__ = [
    "SurgicalPhaseDataset",
    "FrameLevelDataset",
    "create_surgical_phase_dataloaders",
    "collate_fn_surgical_phase",
    "DataManager",
    "create_data_splits_from_directory",
    "SequentialSurgicalPhaseDatasetAugOverlap",
]
