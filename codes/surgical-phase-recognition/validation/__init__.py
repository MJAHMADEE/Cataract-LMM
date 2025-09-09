#!/usr/bin/env python3
"""
Validation package for surgical phase recognition system.

This package provides comprehensive validation tools for evaluating surgical
phase classification models, including the exact validation functionality
used in the reference notebook.

Modules:
    training_framework: PyTorch Lightning training module for phase classification
    comprehensive_validator: Notebook-compatible validation functions
"""

try:
    from .training_framework import PhaseClassificationTraining
except ImportError:
    # PyTorch Lightning not available
    pass
from .comprehensive_validator import (
    DEFAULT_LABEL_TO_IDX,
    DEFAULT_PHASE_NAMES,
    ValidationModule,
    analyze_model_metrics,
    strip_prefix,
    validate_model,
)

__all__ = [
    "PhaseClassificationTraining",
    "ValidationModule",
    "validate_model",
    "analyze_model_metrics",
    "strip_prefix",
    "DEFAULT_PHASE_NAMES",
    "DEFAULT_LABEL_TO_IDX",
]

__version__ = "1.0.0"
__author__ = "Surgical Phase Recognition Team"
