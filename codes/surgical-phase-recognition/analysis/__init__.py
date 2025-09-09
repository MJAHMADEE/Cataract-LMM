#!/usr/bin/env python3
"""
Analysis package for surgical phase recognition system.

This package provides comprehensive analysis tools for surgical phase recognition models,
including model performance analysis, error pattern detection, and visualization capabilities.

Modules:
    model_analyzer: Comprehensive model performance analysis with visualization
    error_analyzer: Detailed error pattern analysis and failure mode detection
"""

from .error_analyzer import ErrorAnalyzer
from .model_analyzer import ModelAnalyzer

__all__ = ["ModelAnalyzer", "ErrorAnalyzer"]

__version__ = "1.0.0"
__author__ = "Surgical Phase Recognition Team"
