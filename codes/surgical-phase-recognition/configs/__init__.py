#!/usr/bin/env python3
"""
Configuration Module for Surgical Phase Recognition

This module provides configuration management utilities and default configurations
for different training scenarios.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

from .config_manager import (
    Config,
    ConfigManager,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    load_config_from_args,
)

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "ConfigManager",
    "load_config_from_args",
]
