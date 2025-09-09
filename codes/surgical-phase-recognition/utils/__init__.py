#!/usr/bin/env python3
"""
Utilities package for surgical phase recognition system.

This package provides common utilities, helper functions, and convenience tools
used across the surgical phase recognition framework.

Modules:
    helpers: General utility functions and helper classes
"""

from .helpers import (
    ConfigValidator,
    MetricsTracker,
    Timer,
    calculate_file_hash,
    create_video_summary_stats,
    ensure_dir,
    format_bytes,
    format_time,
    get_gpu_memory_info,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
    setup_logging,
    validate_phase_annotations,
)

__all__ = [
    "setup_logging",
    "ensure_dir",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "calculate_file_hash",
    "get_gpu_memory_info",
    "format_time",
    "format_bytes",
    "Timer",
    "ConfigValidator",
    "MetricsTracker",
    "create_video_summary_stats",
    "validate_phase_annotations",
]

__version__ = "1.0.0"
__author__ = "Surgical Phase Recognition Team"
