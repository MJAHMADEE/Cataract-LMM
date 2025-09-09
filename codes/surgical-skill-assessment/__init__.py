"""
Surgical Skill Assessment Package

A comprehensive video-based surgical skill assessment system using deep learning.
This package provides tools for data processing, model training, evaluation, and inference
for surgical procedure classification tasks.
"""

__version__ = "1.0.0"
__author__ = "Surgical Skill Assessment Team"
__email__ = "your-email@example.com"

# Package level imports for easier access
try:
    from .data.dataset import VideoDataset
    from .data.split import create_splits
    from .models.factory import create_model
    from .utils.helpers import (
        check_gpu_memory,
        print_section,
        seed_everything,
        setup_output_dirs,
    )

    __all__ = [
        "seed_everything",
        "check_gpu_memory",
        "print_section",
        "setup_output_dirs",
        "VideoDataset",
        "create_splits",
        "create_model",
    ]
except ImportError as e:
    # Fallback for testing
    seed_everything = None
    check_gpu_memory = None
    print_section = None
    setup_output_dirs = None
    VideoDataset = None
    create_splits = None
    create_model = None

    __all__ = []
