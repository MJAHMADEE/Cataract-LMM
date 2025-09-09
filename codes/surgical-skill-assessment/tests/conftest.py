"""
Test configuration for surgical skill assessment package.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def device():
    """Provide device for testing."""
    return torch.device("cpu")  # Use CPU for testing


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "data": {
            "clip_len": 16,
            "frame_rate": 5,
            "overlap": 0,
            "num_workers": 0,  # Use 0 workers for testing
            "split_mode": "stratified",
            "global_split_pct": {"train": 70, "val": 15, "test": 15},
        },
        "model": {"model_name": "cnn_lstm", "freeze_backbone": False, "dropout": 0.5},
        "train": {"batch_size": 2, "lr": 1e-4, "seed": 42, "gradient_accumulation": 1},
        "hardware": {"gpus": 0, "mixed_precision": False},
        "logging": {"print_freq": 5, "save_detailed_predictions": True},
        "metrics": [
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "sensitivity",
            "specificity",
            "auc",
        ],
        "override": {"class_names": None},
    }


@pytest.fixture
def sample_video_data():
    """Provide sample video metadata for testing."""
    return [
        {
            "video_path": "/fake/path/SK_0001_S1_P03.mp4",
            "class_idx": 0,
            "class_name": "class_0",
        },
        {
            "video_path": "/fake/path/SK_0002_S2_P03.mp4",
            "class_idx": 1,
            "class_name": "class_1",
        },
    ]
