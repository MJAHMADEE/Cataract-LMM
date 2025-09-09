"""
Surgical Phase Recognition Framework

This package provides deep learning models and tools for automated surgical phase
recognition in cataract surgery videos, implementing the methodologies described
in the Cataract-LMM research paper.

Key Features:
- 13-phase surgical taxonomy classification
- Multi-center domain adaptation (Farabi S1 ↔ Noor S2)
- Hybrid CNN-RNN and pure 3D-CNN architectures
- Video transformer models (MViT, Swin, etc.)
- Comprehensive evaluation metrics

Models supported:
- ResNet + LSTM/GRU/TeCNO
- SlowFast, X3D, R(2+1)D, MC3, R3D
- MViT, Video Swin Transformer

Author: Cataract-LMM Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Cataract-LMM Team"

# Import main components
try:
    from .data import SequentialDataset, create_data_loaders
    from .models import create_model, get_model_parameters
    from .validation import TrainingFramework, evaluate_model

    __all__ = [
        "create_model",
        "get_model_parameters",
        "create_data_loaders",
        "SequentialDataset",
        "TrainingFramework",
        "evaluate_model",
    ]
except ImportError as e:
    # Fallback for testing
    create_model = None
    get_model_parameters = None
    create_data_loaders = None
    SequentialDataset = None
    TrainingFramework = None
    evaluate_model = None

    __all__ = []
