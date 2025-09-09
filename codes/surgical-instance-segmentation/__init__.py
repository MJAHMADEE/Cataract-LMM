"""
Surgical Instance Segmentation Framework

A comprehensive, production-ready framework for surgical instance segmentation
in cataract surgery videos using state-of-the-art deep learning models.

This package provides:
- Mask R-CNN for instance segmentation
- SAM (Segment Anything Model) for prompt-based segmentation
- YOLO models (v8/v11) for real-time segmentation
- Unified training, inference, and evaluation pipelines
- Production deployment utilities

Core Modules:
- models: Model implementations and architecture definitions
- training: Training pipelines and experiment management
- inference: Real-time and batch inference engines
- evaluation: Comprehensive evaluation metrics and protocols
- data: Dataset utilities and preprocessing pipelines
- utils: Visualization, configuration, and utility functions

Example Usage:
    >>> from surgical_instance_segmentation import ModelFactory, TrainingManager
    >>> from surgical_instance_segmentation.inference import InferenceEngine

    # Create and train a Mask R-CNN model
    >>> factory = ModelFactory()
    >>> model = factory.create_mask_rcnn_model(num_classes=13)
    >>> manager = TrainingManager()
    >>> manager.train_mask_rcnn(data_root="/path/to/dataset")

    # Run inference
    >>> engine = InferenceEngine()
    >>> engine.load_mask_rcnn_model("/path/to/trained/model.pth")
    >>> predictions = engine.predict_mask_rcnn("/path/to/image.jpg")

Reference Notebooks:
- mask_rcnn_training.ipynb: Complete Mask R-CNN training workflow
- sam_inference.ipynb: SAM-based segmentation with bbox prompts
- yolov8_segmentation_training.ipynb: YOLOv8 training pipeline
- yolov11_segmentation_training.ipynb: YOLOv11 training pipeline

Authors: Surgical AI Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Surgical AI Research Team"
__email__ = "surgical.ai@research.com"
__license__ = "MIT"

# Core imports for convenient access
try:
    from .data.dataset_utils import SurgicalCocoDataset
    from .evaluation import SegmentationMetrics
    from .inference import InferenceEngine
    from .models.model_factory import ModelFactory
    from .training.training_manager import TrainingManager
    from .utils.config_manager import ConfigManager

    __all__ = [
        "ModelFactory",
        "TrainingManager",
        "InferenceEngine",
        "SegmentationMetrics",
        "ConfigManager",
        "SurgicalCocoDataset",
        "__version__",
        "version_info",
    ]
except (ImportError, ValueError, Exception) as e:
    # Fallback for testing or when dependencies aren't available
    ModelFactory = None
    TrainingManager = None
    InferenceEngine = None
    SegmentationMetrics = None
    ConfigManager = None
    SurgicalCocoDataset = None

    __all__ = ["__version__", "version_info"]

# Version info
version_info = (1, 0, 0)

# Framework configuration
DEFAULT_CONFIG = {
    "device": "auto",
    "precision": "float32",
    "backend": "pytorch",
    "log_level": "INFO",
}


def get_version():
    """Get the current version of the framework."""
    return __version__


def print_info():
    """Print framework information."""
    print(f"Surgical Instance Segmentation Framework v{__version__}")
    print(f"Authors: {__author__}")
    print(f"License: {__license__}")
    print("\nSupported Models:")
    print("  - Mask R-CNN (Instance Segmentation)")
    print("  - SAM (Segment Anything Model)")
    print("  - YOLO v8/v11 (Real-time Segmentation)")
    print("\nFor documentation and examples, see the notebooks/ directory.")


# Convenience function for quick setup
def quick_setup(model_type="mask_rcnn", device="auto"):
    """
    Quick setup for common use cases.

    Args:
        model_type (str): Type of model ('mask_rcnn', 'yolo', 'sam')
        device (str): Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Tuple of (model_factory, inference_engine) ready for use
    """
    if ModelFactory is None or InferenceEngine is None:
        raise ImportError(
            "Required dependencies not available. Cannot perform quick setup."
        )

    factory = ModelFactory(device=device)
    engine = InferenceEngine(device=device)

    print(f"✅ Framework initialized for {model_type} on {device}")
    print("📚 Check notebooks/ for complete training and inference examples")

    return factory, engine


# Validate imports on module load
def _validate_dependencies():
    """Validate that required dependencies are available."""
    missing_deps = []

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        import torchvision
    except ImportError:
        missing_deps.append("torchvision")

    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")

    if missing_deps:
        print(f"⚠️  Warning: Missing optional dependencies: {', '.join(missing_deps)}")
        print("   Some functionality may be limited. Install with:")
        print(f"   pip install {' '.join(missing_deps)}")


# Run validation only if main imports were successful
if ModelFactory is not None:
    try:
        _validate_dependencies()
    except Exception as e:
        # Silently handle any dependency validation errors
        pass
