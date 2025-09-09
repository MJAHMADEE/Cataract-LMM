"""
Models Module Initialization

This module provides the initialization for all model architectures
in the surgical instance segmentation framework.
"""

# Import all model implementations
from .mask_rcnn import (
    MASK_RCNN_CONFIG,
    CustomCocoDataset,
    MaskRCNN,
    create_mask_rcnn_model,
)
from .sam import (
    SAM_CONFIG,
    SAMModel,
    create_sam_model,
    evaluate_sam_on_coco,
    mask_to_bbox,
    mask_to_rle,
)
from .yolo import (
    YOLO8_CONFIG,
    YOLO11_CONFIG,
    YOLO_CONFIG,
    YOLO8Model,
    YOLO11Model,
    create_data_config,
    create_yolo8_model,
    create_yolo11_model,
    create_yolo_model,
)

# Aliases for compatibility with documentation
SurgicalMaskRCNN = MaskRCNN
MaskRCNNPredictor = MaskRCNN
MaskRCNNPreprocessor = CustomCocoDataset
MaskRCNNPostprocessor = MaskRCNN

SurgicalSAM = SAMModel
SAMPredictor = SAMModel
SAMPreprocessor = mask_to_rle
SAMPostprocessor = mask_to_bbox

SurgicalYOLOSegmentation = YOLO8Model
YOLOv8Segmentation = YOLO8Model
YOLOv11Segmentation = YOLO11Model
YOLOPredictor = YOLO8Model


def predict_surgical_instruments(*args, **kwargs):
    """Alias for compatibility."""
    return create_mask_rcnn_model(*args, **kwargs)


def predict_with_yolo(*args, **kwargs):
    """Alias for compatibility."""
    return create_yolo8_model(*args, **kwargs)


# Import factory and utilities (with try/except for missing modules)
try:
    from .model_factory import (
        ModelFactory,
        create_segmentation_model,
        create_segmentation_predictor,
    )
except ImportError:
    # Create simple factory functions if model_factory doesn't exist
    def ModelFactory(model_type: str, **kwargs):
        """Simple model factory."""
        if model_type.lower() == "maskrcnn":
            return create_mask_rcnn_model(**kwargs)
        elif model_type.lower() in ["sam"]:
            return create_sam_model(**kwargs)
        elif model_type.lower() in ["yolo8", "yolov8"]:
            return create_yolo8_model(**kwargs)
        elif model_type.lower() in ["yolo11", "yolov11"]:
            return create_yolo11_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_segmentation_model(model_type: str, **kwargs):
        """Create segmentation model."""
        return ModelFactory(model_type, **kwargs)

    def create_segmentation_predictor(model_type: str, **kwargs):
        """Create segmentation predictor."""
        return ModelFactory(model_type, **kwargs)


# Model configuration mapping
MODEL_CONFIGS = {
    "mask_rcnn": MASK_RCNN_CONFIG,
    "sam": SAM_CONFIG,
    "yolo8": YOLO8_CONFIG,
    "yolo11": YOLO11_CONFIG,
    "yolo": YOLO_CONFIG,
}

# Define public API
__all__ = [
    # Mask R-CNN
    "SurgicalMaskRCNN",
    "MaskRCNNPredictor",
    "MaskRCNNPreprocessor",
    "MaskRCNNPostprocessor",
    "create_mask_rcnn_model",
    "predict_surgical_instruments",
    "MASK_RCNN_CONFIG",
    # SAM
    "SurgicalSAM",
    "SAMPredictor",
    "SAMPreprocessor",
    "SAMPostprocessor",
    "create_sam_model",
    "evaluate_sam_on_coco",
    "SAM_CONFIG",
    # YOLO
    "SurgicalYOLOSegmentation",
    "YOLOv8Segmentation",
    "YOLOv11Segmentation",
    "YOLOPredictor",
    "create_yolo_model",
    "predict_with_yolo",
    "YOLO_CONFIG",
    # Factory
    "ModelFactory",
    "create_segmentation_model",
    "create_segmentation_predictor",
    "get_model_recommendations",
    "MODEL_SUMMARY",
]

# Model registry for easy access
MODEL_REGISTRY = {
    "mask_rcnn": {
        "model_class": SurgicalMaskRCNN,
        "predictor_class": MaskRCNNPredictor,
        "factory_function": create_mask_rcnn_model,
        "config": MASK_RCNN_CONFIG,
    },
    "sam": {
        "model_class": SurgicalSAM,
        "predictor_class": SAMPredictor,
        "factory_function": create_sam_model,
        "config": SAM_CONFIG,
    },
    "yolov8": {
        "model_class": YOLOv8Segmentation,
        "predictor_class": YOLOPredictor,
        "factory_function": lambda **kwargs: create_yolo_model(
            version="yolov8", **kwargs
        ),
        "config": YOLO_CONFIG,
    },
    "yolov11": {
        "model_class": YOLOv11Segmentation,
        "predictor_class": YOLOPredictor,
        "factory_function": lambda **kwargs: create_yolo_model(
            version="yolov11", **kwargs
        ),
        "config": YOLO_CONFIG,
    },
}

# Version and framework information
__version__ = "1.0.0"
__framework__ = "Surgical Instance Segmentation"
__supported_models__ = list(MODEL_REGISTRY.keys())
__description__ = """
Comprehensive surgical instance segmentation framework supporting multiple
state-of-the-art architectures including Mask R-CNN, SAM, and YOLO models.
"""
