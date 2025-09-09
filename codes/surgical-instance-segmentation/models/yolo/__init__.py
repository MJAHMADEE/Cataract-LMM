"""
YOLO Models Implementation for Surgical Instance Segmentation

This module provides complete YOLO implementations exactly matching the reference notebooks:
- /workspaces/Cataract_LMM/codes/Segmentation/train_yolo8.ipynb
- /workspaces/Cataract_LMM/codes/Segmentation/train_yolo11.ipynb

Key Components:
- YOLO8Model for YOLOv8l-seg training and inference
- YOLO11Model for YOLOv11l-seg training and inference
- Training configuration exactly matching notebooks
- Inference pipeline for surgical instance segmentation
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import yaml

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. YOLO functionality will be limited.")
    ULTRALYTICS_AVAILABLE = False
    YOLO = None


class YOLO8Model:
    """YOLOv8 model implementation exactly matching train_yolo8.ipynb."""

    def __init__(self, model_name: str = "yolov8l-seg.pt", device: str = "cuda"):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package not installed")

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def train(
        self,
        data_config: Union[str, Dict],
        epochs: int = 80,
        imgsz: int = 640,
        batch: int = 20,
        **kwargs,
    ) -> Any:
        """Train YOLOv8 exactly as in the reference notebook."""
        training_args = {
            "data": data_config,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": self.device,
            **kwargs,
        }
        return self.model.train(**training_args)

    def predict(self, source: Union[str, List], **kwargs) -> Any:
        """Run inference using the trained YOLOv8 model."""
        return self.model.predict(source, **kwargs)

    def val(self, **kwargs) -> Any:
        """Validate the YOLOv8 model."""
        return self.model.val(**kwargs)

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the YOLOv8 model."""
        return self.model.export(format=format, **kwargs)


class YOLO11Model:
    """YOLOv11 model implementation exactly matching train_yolo11.ipynb."""

    def __init__(self, model_name: str = "yolo11l-seg.pt", device: str = "cuda"):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package not installed")

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.model.to(self.device)

    def train(
        self,
        data_config: Union[str, Dict],
        epochs: int = 80,
        imgsz: int = 640,
        batch: int = 20,
        **kwargs,
    ) -> Any:
        """Train YOLOv11 exactly as in the reference notebook."""
        training_args = {
            "data": data_config,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": self.device,
            **kwargs,
        }
        return self.model.train(**training_args)

    def predict(self, source: Union[str, List], **kwargs) -> Any:
        """Run inference using the trained YOLOv11 model."""
        return self.model.predict(source, **kwargs)

    def val(self, **kwargs) -> Any:
        """Validate the YOLOv11 model."""
        return self.model.val(**kwargs)

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the YOLOv11 model."""
        return self.model.export(format=format, **kwargs)


def create_yolo8_model(
    model_name: str = "yolov8l-seg.pt", device: str = "cuda"
) -> YOLO8Model:
    """Create YOLOv8 model exactly as in the reference notebook."""
    return YOLO8Model(model_name=model_name, device=device)


def create_yolo11_model(
    model_name: str = "yolo11l-seg.pt", device: str = "cuda"
) -> YOLO11Model:
    """Create YOLOv11 model exactly as in the reference notebook."""
    return YOLO11Model(model_name=model_name, device=device)


def create_data_config(
    train_path: str, val_path: str, nc: int = 13, names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create YOLO data configuration exactly as in the notebooks."""
    if names is None:
        names = [
            "Bipolar forceps",
            "Charlieux cannula",
            "Hydrodissection cannula",
            "I/A handpiece",
            "Micromanipulator",
            "Needle holder",
            "Primary knife",
            "Rycroft cannula",
            "Secondary knife",
            "Suturing needle",
            "Vannas scissors",
            "Lens injector",
            "Phacoemulsifier handpiece",
        ]

    return {"train": train_path, "val": val_path, "nc": nc, "names": names}


# Configuration exactly matching the notebooks
YOLO8_CONFIG = {
    "model_name": "yolov8l-seg.pt",
    "epochs": 80,
    "imgsz": 640,
    "batch": 20,
    "device": "cuda",
}

YOLO11_CONFIG = {
    "model_name": "yolo11l-seg.pt",
    "epochs": 80,
    "imgsz": 640,
    "batch": 20,
    "device": "cuda",
}

# Aliases for compatibility
SurgicalYOLOSegmentation = YOLO8Model  # Default to YOLO8
YOLOv8Segmentation = YOLO8Model
YOLOv11Segmentation = YOLO11Model
YOLOPredictor = YOLO8Model  # Default to YOLO8
YOLO_CONFIG = YOLO8_CONFIG  # Default config


def create_yolo_model(
    version: str = "yolo8", **kwargs
) -> Union[YOLO8Model, YOLO11Model]:
    """Create YOLO model based on version."""
    if version.lower() in ["yolo8", "yolov8", "8"]:
        return create_yolo8_model(**kwargs)
    elif version.lower() in ["yolo11", "yolov11", "11"]:
        return create_yolo11_model(**kwargs)
    else:
        raise ValueError(f"Unknown YOLO version: {version}")


def predict_with_yolo(
    model: Union[YOLO8Model, YOLO11Model], source: Union[str, List], **kwargs
) -> Any:
    """Run prediction with YOLO model."""
    return model.predict(source, **kwargs)


__all__ = [
    "YOLO8Model",
    "YOLO11Model",
    "create_yolo8_model",
    "create_yolo11_model",
    "create_data_config",
    "SurgicalYOLOSegmentation",
    "YOLOv8Segmentation",
    "YOLOv11Segmentation",
    "YOLOPredictor",
    "create_yolo_model",
    "predict_with_yolo",
    "YOLO8_CONFIG",
    "YOLO11_CONFIG",
    "YOLO_CONFIG",
]
