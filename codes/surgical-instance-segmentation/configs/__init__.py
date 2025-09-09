"""
Configuration Management for Surgical Instance Segmentation Framework

This module provides centralized configuration management for all framework components.

Configuration Categories:
- Model Configs: Architecture and hyperparameter settings
- Training Configs: Training pipeline configurations
- Data Configs: Dataset and preprocessing settings
- Inference Configs: Inference pipeline settings
- Evaluation Configs: Metrics and validation settings

Example Usage:
    >>> from surgical_instance_segmentation.configs import load_config
    >>> from surgical_instance_segmentation.configs import ModelConfig

    # Load configuration from file
    >>> config = load_config('configs/yolo_config.yaml')

    # Create model configuration
    >>> model_config = ModelConfig(
    ...     model_type='yolo11',
    ...     num_classes=1,
    ...     input_size=(640, 640)
    ... )

Configuration Files:
- mask_rcnn_config.yaml: Mask R-CNN configuration
- sam_config.yaml: SAM model configuration
- yolo_config.yaml: YOLO configuration
- data_config.yaml: Dataset configuration
- training_config.yaml: Training pipeline configuration
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Configuration directory
CONFIG_DIR = Path(__file__).parent


@dataclass
class ModelConfig:
    """Base configuration class for all models."""

    model_type: str
    num_classes: int = 1
    input_size: tuple = (640, 640)
    device: str = "cuda"
    weights_path: Optional[str] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "StepLR"
    scheduler_params: Dict[str, Any] = None
    save_frequency: int = 10
    validation_frequency: int = 5
    early_stopping_patience: int = 10
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0

    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {"step_size": 30, "gamma": 0.1}


@dataclass
class DataConfig:
    """Data configuration."""

    train_images_path: str
    train_annotations_path: str
    val_images_path: str
    val_annotations_path: str
    test_images_path: Optional[str] = None
    test_annotations_path: Optional[str] = None
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")
    annotation_format: str = "coco"  # coco, yolo, pascal_voc
    augmentation_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.augmentation_config is None:
            self.augmentation_config = {
                "horizontal_flip": 0.5,
                "vertical_flip": 0.0,
                "rotation": 0.1,
                "brightness": 0.1,
                "contrast": 0.1,
            }


@dataclass
class InferenceConfig:
    """Inference configuration."""

    batch_size: int = 1
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    visualization: bool = True
    save_results: bool = True
    output_format: str = "coco"  # coco, yolo, masks
    device: str = "cuda"


class ConfigManager:
    """Configuration manager for loading and saving configurations."""

    def __init__(self, config_dir: Union[str, Path] = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_path)

        if not config_path.is_absolute():
            config_path = self.config_dir / config_path

        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.safe_dump(config, f, indent=2, default_flow_style=False)
            elif config_path.suffix.lower() == ".json":
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get model configuration by type."""
        config_file = f"{model_type}_config.yaml"
        try:
            config_dict = self.load_config(config_file)
            return ModelConfig.from_dict(config_dict.get("model", {}))
        except FileNotFoundError:
            # Return default configuration
            return ModelConfig(model_type=model_type)

    def get_training_config(
        self, config_name: str = "training_config.yaml"
    ) -> TrainingConfig:
        """Get training configuration."""
        try:
            config_dict = self.load_config(config_name)
            return TrainingConfig(**config_dict.get("training", {}))
        except FileNotFoundError:
            return TrainingConfig()

    def get_data_config(self, config_name: str = "data_config.yaml") -> DataConfig:
        """Get data configuration."""
        config_dict = self.load_config(config_name)
        return DataConfig(**config_dict.get("data", {}))

    def get_inference_config(
        self, config_name: str = "inference_config.yaml"
    ) -> InferenceConfig:
        """Get inference configuration."""
        try:
            config_dict = self.load_config(config_name)
            return InferenceConfig(**config_dict.get("inference", {}))
        except FileNotFoundError:
            return InferenceConfig()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to load configuration."""
    manager = ConfigManager()
    return manager.load_config(config_path)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """Convenience function to save configuration."""
    manager = ConfigManager()
    return manager.save_config(config, config_path)


def create_default_configs():
    """Create default configuration files."""
    manager = ConfigManager()

    # Default YOLO configuration
    yolo_config = {
        "model": {
            "model_type": "yolo11",
            "num_classes": 1,
            "input_size": [640, 640],
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
        },
        "training": {
            "epochs": 80,
            "batch_size": 20,
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "optimizer": "AdamW",
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": 0.1},
        },
    }

    # Default Mask R-CNN configuration
    mask_rcnn_config = {
        "model": {
            "model_type": "mask_rcnn",
            "num_classes": 1,
            "backbone": "resnet50",
            "confidence_threshold": 0.5,
        },
        "training": {
            "epochs": 100,
            "batch_size": 4,
            "learning_rate": 0.005,
            "weight_decay": 0.0005,
            "optimizer": "AdamW",
            "scheduler": "StepLR",
            "scheduler_params": {"step_size": 30, "gamma": 0.1},
        },
    }

    # Default SAM configuration
    sam_config = {
        "model": {
            "model_type": "sam",
            "model_size": "vit_h",
            "checkpoint_path": "sam_vit_h_4b8939.pth",
        },
        "inference": {
            "confidence_threshold": 0.8,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
        },
    }

    # Save configurations
    manager.save_config(yolo_config, "yolo_config.yaml")
    manager.save_config(mask_rcnn_config, "mask_rcnn_config.yaml")
    manager.save_config(sam_config, "sam_config.yaml")

    print("✅ Default configuration files created")


# Configuration utilities
__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "InferenceConfig",
    "ConfigManager",
    "load_config",
    "save_config",
    "create_default_configs",
]

if __name__ == "__main__":
    # Create default configurations when module is executed directly
    create_default_configs()
