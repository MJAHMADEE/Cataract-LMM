"""
Configuration Management for Surgical Instance Segmentation Framework

This module provides comprehensive configuration management for all components:
- Model configurations (Mask R-CNN, YOLO, SAM)
- Training configurations (hyperparameters, schedules, optimization)
- Dataset configurations (paths, splits, transformations)
- Inference configurations (thresholds, post-processing)
- Evaluation configurations (metrics, protocols)

Supports YAML, JSON, and Python configuration files with validation and defaults.
"""

import json
import logging
import os
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml

    yaml_available = True
except ImportError:
    print("Warning: PyYAML not installed. YAML configuration support will be limited.")
    yaml_available = False


@dataclass
class ModelConfig:
    """Base configuration for all models"""

    name: str = "base_model"
    type: str = "unknown"
    device: str = "auto"
    num_classes: int = 91
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class MaskRCNNConfig(ModelConfig):
    """Configuration for Mask R-CNN model - matches maskRCNN.ipynb exactly"""

    type: str = "mask_rcnn"
    name: str = "mask_rcnn_model"

    # Architecture parameters (same as notebook)
    backbone: str = "resnet50"
    fpn: bool = True

    # Training parameters (exact same as notebook)
    batch_size: int = 4  # Same as notebook
    learning_rate: float = 0.0005  # Same as notebook (AdamW lr)
    num_epochs: int = 10  # Same as notebook
    optimizer: str = "adamw"  # Same as notebook
    scheduler: str = "steplr"  # Same as notebook
    step_size: int = 3  # Same as notebook (StepLR step_size)
    gamma: float = 0.1  # Same as notebook (StepLR gamma)

    # Loss weights
    bbox_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    classification_loss_weight: float = 1.0

    # Anchor parameters
    anchor_sizes: List[List[int]] = field(
        default_factory=lambda: [[32], [64], [128], [256], [512]]
    )
    aspect_ratios: List[List[float]] = field(
        default_factory=lambda: [[0.5, 1.0, 2.0]] * 5
    )

    # RPN parameters
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.7

    # ROI parameters
    box_detections_per_img: int = 100
    box_nms_thresh: float = 0.5
    box_score_thresh: float = 0.05


@dataclass
class YOLOConfig(ModelConfig):
    """Configuration for YOLO model - matches train_yolo8.ipynb and train_yolo11.ipynb exactly"""

    type: str = "yolo"
    name: str = "yolo_model"

    # Model variants (exact same as notebooks)
    model_name: str = "yolo11l-seg"  # Same as train_yolo11.ipynb

    # Training parameters (exact same as notebooks)
    epochs: int = 80  # Same as notebooks
    imgsz: int = 640  # Same as notebooks
    batch: int = 20  # Same as notebooks
    device: Union[int, str] = 0  # Same as notebooks (GPU)
    plots: bool = True  # Same as notebooks
    resume: bool = True  # Same as notebooks

    # Data configuration
    data_yaml: str = "./data.yaml"  # Same as notebooks

    # Optimization parameters
    optimizer: str = "auto"
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation parameters
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0


@dataclass
class SAMConfig(ModelConfig):
    """Configuration for SAM model - matches SAM_inference.ipynb exactly"""

    type: str = "sam"
    name: str = "sam_model"

    # Model parameters (exact same as notebook)
    model_type: str = "vit_h"  # Same as notebook
    checkpoint_path: str = ""  # Path to SAM checkpoint

    # Inference parameters
    multimask_output: bool = True  # Same as notebook

    # Automatic mask generator parameters
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0


@dataclass
class DatasetConfig:
    """Configuration for dataset handling"""

    # Dataset paths
    root_dir: str = ""
    train_images_dir: str = "train/images"
    val_images_dir: str = "val/images"
    test_images_dir: str = "test/images"
    train_annotations: str = "train/annotations.json"
    val_annotations: str = "val/annotations.json"
    test_annotations: str = "test/annotations.json"

    # Dataset format
    format: str = "coco"  # "coco", "yolo", "pascal_voc"

    # Class information
    num_classes: int = 91
    class_names: List[str] = field(default_factory=list)

    # Data loading
    batch_size: int = 4
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True

    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Transformations
    image_size: int = 640
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation parameters
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    rotation_degrees: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training process"""

    # Basic training parameters
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.0005

    # Optimizer configuration
    optimizer: str = "adamw"
    weight_decay: float = 0.0001
    momentum: float = 0.9

    # Scheduler configuration
    scheduler: str = "steplr"
    step_size: int = 3
    gamma: float = 0.1
    warmup_epochs: int = 0

    # Loss configuration
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {"bbox": 1.0, "mask": 1.0, "classification": 1.0}
    )

    # Training behavior
    save_best: bool = True
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0

    # Validation
    validate_every_n_epochs: int = 1
    validation_metric: str = "ap"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # Logging
    log_every_n_steps: int = 100
    log_dir: str = "./logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "surgical_segmentation"


@dataclass
class InferenceConfig:
    """Configuration for inference"""

    # Confidence thresholds
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5

    # Input processing
    image_size: int = 640
    normalize: bool = True

    # Batch processing
    batch_size: int = 1

    # Post-processing
    max_detections: int = 100

    # Output format
    output_format: str = "coco"  # "coco", "yolo", "json"
    save_predictions: bool = True
    save_visualizations: bool = False

    # Performance optimization
    use_fp16: bool = True
    use_tensorrt: bool = False
    use_onnx: bool = False


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    # Metrics to compute
    compute_coco_metrics: bool = True
    compute_detection_metrics: bool = True
    compute_segmentation_metrics: bool = True

    # COCO evaluation parameters
    iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    max_detections: List[int] = field(default_factory=lambda: [1, 10, 100])
    area_ranges: List[List[float]] = field(
        default_factory=lambda: [
            [0, 10000000000.0],
            [0, 1024],
            [1024, 9216],
            [9216, 10000000000.0],
        ]
    )

    # Custom evaluation parameters
    confidence_thresholds: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.7, 0.9]
    )

    # Output configuration
    save_per_image_results: bool = True
    save_confusion_matrix: bool = True
    generate_visualizations: bool = True
    output_dir: str = "./evaluation_results"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""

    # Experiment metadata
    name: str = "surgical_segmentation_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Output configuration
    output_dir: str = "./experiment_outputs"
    save_config: bool = True


class ConfigManager:
    """
    Configuration manager for the surgical segmentation framework

    This class handles loading, saving, validation, and merging of configurations
    from various sources (YAML, JSON, Python files).
    """

    def __init__(self):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)

        # Default configurations
        self.default_configs = {
            "mask_rcnn": MaskRCNNConfig(),
            "yolo": YOLOConfig(),
            "sam": SAMConfig(),
            "dataset": DatasetConfig(),
            "training": TrainingConfig(),
            "inference": InferenceConfig(),
            "evaluation": EvaluationConfig(),
            "experiment": ExperimentConfig(),
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file

        Args:
            config_path (str): Path to configuration file

        Returns:
            Dict containing configuration
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine file format
        suffix = config_path.suffix.lower()

        if suffix == ".json":
            return self._load_json_config(config_path)
        elif suffix in [".yaml", ".yml"]:
            return self._load_yaml_config(config_path)
        elif suffix == ".py":
            return self._load_python_config(config_path)
        else:
            raise ValueError(f"Unsupported configuration format: {suffix}")

    def save_config(
        self, config: Dict[str, Any], save_path: str, format: str = "auto"
    ) -> None:
        """
        Save configuration to file

        Args:
            config: Configuration dictionary
            save_path (str): Path to save configuration
            format (str): Output format ("json", "yaml", "auto")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format
        if format == "auto":
            suffix = save_path.suffix.lower()
            if suffix == ".json":
                format = "json"
            elif suffix in [".yaml", ".yml"]:
                format = "yaml"
            else:
                format = "json"  # Default to JSON

        # Convert dataclasses to dictionaries
        serializable_config = self._make_serializable(config)

        if format == "json":
            self._save_json_config(serializable_config, save_path)
        elif format == "yaml":
            self._save_yaml_config(serializable_config, save_path)
        else:
            raise ValueError(f"Unsupported save format: {format}")

    def create_notebook_compatible_config(self, model_type: str) -> Dict[str, Any]:
        """
        Create configuration that exactly matches the reference notebooks

        Args:
            model_type (str): Type of model ("mask_rcnn", "yolo8", "yolo11", "sam")

        Returns:
            Dict containing notebook-compatible configuration
        """
        if model_type == "mask_rcnn":
            # Configuration exactly matching maskRCNN.ipynb
            config = {
                "model": MaskRCNNConfig(
                    batch_size=4,  # Same as notebook
                    learning_rate=0.0005,  # Same as notebook
                    num_epochs=10,  # Same as notebook
                    optimizer="adamw",  # Same as notebook
                    scheduler="steplr",  # Same as notebook
                    step_size=3,  # Same as notebook
                    gamma=0.1,  # Same as notebook
                ),
                "training": TrainingConfig(
                    batch_size=4,
                    learning_rate=0.0005,
                    num_epochs=10,
                    optimizer="adamw",
                    scheduler="steplr",
                    step_size=3,
                    gamma=0.1,
                ),
                "dataset": DatasetConfig(format="coco", batch_size=4),
            }

        elif model_type in ["yolo8", "yolo11"]:
            # Configuration exactly matching train_yolo8.ipynb and train_yolo11.ipynb
            model_name = "yolo8l-seg" if model_type == "yolo8" else "yolo11l-seg"

            config = {
                "model": YOLOConfig(
                    model_name=model_name,  # Exact same as notebooks
                    epochs=80,  # Same as notebooks
                    imgsz=640,  # Same as notebooks
                    batch=20,  # Same as notebooks
                    device=0,  # Same as notebooks
                    plots=True,  # Same as notebooks
                    resume=True,  # Same as notebooks
                    data_yaml="./data.yaml",  # Same as notebooks
                ),
                "dataset": DatasetConfig(format="yolo", batch_size=20),
            }

        elif model_type == "sam":
            # Configuration exactly matching SAM_inference.ipynb
            config = {
                "model": SAMConfig(
                    model_type="vit_h",  # Same as notebook
                    multimask_output=True,  # Same as notebook
                ),
                "inference": InferenceConfig(confidence_threshold=0.5, batch_size=1),
                "evaluation": EvaluationConfig(
                    compute_coco_metrics=True  # Same evaluation as notebook
                ),
            }

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return config

    def merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge configurations with override taking precedence

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        merged = deepcopy(base_config)

        for key, value in override_config.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def validate_config(
        self, config: Dict[str, Any], config_type: str = "experiment"
    ) -> bool:
        """
        Validate configuration against schema

        Args:
            config: Configuration to validate
            config_type (str): Type of configuration to validate against

        Returns:
            bool: True if valid, raises exception if invalid
        """
        # Basic validation - check required fields
        if config_type == "experiment":
            required_sections = ["model", "dataset", "training"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(
                        f"Missing required configuration section: {section}"
                    )

        # Model-specific validation
        if "model" in config:
            model_config = config["model"]
            if "type" in model_config:
                model_type = model_config["type"]
                if model_type == "mask_rcnn":
                    self._validate_mask_rcnn_config(model_config)
                elif model_type == "yolo":
                    self._validate_yolo_config(model_config)
                elif model_type == "sam":
                    self._validate_sam_config(model_config)

        return True

    def get_default_config(self, config_type: str) -> Any:
        """
        Get default configuration for specified type

        Args:
            config_type (str): Type of configuration

        Returns:
            Default configuration instance
        """
        if config_type in self.default_configs:
            return deepcopy(self.default_configs[config_type])
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")

    def _load_json_config(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file"""
        with open(config_path, "r") as f:
            return json.load(f)

    def _load_yaml_config(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not yaml_available:
            raise ImportError("PyYAML is required for YAML configuration files")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_python_config(self, config_path: Path) -> Dict[str, Any]:
        """Load Python configuration file"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Extract configuration from module
        config = {}
        for attr_name in dir(config_module):
            if not attr_name.startswith("_"):
                attr = getattr(config_module, attr_name)
                if hasattr(attr, "__dict__") and hasattr(attr, "__dataclass_fields__"):
                    # It's a dataclass
                    config[attr_name] = attr
                elif isinstance(attr, dict):
                    config[attr_name] = attr

        return config

    def _save_json_config(self, config: Dict[str, Any], save_path: Path) -> None:
        """Save configuration as JSON"""
        with open(save_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def _save_yaml_config(self, config: Dict[str, Any], save_path: Path) -> None:
        """Save configuration as YAML"""
        if not yaml_available:
            raise ImportError("PyYAML is required for YAML configuration files")

        with open(save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    def _make_serializable(self, config: Any) -> Any:
        """Convert configuration to serializable format"""
        if hasattr(config, "__dict__") and hasattr(config, "__dataclass_fields__"):
            # It's a dataclass
            return asdict(config)
        elif isinstance(config, dict):
            return {
                key: self._make_serializable(value) for key, value in config.items()
            }
        elif isinstance(config, (list, tuple)):
            return [self._make_serializable(item) for item in config]
        else:
            return config

    def _validate_mask_rcnn_config(self, config: Dict[str, Any]) -> None:
        """Validate Mask R-CNN specific configuration"""
        required_fields = ["num_classes", "learning_rate", "batch_size"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required Mask R-CNN config field: {field}")

    def _validate_yolo_config(self, config: Dict[str, Any]) -> None:
        """Validate YOLO specific configuration"""
        required_fields = ["model_name", "epochs", "imgsz"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required YOLO config field: {field}")

    def _validate_sam_config(self, config: Dict[str, Any]) -> None:
        """Validate SAM specific configuration"""
        required_fields = ["model_type", "checkpoint_path"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required SAM config field: {field}")


def create_config_from_notebook(
    notebook_type: str, data_root: str, output_dir: str = "./experiment_outputs"
) -> Dict[str, Any]:
    """
    Create configuration exactly matching a specific notebook

    Args:
        notebook_type (str): Type of notebook ("mask_rcnn", "yolo8", "yolo11", "sam")
        data_root (str): Path to dataset root
        output_dir (str): Output directory for experiment

    Returns:
        Dict containing complete experiment configuration
    """
    config_manager = ConfigManager()

    # Get notebook-compatible configuration
    base_config = config_manager.create_notebook_compatible_config(notebook_type)

    # Add experiment-specific settings
    experiment_config = {
        "experiment": ExperimentConfig(
            name=f"surgical_segmentation_{notebook_type}",
            description=f"Experiment following {notebook_type} notebook procedure",
            output_dir=output_dir,
        ),
        "dataset": DatasetConfig(root_dir=data_root),
    }

    # Merge configurations
    final_config = config_manager.merge_configs(base_config, experiment_config)

    return final_config


def save_notebook_compatible_configs(output_dir: str = "./configs") -> None:
    """
    Save notebook-compatible configuration files for all model types

    Args:
        output_dir (str): Directory to save configuration files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_manager = ConfigManager()

    # Create configurations for each notebook
    notebook_types = ["mask_rcnn", "yolo8", "yolo11", "sam"]

    for notebook_type in notebook_types:
        config = config_manager.create_notebook_compatible_config(notebook_type)

        # Save as both JSON and YAML
        json_path = output_dir / f"{notebook_type}_config.json"
        yaml_path = output_dir / f"{notebook_type}_config.yaml"

        config_manager.save_config(config, json_path, "json")
        if yaml_available:
            config_manager.save_config(config, yaml_path, "yaml")

    print(f"Notebook-compatible configurations saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = ConfigManager()

    # Create notebook-compatible configurations
    mask_rcnn_config = create_config_from_notebook(
        "mask_rcnn", "/path/to/coco/dataset", "./mask_rcnn_experiment"
    )

    yolo_config = create_config_from_notebook(
        "yolo11", "/path/to/yolo/dataset", "./yolo11_experiment"
    )

    # Save configurations
    config_manager.save_config(mask_rcnn_config, "./mask_rcnn_config.json")
    config_manager.save_config(yolo_config, "./yolo11_config.yaml")

    # Save all notebook-compatible configs
    save_notebook_compatible_configs("./notebook_configs")

    print("Configuration management ready for use.")
