#!/usr/bin/env python3
"""
Configuration Management for Surgical Phase Recognition

This module provides configuration management utilities including model configs,
training configs, data configs, and experiment configs.

Author: Surgical Phase Recognition Team
Date: August 2025
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    name: str = "swin3d"
    num_classes: int = 11
    input_size: tuple = (224, 224)
    clip_length: int = 16
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    dropout: float = 0.1
    pretrained: bool = True

    # Model-specific parameters
    hidden_size: int = 256  # For RNN models
    bidirectional: bool = True  # For RNN models
    backbone: str = "resnet50"  # For hybrid models
    use_attention: bool = True  # For advanced models


@dataclass
class DataConfig:
    """Data configuration parameters."""

    data_root: str = "/path/to/data"
    train_annotation: str = "train_annotations.csv"
    val_annotation: str = "val_annotations.csv"
    test_annotation: str = "test_annotations.csv"

    # Video preprocessing
    target_fps: Optional[float] = None
    max_frames: Optional[int] = None
    sampling_strategy: str = "uniform"

    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    augment_prob: float = 0.5
    use_advanced_augmentation: bool = True

    # Phase mapping
    phase_mapping: Dict[str, int] = None

    def __post_init__(self):
        if self.phase_mapping is None:
            self.phase_mapping = {
                "Incision": 0,
                "Viscous Agent Injection": 1,
                "Rhexis": 2,
                "Hydrodissection": 3,
                "Phacoemulsification": 4,
                "Irrigation and Aspiration": 5,
                "Capsule Polishing": 6,
                "Lens Implant Setting": 7,
                "Viscous Agent Removal": 8,
                "Suturing": 9,
                "Tonifying Antibiotics": 10,
            }


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001

    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val_f1"
    mode: str = "max"

    # Gradient settings
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1

    # Mixed precision
    precision: str = "16-mixed"

    # Validation
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""

    name: str = "surgical_phase_recognition"
    version: Optional[str] = None
    description: str = ""
    tags: list = None

    # Logging
    log_every_n_steps: int = 50
    log_model: bool = True

    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "surgical-phase-recognition"
    wandb_entity: Optional[str] = None

    # Output directories
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Config:
    """Main configuration class combining all config types."""

    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    experiment: ExperimentConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()


class ConfigManager:
    """
    Configuration manager for loading, saving, and validating configs.

    Supports YAML and JSON formats for configuration files.
    """

    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: Config, filename: str) -> str:
        """
        Save configuration to file.

        Args:
            config (Config): Configuration to save
            filename (str): Output filename

        Returns:
            str: Path to saved config file
        """
        filepath = self.config_dir / filename

        # Convert dataclass to dict
        config_dict = asdict(config)

        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filepath, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif filename.endswith(".json"):
            with open(filepath, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {filename}")

        logger.info(f"Saved configuration to {filepath}")
        return str(filepath)

    def load_config(self, filename: str) -> Config:
        """
        Load configuration from file.

        Args:
            filename (str): Config filename

        Returns:
            Config: Loaded configuration
        """
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
        elif filename.endswith(".json"):
            with open(filepath, "r") as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {filename}")

        # Convert dict to dataclasses
        config = Config(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
        )

        logger.info(f"Loaded configuration from {filepath}")
        return config

    def create_default_configs(self) -> Dict[str, str]:
        """
        Create default configuration files for different scenarios.

        Returns:
            Dict[str, str]: Mapping of scenario names to config file paths
        """
        configs = {}

        # Default config
        default_config = Config()
        configs["default"] = self.save_config(default_config, "default.yaml")

        # Fast training config (for development/testing)
        fast_config = Config()
        fast_config.model.name = "x3d_xs"
        fast_config.model.clip_length = 8
        fast_config.data.batch_size = 16
        fast_config.training.max_epochs = 10
        fast_config.training.learning_rate = 1e-3
        fast_config.experiment.name = "fast_training"
        configs["fast"] = self.save_config(fast_config, "fast_training.yaml")

        # High accuracy config
        high_acc_config = Config()
        high_acc_config.model.name = "swin3d"
        high_acc_config.model.clip_length = 32
        high_acc_config.model.embed_dim = 128
        high_acc_config.data.batch_size = 4
        high_acc_config.training.max_epochs = 200
        high_acc_config.training.learning_rate = 5e-5
        high_acc_config.experiment.name = "high_accuracy"
        configs["high_accuracy"] = self.save_config(
            high_acc_config, "high_accuracy.yaml"
        )

        # CNN-RNN hybrid config
        hybrid_config = Config()
        hybrid_config.model.name = "resnet50_lstm"
        hybrid_config.model.hidden_size = 512
        hybrid_config.model.bidirectional = True
        hybrid_config.data.batch_size = 8
        hybrid_config.training.max_epochs = 150
        hybrid_config.experiment.name = "cnn_rnn_hybrid"
        configs["hybrid"] = self.save_config(hybrid_config, "hybrid.yaml")

        # TeCNO model config
        tecno_config = Config()
        tecno_config.model.name = "tecno"
        tecno_config.model.backbone = "resnet50"
        tecno_config.model.use_attention = True
        tecno_config.model.hidden_size = 512
        tecno_config.data.batch_size = 6
        tecno_config.training.max_epochs = 180
        tecno_config.training.learning_rate = 3e-5
        tecno_config.experiment.name = "tecno_model"
        configs["tecno"] = self.save_config(tecno_config, "tecno.yaml")

        logger.info(f"Created {len(configs)} default configuration files")
        return configs

    def validate_config(self, config: Config) -> Dict[str, Any]:
        """
        Validate configuration for consistency and completeness.

        Args:
            config (Config): Configuration to validate

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Validate model config
        if config.model.num_classes <= 0:
            validation_results["errors"].append("num_classes must be positive")
            validation_results["valid"] = False

        if config.model.clip_length <= 0:
            validation_results["errors"].append("clip_length must be positive")
            validation_results["valid"] = False

        # Validate data config
        if config.data.batch_size <= 0:
            validation_results["errors"].append("batch_size must be positive")
            validation_results["valid"] = False

        if config.data.num_workers < 0:
            validation_results["errors"].append("num_workers must be non-negative")
            validation_results["valid"] = False

        # Validate training config
        if config.training.learning_rate <= 0:
            validation_results["errors"].append("learning_rate must be positive")
            validation_results["valid"] = False

        if config.training.max_epochs <= 0:
            validation_results["errors"].append("max_epochs must be positive")
            validation_results["valid"] = False

        # Warnings for potentially suboptimal settings
        if config.training.learning_rate > 1e-2:
            validation_results["warnings"].append("Learning rate might be too high")

        if config.data.batch_size > 32:
            validation_results["warnings"].append(
                "Large batch size may require significant GPU memory"
            )

        if config.model.clip_length > 64:
            validation_results["warnings"].append(
                "Large clip length may require significant memory"
            )

        return validation_results

    def merge_configs(
        self, base_config: Config, override_config: Dict[str, Any]
    ) -> Config:
        """
        Merge configuration with override parameters.

        Args:
            base_config (Config): Base configuration
            override_config (Dict[str, Any]): Override parameters

        Returns:
            Config: Merged configuration
        """
        # Convert base config to dict
        base_dict = asdict(base_config)

        # Deep merge override config
        def deep_merge(base_dict, override_dict):
            for key, value in override_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_merge(base_dict, override_config)

        # Convert back to Config
        merged_config = Config(
            model=ModelConfig(**base_dict.get("model", {})),
            data=DataConfig(**base_dict.get("data", {})),
            training=TrainingConfig(**base_dict.get("training", {})),
            experiment=ExperimentConfig(**base_dict.get("experiment", {})),
        )

        return merged_config


def load_config_from_args(args) -> Config:
    """
    Load configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Config: Configuration object
    """
    config_manager = ConfigManager()

    if hasattr(args, "config") and args.config:
        # Load from file
        config = config_manager.load_config(args.config)
    else:
        # Create default config
        config = Config()

    # Override with command line arguments
    overrides = {}

    # Model overrides
    if hasattr(args, "model_name") and args.model_name:
        overrides.setdefault("model", {})["name"] = args.model_name

    if hasattr(args, "num_classes") and args.num_classes:
        overrides.setdefault("model", {})["num_classes"] = args.num_classes

    # Training overrides
    if hasattr(args, "learning_rate") and args.learning_rate:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate

    if hasattr(args, "batch_size") and args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size

    if hasattr(args, "max_epochs") and args.max_epochs:
        overrides.setdefault("training", {})["max_epochs"] = args.max_epochs

    # Apply overrides
    if overrides:
        config = config_manager.merge_configs(config, overrides)

    return config


if __name__ == "__main__":
    # Test configuration management
    config_manager = ConfigManager()

    # Create default configs
    default_configs = config_manager.create_default_configs()
    print(f"Created default configs: {list(default_configs.keys())}")

    # Load and validate a config
    config = config_manager.load_config("default.yaml")
    validation_results = config_manager.validate_config(config)

    print(f"Config validation: {'PASSED' if validation_results['valid'] else 'FAILED'}")
    if validation_results["errors"]:
        print(f"Errors: {validation_results['errors']}")
    if validation_results["warnings"]:
        print(f"Warnings: {validation_results['warnings']}")

    print("Configuration management system ready!")
