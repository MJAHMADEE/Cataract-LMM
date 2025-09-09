"""
Configuration Management for Surgical Video Processing

This module provides centralized configuration management for all
video processing operations, with support for YAML configuration files,
environment variable overrides, and validation.

Classes:
    ConfigManager: Central configuration management
    ConfigValidator: Configuration validation and error checking
    EnvironmentConfig: Environment-specific configuration handling
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..compression import CompressionConfig  # Correct reference to compression module
from ..core import ProcessingConfig
from ..deidentification import DeidentificationConfig
from ..pipelines import PipelineConfig
from ..preprocessing import PreprocessingConfig
from ..quality_control import QualityControlConfig

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats"""

    YAML = "yaml"
    JSON = "json"


@dataclass
class MasterConfig:
    """
    Master configuration containing all module configurations

    Attributes:
        processing: General processing configuration
        preprocessing: Video preprocessing configuration
        quality_control: Quality control configuration
        deidentification: De-identification configuration
        compression: Video compression configuration
        pipeline: Pipeline orchestration configuration
        environment: Environment-specific settings
    """

    processing: ProcessingConfig
    preprocessing: PreprocessingConfig
    quality_control: QualityControlConfig
    deidentification: DeidentificationConfig
    compression: CompressionConfig
    pipeline: PipelineConfig
    environment: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "processing": asdict(self.processing),
            "preprocessing": asdict(self.preprocessing),
            "quality_control": asdict(self.quality_control),
            "deidentification": asdict(self.deidentification),
            "compression": asdict(self.compression),
            "pipeline": asdict(self.pipeline),
            "environment": self.environment,
        }


class ConfigManager:
    """
    Central configuration management system

    Provides loading, saving, validation, and environment override
    capabilities for all configuration types used in the framework.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default configurations
        self._default_configs = {
            "processing": ProcessingConfig(),
            "preprocessing": PreprocessingConfig(),
            "quality_control": QualityControlConfig(),
            "deidentification": DeidentificationConfig(),
            "compression": CompressionConfig(),
            "pipeline": PipelineConfig(),
        }

    def load_config(
        self,
        config_name: str = "default",
        config_format: ConfigFormat = ConfigFormat.YAML,
    ) -> MasterConfig:
        """
        Load configuration from file with environment overrides

        Args:
            config_name: Name of the configuration file (without extension)
            config_format: Configuration file format

        Returns:
            MasterConfig object with loaded configuration
        """
        # Determine config file path
        if config_format == ConfigFormat.YAML:
            config_file = self.config_dir / f"{config_name}.yaml"
        else:
            config_file = self.config_dir / f"{config_name}.json"

        # Load base configuration
        if config_file.exists():
            config_data = self._load_config_file(config_file, config_format)
        else:
            self.logger.warning(
                f"Config file not found: {config_file}. Using defaults."
            )
            config_data = self._get_default_config_dict()

        # Apply environment overrides
        config_data = self._apply_environment_overrides(config_data)

        # Validate and create configuration objects
        master_config = self._create_master_config(config_data)

        # Validate configuration
        self._validate_config(master_config)

        self.logger.info(f"Loaded configuration: {config_name}")
        return master_config

    def save_config(
        self,
        config: MasterConfig,
        config_name: str,
        config_format: ConfigFormat = ConfigFormat.YAML,
    ):
        """
        Save configuration to file

        Args:
            config: Configuration object to save
            config_name: Name for the configuration file
            config_format: Format to save in
        """
        config_data = config.to_dict()

        if config_format == ConfigFormat.YAML:
            config_file = self.config_dir / f"{config_name}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_data, f, indent=2, default_flow_style=False)
        else:
            config_file = self.config_dir / f"{config_name}.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

        self.logger.info(f"Saved configuration: {config_file}")

    def create_hospital_config(self, hospital_name: str) -> MasterConfig:
        """
        Create hospital-specific configuration

        Args:
            hospital_name: Name of the hospital (farabi, noor)

        Returns:
            Hospital-specific configuration
        """
        base_config = self.load_config("default")

        if hospital_name.lower() == "farabi":
            # Farabi Hospital (Haag-Streit HS Hi-R NEO 900)
            base_config.preprocessing.target_width = 720
            base_config.preprocessing.target_height = 480
            base_config.preprocessing.target_fps = 30.0
            base_config.preprocessing.brightness_adjustment = 10
            base_config.preprocessing.contrast_adjustment = 1.1
            base_config.preprocessing.reduce_noise = True
            base_config.preprocessing.crop_margins = (10, 10, 10, 10)

            base_config.compression.crf_value = 21  # Slightly higher quality
            base_config.compression.preserve_surgical_detail = True

        elif hospital_name.lower() == "noor":
            # Noor Hospital (ZEISS ARTEVO 800)
            base_config.preprocessing.target_width = 1920
            base_config.preprocessing.target_height = 1080
            base_config.preprocessing.target_fps = 60.0
            base_config.preprocessing.brightness_adjustment = 0
            base_config.preprocessing.contrast_adjustment = 1.0
            base_config.preprocessing.reduce_noise = False
            base_config.preprocessing.crop_margins = (0, 0, 0, 0)

            base_config.compression.crf_value = 20  # High quality for high-res source
            base_config.compression.preserve_surgical_detail = True

        # Save hospital-specific config
        self.save_config(base_config, f"{hospital_name.lower()}_config")

        return base_config

    def create_use_case_configs(self):
        """Create configurations for different use cases"""
        base_config = self.load_config("default")

        # High Quality Configuration
        high_quality_config = MasterConfig(**base_config.__dict__)
        high_quality_config.compression.crf_value = 15
        high_quality_config.compression.compression_speed = "veryslow"
        high_quality_config.compression.preserve_surgical_detail = True
        high_quality_config.preprocessing.enhance_contrast = True
        high_quality_config.preprocessing.reduce_noise = True
        high_quality_config.quality_control.min_overall_score = 80.0
        self.save_config(high_quality_config, "high_quality")

        # Fast Processing Configuration
        fast_config = MasterConfig(**base_config.__dict__)
        fast_config.compression.crf_value = 25
        fast_config.compression.compression_speed = "veryfast"
        fast_config.preprocessing.enhance_contrast = False
        fast_config.preprocessing.reduce_noise = False
        fast_config.quality_control.sample_frame_count = 20
        fast_config.pipeline.parallel_stages = True
        self.save_config(fast_config, "fast_processing")

        # Analysis Preparation Configuration
        analysis_config = MasterConfig(**base_config.__dict__)
        analysis_config.compression.crf_value = 18
        analysis_config.compression.preserve_surgical_detail = True
        analysis_config.preprocessing.standardize_resolution = True
        analysis_config.preprocessing.target_width = 1280
        analysis_config.preprocessing.target_height = 720
        analysis_config.preprocessing.enhance_contrast = True
        analysis_config.quality_control.min_overall_score = 70.0
        analysis_config.deidentification.remove_timestamps = True
        analysis_config.deidentification.remove_watermarks = True
        self.save_config(analysis_config, "analysis_preparation")

        # Archival Configuration
        archival_config = MasterConfig(**base_config.__dict__)
        archival_config.compression.crf_value = 28
        archival_config.compression.compression_speed = "slow"
        archival_config.compression.two_pass_encoding = True
        archival_config.deidentification.remove_metadata = True
        archival_config.deidentification.remove_audio = True
        archival_config.pipeline.backup_originals = False
        archival_config.pipeline.cleanup_intermediates = True
        self.save_config(archival_config, "archival")

    def _load_config_file(
        self, config_file: Path, config_format: ConfigFormat
    ) -> Dict[str, Any]:
        """Load configuration data from file"""
        try:
            with open(config_file, "r") as f:
                if config_format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
            raise

    def _get_default_config_dict(self) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        return {
            "processing": asdict(self._default_configs["processing"]),
            "preprocessing": asdict(self._default_configs["preprocessing"]),
            "quality_control": asdict(self._default_configs["quality_control"]),
            "deidentification": asdict(self._default_configs["deidentification"]),
            "compression": asdict(self._default_configs["compression"]),
            "pipeline": asdict(self._default_configs["pipeline"]),
            "environment": {},
        }

    def _apply_environment_overrides(
        self, config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        # Environment variable prefix
        prefix = "SURGICAL_VIDEO_"

        # Map environment variables to config paths
        env_mappings = {
            f"{prefix}QUALITY_THRESHOLD": "quality_control.min_overall_score",
            f"{prefix}CRF_VALUE": "compression.crf_value",
            f"{prefix}TARGET_WIDTH": "preprocessing.target_width",
            f"{prefix}TARGET_HEIGHT": "preprocessing.target_height",
            f"{prefix}TARGET_FPS": "preprocessing.target_fps",
            f"{prefix}REMOVE_AUDIO": "deidentification.remove_audio",
            f"{prefix}BACKUP_ORIGINALS": "pipeline.backup_originals",
            f"{prefix}PARALLEL_PROCESSING": "processing.parallel_processing",
            f"{prefix}MAX_WORKERS": "processing.max_workers",
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Convert string values to appropriate types
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif "." in value and value.replace(".", "").isdigit():
                    value = float(value)

                # Set value in config data
                self._set_nested_value(config_data, config_path, value)
                self.logger.info(
                    f"Applied environment override: {config_path} = {value}"
                )

        return config_data

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _create_master_config(self, config_data: Dict[str, Any]) -> MasterConfig:
        """Create MasterConfig object from dictionary"""
        try:
            return MasterConfig(
                processing=ProcessingConfig(**config_data.get("processing", {})),
                preprocessing=PreprocessingConfig(
                    **config_data.get("preprocessing", {})
                ),
                quality_control=QualityControlConfig(
                    **config_data.get("quality_control", {})
                ),
                deidentification=DeidentificationConfig(
                    **config_data.get("deidentification", {})
                ),
                compression=CompressionConfig(**config_data.get("compression", {})),
                pipeline=PipelineConfig(**config_data.get("pipeline", {})),
                environment=config_data.get("environment", {}),
            )
        except Exception as e:
            self.logger.error(f"Failed to create configuration objects: {e}")
            raise

    def _validate_config(self, config: MasterConfig):
        """Validate configuration for consistency and correctness"""
        validator = ConfigValidator()

        # Validate individual configurations
        validator.validate_processing_config(config.processing)
        validator.validate_preprocessing_config(config.preprocessing)
        validator.validate_quality_control_config(config.quality_control)
        validator.validate_deidentification_config(config.deidentification)
        validator.validate_compression_config(config.compression)
        validator.validate_pipeline_config(config.pipeline)

        # Cross-validate configurations
        validator.cross_validate_configs(config)


class ConfigValidator:
    """
    Configuration validation and error checking

    Validates individual configuration objects and performs cross-validation
    to ensure consistency between different configuration modules.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_processing_config(self, config: ProcessingConfig):
        """Validate general processing configuration"""
        if config.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if config.compression_crf < 0 or config.compression_crf > 51:
            raise ValueError("compression_crf must be between 0 and 51")

    def validate_preprocessing_config(self, config: PreprocessingConfig):
        """Validate preprocessing configuration"""
        if config.target_width < 100 or config.target_height < 100:
            raise ValueError("Target resolution must be at least 100x100")

        if config.target_fps <= 0:
            raise ValueError("Target FPS must be positive")

        if config.brightness_adjustment < -100 or config.brightness_adjustment > 100:
            raise ValueError("Brightness adjustment must be between -100 and 100")

        if config.contrast_adjustment < 0.1 or config.contrast_adjustment > 5.0:
            raise ValueError("Contrast adjustment must be between 0.1 and 5.0")

    def validate_quality_control_config(self, config: QualityControlConfig):
        """Validate quality control configuration"""
        if config.min_overall_score < 0 or config.min_overall_score > 100:
            raise ValueError("Quality scores must be between 0 and 100")

        if config.sample_frame_count < 1:
            raise ValueError("Sample frame count must be at least 1")

        if config.sample_interval <= 0:
            raise ValueError("Sample interval must be positive")

    def validate_deidentification_config(self, config: DeidentificationConfig):
        """Validate de-identification configuration"""
        if config.blur_strength < 1 or config.blur_strength > 50:
            raise ValueError("Blur strength must be between 1 and 50")

        if len(config.replacement_color) != 3:
            raise ValueError("Replacement color must be RGB tuple (3 values)")

        for color_value in config.replacement_color:
            if color_value < 0 or color_value > 255:
                raise ValueError("Color values must be between 0 and 255")

    def validate_compression_config(self, config: CompressionConfig):
        """Validate compression configuration"""
        if config.target_quality < 0 or config.target_quality > 100:
            raise ValueError("Target quality must be between 0 and 100")

        if config.crf_value < 0 or config.crf_value > 51:
            raise ValueError("CRF value must be between 0 and 51")

        valid_speeds = [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ]
        if config.compression_speed not in valid_speeds:
            raise ValueError(
                f"Invalid compression speed. Must be one of: {valid_speeds}"
            )

        valid_formats = ["mp4", "mkv", "avi"]
        if config.output_format not in valid_formats:
            raise ValueError(f"Invalid output format. Must be one of: {valid_formats}")

    def validate_pipeline_config(self, config: PipelineConfig):
        """Validate pipeline configuration"""
        valid_structures = ["flat", "nested", "dated"]
        if config.output_structure not in valid_structures:
            raise ValueError(
                f"Invalid output structure. Must be one of: {valid_structures}"
            )

    def cross_validate_configs(self, master_config: MasterConfig):
        """Perform cross-validation between different configuration modules"""
        # Validate compression CRF consistency
        if (
            master_config.processing.compression_crf
            != master_config.compression.crf_value
            and master_config.compression.crf_value != 23
        ):  # Default value
            self.logger.warning(
                "CRF values differ between processing and compression configs"
            )

        # Validate resolution consistency
        if (
            master_config.compression.preserve_surgical_detail
            and master_config.compression.crf_value > 25
        ):
            self.logger.warning(
                "High CRF value may not preserve surgical detail adequately"
            )

        # Validate quality control thresholds
        if (
            master_config.quality_control.min_overall_score > 80
            and master_config.pipeline.stop_on_quality_failure
        ):
            self.logger.warning(
                "High quality threshold with strict failure handling may reject many videos"
            )

        # Validate parallel processing settings
        if (
            master_config.pipeline.parallel_stages
            and master_config.processing.max_workers == 1
        ):
            self.logger.warning("Parallel stages enabled but max_workers is 1")
